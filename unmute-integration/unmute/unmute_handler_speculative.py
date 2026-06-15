"""
UnmuteHandlerSpeculative: Extends UnmuteHandler with endpoint anticipation.

Design:
  - The endpoint anticipator (port 8093) streams `user_end_probability` at 12.5Hz.
  - When probability crosses ANTICIPATE_THRESHOLD, we speculatively start LLM+TTS
    generation and buffer the audio.
  - If the real VAD fires within ANTICIPATE_WINDOW_SEC (960ms), we commit the buffer
    to the output queue and continue generation from the already-generated prefix.
  - If the window expires without a real VAD, we discard the speculative generation.
"""
import asyncio
import time
from dataclasses import dataclass, field
from functools import partial
from logging import getLogger
from typing import Any

import numpy as np
import websockets

from unmute.endpointer import Endpointer
from unmute.kyutai_constants import SAMPLE_RATE, SAMPLES_PER_FRAME
from unmute.llm.llm_utils import (
    VLLMStream,
    rechunk_to_words,
)
from unmute.quest_manager import Quest
from unmute.service_discovery import find_instance
from unmute.tts.text_to_speech import TextToSpeech, TTSAudioMessage, TTSClientEosMessage, TTSTextMessage
from unmute.unmute_handler import UnmuteHandler
import unmute.openai_realtime_api_events as ora

logger = getLogger(__name__)

# ─── Tunable constants ────────────────────────────────────────────────────────
# Threshold for triggering speculative generation. The anticipator model outputs
# a probability that the user will finish speaking within the next 960ms.
ANTICIPATE_THRESHOLD: float = 0.5

# How long we wait for the real VAD to confirm before stopping active
# speculative generation. Buffered audio can still be committed later.
# Should match the anticipator's lookahead window (12 Mimi frames × 80ms = 960ms).
ANTICIPATE_WINDOW_SEC: float = 0.960

# Minimum gap between two speculative starts. Keep this aligned with the
# anticipation window so a trigger is skipped for one full speculation window.
ANTICIPATE_COOLDOWN_SEC: float = ANTICIPATE_WINDOW_SEC

# After commit, speculative TTS may still append chunks for a short period.
# We must drain the committed speculative prefix fully before continuation starts.
# Keep a generous hard timeout only as a last-resort safeguard against hangs.
SPEC_COMMIT_DRAIN_MAX_SEC: float = 30.0

# Anticipator server expects 960-sample cadence (40ms @ 24kHz).
ANTICIPATOR_FRAME_SAMPLES: int = 960
ANTICIPATOR_RECONNECT_BASE_SEC: float = 0.5
ANTICIPATOR_RECONNECT_MAX_SEC: float = 4.0

# Diagnostics threshold for treating an audio chunk as effectively silent.
AUDIO_SILENCE_RMS_THRESHOLD: float = 1e-4

# Guarded perceptual-tail trim for committed speculative prefix.
COMMITTED_TAIL_TRIM_RMS_THRESHOLD: float = 5e-4
COMMITTED_TAIL_TRIM_MIN_CONSECUTIVE_CHUNKS: int = 6
COMMITTED_TAIL_TRIM_MIN_EMITTED_SEC: float = 2.0
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class SpeculativeState:
    """Holds everything produced by a single speculative generation attempt."""
    audio_chunks: list[np.ndarray] = field(default_factory=list)
    text_tokens: list[str] = field(default_factory=list)
    started_at: float = field(default_factory=time.perf_counter)
    task: asyncio.Task | None = None  # the running _speculative_generation_task
    tts_connection: TextToSpeech | None = None  # direct TTS connection (not via quest)
    # Set when TTS is kept open after token-limit stop, for seamless continuation.
    # None in all other cases (window_expired, natural completion, discarded).
    tts_receive_task: asyncio.Task | None = None
    committed: bool = False
    discarded: bool = False
    window_expired: bool = False
    # When set, the task stops LLM generation (via window_expired) but continues
    # to drain TTS so that already-queued text produces complete audio.  The
    # finished audio is then saved as _fallback_spec instead of being thrown away.
    graceful_supersede: bool = False
    trace_index: int | None = None
    llm_naturally_completed: bool = False
    stopped_by_token_limit: bool = False

    @property
    def text_so_far(self) -> str:
        return "".join(self.text_tokens)

    def age_sec(self) -> float:
        return time.perf_counter() - self.started_at


class UnmuteHandlerSpeculative(UnmuteHandler):
    """
    Subclass of UnmuteHandler that adds speculative LLM+TTS generation driven
    by an endpoint anticipator model.

    Override summary:
      start_up()        — also starts the endpointer quest
      receive()         — also forwards audio to the endpointer
      _generate_response() — checks for buffered speculative audio before generating
    """

    def __init__(self) -> None:
        super().__init__()
        self._speculative: SpeculativeState | None = None
        self._last_anticipation_time: float = 0.0
        # Lock to serialise commit/discard transitions
        self._spec_lock = asyncio.Lock()
        self.speculation_trace: list[dict[str, Any]] = []
        self.anticipation_signal_trace: list[dict[str, Any]] = []
        # Most recently discarded spec that had buffered audio. Used as fallback
        # at VAD when the current spec fired too close to VAD to produce audio.
        self._fallback_spec: SpeculativeState | None = None
        # Task for an in-flight graceful TTS drain (set when graceful_supersede is
        # initiated, cleared when drain completes in _speculative_generation_task).
        # Waited on at VAD so a completed drain can set _fallback_spec in time.
        self._graceful_drain_task: asyncio.Task | None = None
        # Set when spec is committed early (at VAD, before STT flush).
        # _generate_response() (called after flush) adds the assistant placeholder
        # and fires this event so the continuation LLM can start.
        self._early_committed: bool = False
        self._stt_flush_event: asyncio.Event | None = None

    def _prediction_audio_time_sec(self, frame_count: int | None) -> float:
        # Anticipator outputs one probability per 960-sample frame (40ms @ 24kHz).
        if frame_count is None:
            return self.audio_received_sec()
        return float(frame_count) * (960.0 / SAMPLE_RATE)

    def _record_anticipation_signal(self, **entry: Any) -> None:
        self.anticipation_signal_trace.append(entry)

    def _update_spec_trace(self, index: int | None, **updates: Any) -> None:
        if index is None:
            return
        if 0 <= index < len(self.speculation_trace):
            self.speculation_trace[index].update(updates)

    def get_speculation_trace(self) -> list[dict[str, Any]]:
        """Return a JSON-serializable copy of speculative attempt traces."""
        return [dict(entry) for entry in self.speculation_trace]

    def get_anticipation_signal_trace(self) -> list[dict[str, Any]]:
        """Return a JSON-serializable copy of anticipator signal/policy decisions."""
        return [dict(entry) for entry in self.anticipation_signal_trace]

    def _get_speculative_messages(self) -> list[dict[str, Any]]:
        """Return the messages snapshot to use for speculative LLM generation.

        Subclasses can override to inject speculative-mode instructions
        without mutating the real chat history.
        """
        return self.chatbot.preprocessed_messages()

    async def _pre_continuation_messages_hook(self) -> None:
        """Called just before the continuation LLM reads preprocessed_messages().

        Subclasses can override to ensure any async work completes before
        the message snapshot is taken.
        """

    def _spec_continuation_needed(self, committed_state: SpeculativeState) -> bool:
        """Whether to run the continuation LLM after committing the spec prefix.

        Always True: even when the spec LLM naturally completed its sentence, the
        continuation LLM may extend the response. The prefix has its trailing
        punctuation stripped before being passed to the continuation LLM so the
        model treats it as an unfinished thought and continues.
        Subclasses can override for specialised behaviour.
        """
        return True

    # ── Endpointer quest ────────────────────────────────────────────────────

    async def start_up_endpointer(self) -> None:
        async def _init() -> Endpointer:
            ep = Endpointer()
            await ep.start_up()
            return ep

        async def _run(ep: Endpointer) -> None:
            await self._endpointer_loop(ep)

        async def _close(ep: Endpointer) -> None:
            await ep.shutdown()

        quest = await self.quest_manager.add(Quest("endpointer", _init, _run, _close))
        # Wait for the connection to be established before returning
        await quest.get()
        logger.info("Endpointer started.")

    async def start_up(self) -> None:
        """Start STT (parent) and then the endpointer."""
        await super().start_up()  # starts STT
        try:
            await self.start_up_endpointer()
        except Exception as exc:
            logger.warning(
                "Endpointer failed to start (%r). Continuing without anticipation.", exc
            )

    # ── Endpointer prediction loop ──────────────────────────────────────────

    async def _endpointer_loop(self, ep: Endpointer) -> None:
        """Consume predictions from the endpointer and fire speculation.

        Keeps a long-lived stream and attempts reconnects on unexpected disconnects.
        """
        reconnect_delay = ANTICIPATOR_RECONNECT_BASE_SEC

        while True:
            try:
                async for msg in ep:
                    prob = msg.user_end_probability
                    logger.debug("Endpointer prob=%.3f", prob)
                    now = time.perf_counter()
                    conversation_state = self.chatbot.conversation_state()
                    audio_time_sec = self._prediction_audio_time_sec(msg.frame_count)
                    cooldown_elapsed = now - self._last_anticipation_time
                    cooldown_remaining_sec = max(0.0, ANTICIPATE_COOLDOWN_SEC - cooldown_elapsed)

                    allow_by_threshold = prob >= ANTICIPATE_THRESHOLD
                    allow_by_cooldown = cooldown_elapsed >= ANTICIPATE_COOLDOWN_SEC
                    allow_by_state = conversation_state == "user_speaking"
                    triggered = allow_by_threshold and allow_by_cooldown and allow_by_state

                    policy_reasons: list[str] = []
                    if not allow_by_threshold:
                        policy_reasons.append("below_threshold")
                    if not allow_by_cooldown:
                        policy_reasons.append("cooldown")
                    if not allow_by_state:
                        policy_reasons.append("blocked_by_conversation_state")

                    self._record_anticipation_signal(
                        audio_time_sec=audio_time_sec,
                        frame_count=msg.frame_count,
                        probability=float(prob),
                        threshold=ANTICIPATE_THRESHOLD,
                        conversation_state=conversation_state,
                        cooldown_remaining_sec=float(cooldown_remaining_sec),
                        policy_triggered=triggered,
                        policy_drop_reasons=policy_reasons,
                    )

                    if not triggered:
                        continue

                    logger.info("🔮 Anticipation fired (prob=%.2f). Starting speculation.", prob)
                    self._last_anticipation_time = now
                    await self._start_speculation(prob, trigger_audio_time_sec=audio_time_sec)

                logger.warning(
                    "Endpointer prediction stream ended; attempting reconnect in %.2fs.",
                    reconnect_delay,
                )
            except asyncio.CancelledError:
                raise
            except websockets.exceptions.ConnectionClosed as exc:
                logger.warning(
                    "Endpointer connection closed (%s); reconnect in %.2fs.",
                    repr(exc),
                    reconnect_delay,
                )
            except Exception as exc:
                logger.warning(
                    "Endpointer loop error (%r); reconnect in %.2fs.",
                    exc,
                    reconnect_delay,
                )

            await asyncio.sleep(reconnect_delay)
            try:
                await ep.start_up()
                reconnect_delay = ANTICIPATOR_RECONNECT_BASE_SEC
                logger.info("Endpointer reconnected.")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Endpointer reconnect failed: %r", exc)
                reconnect_delay = min(
                    ANTICIPATOR_RECONNECT_MAX_SEC,
                    reconnect_delay * 2,
                )

    # ── Speculative generation ───────────────────────────────────────────────

    async def _start_speculation(self, prob: float, trigger_audio_time_sec: float | None = None) -> None:
        """Cancel any existing speculation and start a new one."""
        async with self._spec_lock:
            await self._cancel_speculation(reason="superseded")

            state = SpeculativeState()
            trace_index = len(self.speculation_trace)
            state.trace_index = trace_index
            self.speculation_trace.append(
                {
                    "attempt_index": trace_index,
                    "trigger_time_sec": (
                        trigger_audio_time_sec
                        if trigger_audio_time_sec is not None
                        else self.audio_received_sec()
                    ),
                    "trigger_probability": float(prob),
                    "status": "started",
                    "status_reason": None,
                    "llm_started_sec": None,
                    "llm_first_token_sec": None,
                    "llm_input_transcript": None,
                    "tts_started_sec": None,
                    "tts_first_audio_sec": None,
                    "committed_time_sec": None,
                    "ended_time_sec": None,
                    "generated_text": "",
                    "generated_word_count": 0,
                    "window_expired_buffer_kept": False,
                    "committed_from_window_expired_buffer": False,
                    "committed_audio_drain_timed_out": False,
                    "committed_audio_forwarded_duration_sec": 0.0,
                    "committed_tail_trim_triggered": False,
                    "committed_tail_trimmed_sec": 0.0,
                    "committed_prefix_drain_finished_sec": None,
                    "llm_naturally_completed": False,
                    "continuation_llm_first_token_sec": None,
                    "continuation_tts_first_audio_buffered_sec": None,
                    "continuation_queue_first_audio_forwarded_sec": None,
                    "continuation_wait_after_prefix_sec": None,
                    "continuation_drain_finished_sec": None,
                    "continuation_producer_done_sec": None,
                    "continuation_producer_error": None,
                    "error": None,
                }
            )
            self._speculative = state

        # Launch the generation task outside the lock so it can acquire it
        state.task = asyncio.create_task(
            self._speculative_generation_task(state),
            name="speculative_generation",
        )

    async def _cancel_speculation(self, reason: str = "cancelled") -> None:
        """Cancel and clean up the current speculative state (must hold _spec_lock)."""
        state = self._speculative
        if state is None or state.committed or state.discarded:
            return

        state.discarded = True

        if state.tts_receive_task is not None:
            # TTS was kept open for seamless continuation but this spec is being
            # discarded — cancel the receive task and close the connection.
            # The gen task has already exited, so no graceful drain is possible.
            if not state.tts_receive_task.done():
                state.tts_receive_task.cancel()
            state.tts_receive_task = None
            if state.tts_connection is not None:
                tts_conn = state.tts_connection
                state.tts_connection = None
                asyncio.create_task(tts_conn.shutdown())
            if len(state.audio_chunks) > 0:
                self._fallback_spec = state
            self._update_spec_trace(
                state.trace_index,
                status="discarded",
                status_reason=reason,
                ended_time_sec=self.audio_received_sec(),
                generated_text=state.text_so_far,
                generated_word_count=len(state.text_so_far.split()),
            )
            logger.info("❌ Speculation DISCARDED (tts-reuse, %s, age=%.0f ms)", reason, state.age_sec() * 1000)

        elif reason == "superseded" and not state.graceful_supersede:
            # Graceful stop: signal the task to stop generating new LLM tokens but
            # let TTS finish encoding whatever text it already received.  The task
            # will drain TTS, accumulate complete audio, then save itself as
            # _fallback_spec so VAD can play it if the next spec has no audio.
            # Cancel any in-flight graceful drain — only the most recent spec's
            # audio is worth saving; letting multiple drains run concurrently
            # saturates the TTS server and slows down the current spec's TTS.
            old_drain = self._graceful_drain_task
            if old_drain is not None and not old_drain.done():
                old_drain.cancel()
            self._graceful_drain_task = None

            state.window_expired = True   # breaks the LLM token loop
            state.graceful_supersede = True
            self._graceful_drain_task = state.task  # track for VAD fallback wait
            # Do NOT cancel the task or shut down TTS — let it finish naturally.
            logger.info(
                "↩️  Speculation GRACEFUL-STOP (superseded, age=%.0f ms, %d chunks so far)",
                state.age_sec() * 1000, len(state.audio_chunks),
            )
        else:
            # Hard cancel: VAD with empty buffer, or another discard reason.
            # Only save partial audio as fallback if there is any — but don't clear
            # an existing fallback just because this spec had an empty buffer.
            if len(state.audio_chunks) > 0:
                self._fallback_spec = state

            self._update_spec_trace(
                state.trace_index,
                status="discarded",
                status_reason=reason,
                ended_time_sec=self.audio_received_sec(),
                generated_text=state.text_so_far,
                generated_word_count=len(state.text_so_far.split()),
            )
            logger.info("❌ Speculation DISCARDED (%s, age=%.0f ms)", reason, state.age_sec() * 1000)

            if state.task and not state.task.done():
                state.task.cancel()

            if state.tts_connection is not None:
                try:
                    await state.tts_connection.shutdown()
                except Exception:
                    pass
                state.tts_connection = None

        self._speculative = None

    async def _speculative_generation_task(self, state: SpeculativeState) -> None:
        """
        Run LLM+TTS speculatively. Audio is placed in state.audio_chunks rather
        than the main output_queue. Text tokens go into state.text_tokens.

                The task is cancelled/stopped if:
                    (a) ANTICIPATE_WINDOW_SEC passes without a real VAD → generation stops,
                            buffered audio is kept for possible later VAD commit
          (b) Real VAD fires → committed, task continues until TTS EOS
          (c) New speculation fires while this one is active → discarded
        """
        try:
            self._update_spec_trace(
                state.trace_index,
                llm_started_sec=self.audio_received_sec(),
            )

            # Take a snapshot of the conversation at this point in time
            messages = self._get_speculative_messages()
            user_transcript = " ".join(
                m.get("content", "")
                for m in messages
                if m.get("role") == "user"
            ).strip()
            self._update_spec_trace(
                state.trace_index,
                llm_input_transcript=user_transcript,
            )

            # Connect to TTS speculatively (separate from the real TTS quest)
            tts = await find_instance(
                "tts",
                partial(
                    TextToSpeech,
                    recorder=None,          # don't record speculative audio
                    get_time=self.audio_received_sec,
                    voice=self.tts_voice,
                ),
            )
            state.tts_connection = tts

            llm = VLLMStream(self.openai_client, temperature=0.3)

            # Start the TTS receive loop as a background task writing into state.audio_chunks
            tts_receive_task = asyncio.create_task(
                self._speculative_tts_receive(tts, state),
                name="speculative_tts_receive",
            )

            # Watchdog: Stop generating more words after the window expires (to save resources)
            # but KEEP the buffer around until real VAD or a new anticipation.
            async def _watchdog():
                await asyncio.sleep(ANTICIPATE_WINDOW_SEC)
                if self._speculative is state and not state.committed:
                    logger.info("🕒 Speculation window expired (%.0f ms). Stopping generation but keeping buffer.", ANTICIPATE_WINDOW_SEC * 1000)
                    # We don't cancel the whole task, just let the LLM loop finish its current word.
                    state.window_expired = True  # will break the llm loop below
                    self._update_spec_trace(
                        state.trace_index,
                        status="buffered",
                        status_reason="window_expired_buffer_kept",
                        window_expired_buffer_kept=True,
                        ended_time_sec=self.audio_received_sec(),
                    )

            watchdog_task = asyncio.create_task(_watchdog(), name="spec_watchdog")

            try:
                # Generate at most a few words speculatively (roughly 20 tokens)
                token_limit = 20
                tokens_generated = 0
                async for delta in rechunk_to_words(llm.chat_completion(messages)):
                    if state.discarded or state.committed or state.window_expired:
                        break
                    if tokens_generated == 0:
                        self._update_spec_trace(
                            state.trace_index,
                            llm_first_token_sec=self.audio_received_sec(),
                        )
                    state.text_tokens.append(delta)
                    if tokens_generated == 0:
                        self._update_spec_trace(
                            state.trace_index,
                            tts_started_sec=self.audio_received_sec(),
                        )
                    await tts.send(delta)
                    tokens_generated += 1
                    if tokens_generated >= token_limit:
                        state.stopped_by_token_limit = True
                        break
                else:
                    state.llm_naturally_completed = True
                    self._update_spec_trace(
                        state.trace_index,
                        llm_naturally_completed=True,
                    )

                if state.stopped_by_token_limit and not state.discarded and not state.graceful_supersede:
                    # Keep TTS open — continuation will send remaining tokens on the
                    # same connection so audio is acoustically seamless.
                    state.tts_receive_task = tts_receive_task
                elif not state.discarded or state.graceful_supersede:
                    await tts.send(TTSClientEosMessage())

            finally:
                watchdog_task.cancel()

            # Wait for TTS to drain unless we're handing it off to continuation.
            if state.tts_receive_task is None and (not state.discarded or state.graceful_supersede):
                await tts_receive_task

        except asyncio.CancelledError:
            logger.debug("Speculative generation task cancelled.")
            self._update_spec_trace(
                state.trace_index,
                status="cancelled",
                status_reason="task_cancelled",
                ended_time_sec=self.audio_received_sec(),
                generated_text=state.text_so_far,
                generated_word_count=len(state.text_so_far.split()),

            )
        except Exception as exc:
            logger.warning("Speculative generation failed: %r", exc)
            self._update_spec_trace(
                state.trace_index,
                status="failed",
                status_reason="exception",
                ended_time_sec=self.audio_received_sec(),
                generated_text=state.text_so_far,
                generated_word_count=len(state.text_so_far.split()),

                error=repr(exc),
            )
        finally:
            if state.committed:
                self._update_spec_trace(
                    state.trace_index,
                    status="committed",
                    status_reason="vad_confirmed",
                    generated_text=state.text_so_far,
                    generated_word_count=len(state.text_so_far.split()),
    
                )
            elif state.graceful_supersede:
                # TTS fully drained — save complete audio as fallback for VAD.
                self._fallback_spec = state
                self._graceful_drain_task = None
                logger.info(
                    "↩️  Graceful supersede complete — saved %d chunks as fallback (text=%r)",
                    len(state.audio_chunks), state.text_so_far,
                )
                self._update_spec_trace(
                    state.trace_index,
                    status="cancelled",
                    status_reason="graceful_supersede_fallback_saved",
                    ended_time_sec=self.audio_received_sec(),
                    generated_text=state.text_so_far,
                    generated_word_count=len(state.text_so_far.split()),
    
                )
            elif state.window_expired:
                self._update_spec_trace(
                    state.trace_index,
                    status="buffered",
                    status_reason="window_expired_buffer_kept",
                    ended_time_sec=self.audio_received_sec(),
                    generated_text=state.text_so_far,
                    generated_word_count=len(state.text_so_far.split()),
    
                    window_expired_buffer_kept=True,
                )
            elif not state.discarded:
                self._update_spec_trace(
                    state.trace_index,
                    status="completed",
                    status_reason="finished_without_commit",
                    ended_time_sec=self.audio_received_sec(),
                    generated_text=state.text_so_far,
                    generated_word_count=len(state.text_so_far.split()),
    
                )

            if state.tts_connection is not None and not state.committed and state.tts_receive_task is None:
                try:
                    await state.tts_connection.shutdown()
                except Exception:
                    pass
                state.tts_connection = None

    async def _speculative_tts_receive(
        self, tts: TextToSpeech, state: SpeculativeState
    ) -> None:
        """Collect TTS audio from the speculative TTS into state.audio_chunks."""
        try:
            async for message in tts:
                if state.discarded and not state.graceful_supersede:
                    break
                if isinstance(message, TTSAudioMessage):
                    audio = np.array(message.pcm, dtype=np.float32)
                    if len(state.audio_chunks) == 0:
                        self._update_spec_trace(
                            state.trace_index,
                            tts_first_audio_sec=self.audio_received_sec(),
                        )
                    state.audio_chunks.append(audio)
                    logger.debug(
                        "Spec TTS: buffered %.0f ms audio (total chunks: %d)",
                        len(audio) / SAMPLE_RATE * 1000,
                        len(state.audio_chunks),
                    )
                elif isinstance(message, TTSTextMessage):
                    # Already accumulated via LLM stream; ignore duplicates here
                    pass
        except Exception as exc:
            if not state.discarded:
                logger.warning("Speculative TTS receive error: %r", exc)

    # ── Audio forwarding to endpointer ──────────────────────────────────────

    async def receive(self, frame: tuple[int, np.ndarray]) -> None:
        """Forward audio to the endpointer in addition to parent processing."""
        # Forward to endpointer first (non-blocking — endpointer is on a different server)
        ep_quest = self.quest_manager.quests.get("endpointer")
        if ep_quest is not None:
            ep: Endpointer | None = ep_quest.get_nowait()
            if ep is not None:
                array = frame[1][0]  # mono
                try:
                    for i in range(0, len(array), ANTICIPATOR_FRAME_SAMPLES):
                        ep_chunk = array[i : i + ANTICIPATOR_FRAME_SAMPLES]
                        if len(ep_chunk) < ANTICIPATOR_FRAME_SAMPLES:
                            ep_chunk = np.pad(
                                ep_chunk,
                                (0, ANTICIPATOR_FRAME_SAMPLES - len(ep_chunk)),
                            )
                        await ep.send_audio(ep_chunk)
                except Exception as exc:
                    logger.debug("Endpointer send_audio failed: %r", exc)

        # Normal STT/VAD processing via parent
        await super().receive(frame)

    # ── Commit speculation on real VAD ──────────────────────────────────────

    async def _on_vad_pause(self) -> None:
        """Commit spec audio immediately at VAD — before STT flush starts.

        The spec audio is already buffered so we can start playing it right away.
        The continuation LLM still needs the final transcript, so it waits for
        an event that _generate_response() fires after the flush completes.
        """
        async with self._spec_lock:
            state = self._speculative
            if state is None or state.discarded or len(state.audio_chunks) == 0:
                return

            state.committed = True
            self._fallback_spec = None
            self._update_spec_trace(
                state.trace_index,
                status="committed",
                status_reason=(
                    "vad_confirmed_from_expired_buffer"
                    if state.window_expired
                    else "vad_confirmed"
                ),
                committed_time_sec=self.audio_received_sec(),
                generated_text=state.text_so_far,
                generated_word_count=len(state.text_so_far.split()),
                committed_from_window_expired_buffer=state.window_expired,
            )
            self._speculative = None
            committed_state = state

        # generating_message_i anticipates the assistant placeholder that
        # _generate_response() will add after STT flush.  STT flush only
        # appends tokens to the existing user message (same role → no new
        # list entry), so len(chat_history) stays the same until the placeholder.
        generating_message_i = len(self.chatbot.chat_history) + 1
        stt_flush_event = asyncio.Event()
        self._stt_flush_event = stt_flush_event
        self._early_committed = True

        age_ms = committed_state.age_sec() * 1000
        logger.info(
            "⚡ Spec EARLY-COMMIT at VAD%s (age=%.0f ms, %d chunks, %d tokens) — "
            "audio drain starts immediately, continuation waits for STT flush",
            " [expired-buffer]" if committed_state.window_expired else "",
            age_ms, len(committed_state.audio_chunks), len(committed_state.text_tokens),
        )

        quest = Quest.from_run_step(
            "llm",
            lambda: self._committed_speculative_response_task(
                committed_state,
                generating_message_i,
                stt_flush_event=stt_flush_event,
            ),
        )
        await self.quest_manager.add(quest)

    async def _generate_response(self) -> None:
        """
        Called when the real VAD fires (after STT flush).

        If we have a speculative state with buffered audio, drain it into the
        output queue first, then continue generation from the already-generated
        text prefix. Otherwise fall back to normal generation.
        """
        # Fast path: spec was early-committed at VAD time (_on_vad_pause).
        # The drain/continuation task is already running; we just need to:
        #   1. Add the assistant placeholder (safe now — STT flush is done)
        #   2. Fire the event so the continuation LLM can call preprocessed_messages()
        if self._early_committed:
            self._early_committed = False
            stt_flush_event = self._stt_flush_event
            self._stt_flush_event = None

            # If the quest was cancelled by an interruption before flush completed,
            # the continuation never needs to run — fall through to normal gen.
            if "llm" not in self.quest_manager.quests:
                logger.info("Early-committed spec quest was interrupted before STT flush. Normal gen.")
                await super()._generate_response()
                return

            await self.add_chat_message_delta("", "assistant")
            if stt_flush_event is not None:
                stt_flush_event.set()
            return

        async with self._spec_lock:
            state = self._speculative

            if state is None or state.discarded or len(state.audio_chunks) == 0:
                # No useful speculation — cancel any pending speculative task and
                # fall through to normal generation.
                if state is not None and not state.discarded:
                    await self._cancel_speculation(reason="VAD with empty buffer")
                self._speculative = None

                # If a graceful TTS drain is still in flight, wait briefly for it
                # to complete and set _fallback_spec before we decide to fall back
                # to normal generation. TTS drain typically takes <400ms wall-clock.
                drain_task = self._graceful_drain_task
                if drain_task is not None and not drain_task.done() and self._fallback_spec is None:
                    try:
                        await asyncio.wait_for(asyncio.shield(drain_task), timeout=0.5)
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        logger.info("Graceful drain wait timed out at VAD — using normal generation.")

                # Try fallback: the most recently discarded spec that had buffered
                # audio. Fires when the last spec triggered too close to VAD to
                # produce audio of its own (all attempts show "cancelled" in trace).
                fallback = self._fallback_spec
                self._fallback_spec = None
                if fallback is not None and len(fallback.audio_chunks) > 0:
                    fallback.committed = True
                    self._update_spec_trace(
                        fallback.trace_index,
                        status="committed",
                        status_reason="vad_confirmed_from_fallback",
                        committed_time_sec=self.audio_received_sec(),
                        committed_from_window_expired_buffer=fallback.window_expired,
                    )
                    committed_state = fallback
                    logger.info(
                        "✅ Speculation COMMITTED [fallback] (%d audio chunks, %d text tokens)",
                        len(committed_state.audio_chunks), len(committed_state.text_tokens),
                    )
                else:
                    logger.info("No speculative audio to commit. Using normal generation.")
                    await super()._generate_response()
                    return
            else:
                # Mark as committed — the receive task and watchdog will check this flag
                state.committed = True
                self._fallback_spec = None  # clear fallback — primary spec is committing
                self._update_spec_trace(
                    state.trace_index,
                    status="committed",
                    status_reason=(
                        "vad_confirmed_from_expired_buffer"
                        if state.window_expired
                        else "vad_confirmed"
                    ),
                    committed_time_sec=self.audio_received_sec(),
                    generated_text=state.text_so_far,
                    generated_word_count=len(state.text_so_far.split()),
    
                    committed_from_window_expired_buffer=state.window_expired,
                )
                self._speculative = None
                committed_state = state

        age_ms = committed_state.age_sec() * 1000
        n_chunks = len(committed_state.audio_chunks)
        logger.info(
            "✅ Speculation COMMITTED%s (age=%.0f ms, %d audio chunks, %d text tokens)",
            " [expired-buffer]" if committed_state.window_expired else "",
            age_ms, n_chunks, len(committed_state.text_tokens),
        )

        # ── Step 1: emit the speculative audio buffer ──────────────────────
        # Empty assistant entry to signal start of response
        await self.add_chat_message_delta("", "assistant")
        generating_message_i = len(self.chatbot.chat_history)
        quest = Quest.from_run_step(
            "llm",
            lambda: self._committed_speculative_response_task(
                committed_state,
                generating_message_i,
            ),
        )
        await self.quest_manager.add(quest)

    async def _committed_speculative_response_task(
        self,
        committed_state: SpeculativeState,
        generating_message_i: int,
        stt_flush_event: asyncio.Event | None = None,
    ) -> None:
        """
        Emit committed speculative audio, then continue normal generation.

        Runs as the "llm" quest so baseline interruption semantics apply:
        `interrupt_bot()` can cancel this task via quest_manager.remove("llm").

        stt_flush_event: if set, the continuation LLM waits for this before
        calling preprocessed_messages(). Set by _generate_response() after the
        STT flush completes. None in the normal (post-flush commit) path.
        """
        await self.output_queue.put(
            ora.ResponseCreated(
                response=ora.Response(
                    status="in_progress",
                    voice=self.tts_voice or "missing",
                    chat_history=self.chatbot.chat_history,
                )
            )
        )

        prefix_text = committed_state.text_so_far
        # Strip trailing sentence-final punctuation from the prefix used in chat
        # For token-limited specs (mid-sentence cuts), strip trailing punctuation so
        # continue_final_message=True treats the prefix as an unfinished thought.
        # For naturally-completed specs, keep the full text — stripping "?" from
        # "Did you see that?" produces "Did you see that", the model then just adds
        # "?" and EOSes immediately, making the continuation a single character.
        prefix_for_continuation = prefix_text.rstrip()
        if committed_state.stopped_by_token_limit and prefix_for_continuation.endswith(('.', '!', '?')):
            prefix_for_continuation = prefix_for_continuation[:-1]
        continuation_needed = self._spec_continuation_needed(committed_state)

        # Reuse the open speculative TTS connection when the spec hit its token
        # limit — the connection carries the AR hidden state so continuation
        # audio is acoustically seamless with the already-buffered prefix.
        reuse_tts = (
            continuation_needed
            and committed_state.tts_receive_task is not None
            and not committed_state.tts_receive_task.done()
        )

        continuation_audio_queue: asyncio.Queue[np.ndarray | None] | None = None
        continuation_task: asyncio.Task[None] | None = None

        if continuation_needed:
            # When stt_flush_event is None the STT flush is already done and the
            # assistant placeholder exists — patch it now so preprocessed_messages()
            # sees the spec prefix.  When stt_flush_event is set we're in the
            # early-commit path: the placeholder doesn't exist yet (STT flush is
            # still in progress), so the patch is deferred into the continuation
            # task itself (it runs after stt_flush_event fires).
            if stt_flush_event is None and prefix_for_continuation:
                for msg in reversed(self.chatbot.chat_history):
                    if msg["role"] == "assistant" and msg["content"] == "":
                        msg["content"] = prefix_for_continuation
                        break

            if reuse_tts:
                continuation_task = asyncio.create_task(
                    self._continuation_on_reused_tts(
                        committed_state,
                        generating_message_i,
                        prefix_for_continuation,
                        committed_state.trace_index,
                        stt_flush_event=stt_flush_event,
                    ),
                    name="spec_continuation_reused_tts",
                )
            else:
                continuation_audio_queue = asyncio.Queue()
                continuation_task = asyncio.create_task(
                    self._continuation_audio_producer_task(
                        generating_message_i,
                        prefix_for_continuation,
                        continuation_audio_queue,
                        committed_state.trace_index,
                        stt_flush_event=stt_flush_event,
                    ),
                    name="spec_continuation_audio_producer",
                )

        # Emit speculative buffer first. If interrupted, stop immediately.
        # NOTE: speculative receiver can still append chunks briefly after commit.
        # We continue draining until producer is done and all chunks are emitted.
        drain_index = 0
        drain_started_at = time.perf_counter()
        committed_chunks_enqueued = 0
        committed_samples_enqueued = 0
        tail_trim_triggered = False
        active_trim_run = False
        tail_trimmed_chunks = 0
        tail_trimmed_samples = 0
        pending_quiet_chunks: list[np.ndarray] = []
        pending_quiet_samples = 0
        drain_timed_out = False

        async def _emit_committed_chunk(chunk: np.ndarray, chunk_rms: float) -> None:
            nonlocal committed_chunks_enqueued
            nonlocal committed_samples_enqueued
            await self.output_queue.put((SAMPLE_RATE, chunk))
            committed_chunks_enqueued += 1
            committed_samples_enqueued += len(chunk)

        while True:
            if len(self.chatbot.chat_history) > generating_message_i:
                if continuation_task is not None:
                    continuation_task.cancel()
                if reuse_tts:
                    if committed_state.tts_receive_task is not None and not committed_state.tts_receive_task.done():
                        committed_state.tts_receive_task.cancel()
                    if committed_state.tts_connection is not None:
                        asyncio.create_task(committed_state.tts_connection.shutdown())
                self._update_spec_trace(
                    committed_state.trace_index,
                    status="interrupted",
                    status_reason="interrupted_while_draining_committed_audio",
                )
                return

            while drain_index < len(committed_state.audio_chunks):
                chunk = committed_state.audio_chunks[drain_index]
                chunk_rms = float(np.sqrt(np.mean(np.square(chunk)))) if len(chunk) else 0.0

                continuation_ready = (
                    continuation_audio_queue is not None
                    and continuation_audio_queue.qsize() > 0
                )
                emitted_sec = committed_samples_enqueued / SAMPLE_RATE
                low_energy = chunk_rms <= COMMITTED_TAIL_TRIM_RMS_THRESHOLD

                if (
                    continuation_ready
                    and emitted_sec >= COMMITTED_TAIL_TRIM_MIN_EMITTED_SEC
                    and low_energy
                ):
                    pending_quiet_chunks.append(chunk)
                    pending_quiet_samples += len(chunk)
                    if (
                        len(pending_quiet_chunks)
                        >= COMMITTED_TAIL_TRIM_MIN_CONSECUTIVE_CHUNKS
                    ):
                        active_trim_run = True
                        tail_trim_triggered = True
                    # Else: keep buffering quiet candidates until we know it's a run.
                else:
                    if pending_quiet_chunks:
                        if active_trim_run:
                            # We were in a trim run, but speech resumed.
                            # Drop buffered quiet run and continue normally.
                            tail_trimmed_chunks += len(pending_quiet_chunks)
                            tail_trimmed_samples += pending_quiet_samples
                        else:
                            # Quiet run too short: keep it.
                            for quiet_chunk in pending_quiet_chunks:
                                quiet_rms = (
                                    float(np.sqrt(np.mean(np.square(quiet_chunk))))
                                    if len(quiet_chunk)
                                    else 0.0
                                )
                                await _emit_committed_chunk(quiet_chunk, quiet_rms)
                        pending_quiet_chunks = []
                        pending_quiet_samples = 0
                        active_trim_run = False

                    await _emit_committed_chunk(chunk, chunk_rms)

                drain_index += 1

            if reuse_tts:
                producer_done = (
                    committed_state.tts_receive_task is None
                    or committed_state.tts_receive_task.done()
                )
            else:
                producer_done = (
                    committed_state.task is None or committed_state.task.done()
                )
            if producer_done and pending_quiet_chunks:
                if active_trim_run:
                    # Ended while trimming: drop trailing low-energy run.
                    tail_trimmed_chunks += len(pending_quiet_chunks)
                    tail_trimmed_samples += pending_quiet_samples
                else:
                    # Run was too short for trim: preserve it.
                    for quiet_chunk in pending_quiet_chunks:
                        quiet_rms = (
                            float(np.sqrt(np.mean(np.square(quiet_chunk))))
                            if len(quiet_chunk)
                            else 0.0
                        )
                        await _emit_committed_chunk(quiet_chunk, quiet_rms)
                pending_quiet_chunks = []
                pending_quiet_samples = 0
                active_trim_run = False

            if producer_done and drain_index >= len(committed_state.audio_chunks):
                break

            now = time.perf_counter()
            if now - drain_started_at >= SPEC_COMMIT_DRAIN_MAX_SEC:
                drain_timed_out = True
                logger.warning(
                    "Committed speculative drain hit hard timeout after %.2fs "
                    "(enqueued=%d, buffered_now=%d, producer_done=%s).",
                    SPEC_COMMIT_DRAIN_MAX_SEC,
                    committed_chunks_enqueued,
                    len(committed_state.audio_chunks),
                    producer_done,
                )
                break

            await asyncio.sleep(0.01)

        self._update_spec_trace(
            committed_state.trace_index,
            committed_audio_drain_timed_out=drain_timed_out,
            committed_audio_forwarded_duration_sec=(
                committed_samples_enqueued / SAMPLE_RATE
            ),
            committed_tail_trim_triggered=tail_trim_triggered,
            committed_tail_trimmed_sec=(tail_trimmed_samples / SAMPLE_RATE),
            committed_prefix_drain_finished_sec=self.audio_received_sec(),
        )

        if len(self.chatbot.chat_history) > generating_message_i:
            if continuation_task is not None:
                continuation_task.cancel()
            if reuse_tts:
                if committed_state.tts_receive_task is not None and not committed_state.tts_receive_task.done():
                    committed_state.tts_receive_task.cancel()
                if committed_state.tts_connection is not None:
                    asyncio.create_task(committed_state.tts_connection.shutdown())
            return

        if not continuation_needed:
            await self.output_queue.put(ora.ResponseTextDone(text=prefix_text))
            await self.output_queue.put(
                (SAMPLE_RATE, np.zeros(SAMPLES_PER_FRAME, dtype=np.float32))
            )
            await self.output_queue.put(self.get_gradio_update())
            await self.output_queue.put(ora.ResponseAudioDone())
            await self.add_chat_message_delta("", "user")
            await asyncio.sleep(1)
            await self.check_for_bot_goodbye()
            self.waiting_for_user_start_time = self.audio_received_sec()
            return

        if not reuse_tts:
            # New TTS connection: drain audio from the continuation queue.
            assert continuation_audio_queue is not None
            assert continuation_task is not None
            await self._drain_continuation_audio_queue(
                continuation_audio_queue,
                continuation_task,
                generating_message_i,
                committed_state.trace_index,
                self.audio_received_sec(),
            )
            if len(self.chatbot.chat_history) > generating_message_i:
                return
        else:
            # Reused TTS: audio flowed directly into committed_state.audio_chunks
            # and was drained by the loop above. Clean up TTS connection now.
            try:
                await committed_state.tts_connection.shutdown()
            except Exception:
                pass
            committed_state.tts_connection = None
            self._update_spec_trace(
                committed_state.trace_index,
                continuation_wait_after_prefix_sec=0.0,
                continuation_drain_finished_sec=self.audio_received_sec(),
            )

        await self.output_queue.put(
            (SAMPLE_RATE, np.zeros(SAMPLES_PER_FRAME, dtype=np.float32))
        )
        await self.output_queue.put(self.get_gradio_update())
        await self.output_queue.put(ora.ResponseAudioDone())
        await self.add_chat_message_delta("", "user")
        await asyncio.sleep(1)
        await self.check_for_bot_goodbye()
        self.waiting_for_user_start_time = self.audio_received_sec()

    async def _continuation_on_reused_tts(
        self,
        committed_state: SpeculativeState,
        generating_message_i: int,
        prefix_text: str,
        trace_index: int | None,
        stt_flush_event: asyncio.Event | None = None,
    ) -> None:
        """Send continuation LLM tokens to the already-open speculative TTS.

        Audio flows through the still-running tts_receive_task into
        committed_state.audio_chunks, which the drain loop reads directly.
        No separate audio queue is needed.
        """
        tts = committed_state.tts_connection
        assert tts is not None
        continuation_tokens: list[str] = []
        try:
            if stt_flush_event is not None:
                await stt_flush_event.wait()
                # Patch assistant placeholder now that STT flush is done
                if prefix_text:
                    for msg in reversed(self.chatbot.chat_history):
                        if msg["role"] == "assistant" and msg["content"] == "":
                            msg["content"] = prefix_text
                            break
            await self._pre_continuation_messages_hook()
            messages = self.chatbot.preprocessed_messages()
            llm = VLLMStream(
                self.openai_client,
                temperature=0.3,
                continue_final_message=True,
            )
            interrupted = False
            saw_first_token = False
            async for delta in rechunk_to_words(llm.chat_completion(messages)):
                if len(self.chatbot.chat_history) > generating_message_i:
                    interrupted = True
                    break
                if not saw_first_token:
                    saw_first_token = True
                    self._update_spec_trace(
                        trace_index,
                        continuation_llm_first_token_sec=self.audio_received_sec(),
                    )
                await self.output_queue.put(ora.UnmuteResponseTextDeltaReady(delta=delta))
                continuation_tokens.append(delta)
                await self.add_chat_message_delta(
                    delta,
                    "assistant",
                    generating_message_i=generating_message_i,
                )
                await tts.send(delta)

            if not interrupted:
                full_text = prefix_text + "".join(continuation_tokens)
                await self.output_queue.put(ora.ResponseTextDone(text=full_text))
                await tts.send(TTSClientEosMessage())
        except Exception as exc:
            self._update_spec_trace(trace_index, continuation_producer_error=repr(exc))
            raise
        finally:
            self._update_spec_trace(
                trace_index,
                continuation_producer_done_sec=self.audio_received_sec(),
            )

    async def _continuation_audio_producer_task(
        self,
        generating_message_i: int,
        prefix_text: str,
        audio_queue: asyncio.Queue[np.ndarray | None],
        trace_index: int | None,
        stt_flush_event: asyncio.Event | None = None,
    ) -> None:
        """Produce continuation audio into a private queue while prefix audio plays."""
        tts = await find_instance(
            "tts",
            partial(
                TextToSpeech,
                recorder=self.recorder,
                get_time=self.audio_received_sec,
                voice=self.tts_voice,
            ),
        )
        async def _recv_tts_audio() -> None:
            first_audio_logged = False
            async for message in tts:
                if len(self.chatbot.chat_history) > generating_message_i:
                    break
                if isinstance(message, TTSAudioMessage):
                    audio = np.array(message.pcm, dtype=np.float32)
                    if not first_audio_logged:
                        self._update_spec_trace(
                            trace_index,
                            continuation_tts_first_audio_buffered_sec=self.audio_received_sec(),
                        )
                        first_audio_logged = True
                    await audio_queue.put(audio)
                elif isinstance(message, TTSTextMessage):
                    await self.output_queue.put(ora.ResponseTextDelta(delta=message.text))

        recv_task = asyncio.create_task(_recv_tts_audio(), name="spec_continuation_tts_recv")
        continuation_tokens: list[str] = []

        try:
            if stt_flush_event is not None:
                await stt_flush_event.wait()
                # Patch assistant placeholder now that STT flush is done
                if prefix_text:
                    for msg in reversed(self.chatbot.chat_history):
                        if msg["role"] == "assistant" and msg["content"] == "":
                            msg["content"] = prefix_text
                            break
            await self._pre_continuation_messages_hook()
            messages = self.chatbot.preprocessed_messages()
            llm = VLLMStream(
                self.openai_client,
                temperature=0.3,
                continue_final_message=True,
            )

            interrupted = False
            saw_first_token = False
            async for delta in rechunk_to_words(llm.chat_completion(messages)):
                if len(self.chatbot.chat_history) > generating_message_i:
                    interrupted = True
                    break

                if not saw_first_token:
                    saw_first_token = True
                    self._update_spec_trace(
                        trace_index,
                        continuation_llm_first_token_sec=self.audio_received_sec(),
                    )

                await self.output_queue.put(ora.UnmuteResponseTextDeltaReady(delta=delta))
                continuation_tokens.append(delta)
                await self.add_chat_message_delta(
                    delta,
                    "assistant",
                    generating_message_i=generating_message_i,
                )
                await tts.send(delta)

            if not interrupted:
                full_text = prefix_text + "".join(continuation_tokens)
                await self.output_queue.put(ora.ResponseTextDone(text=full_text))
                await tts.send(TTSClientEosMessage())

            await recv_task
        except Exception as exc:
            self._update_spec_trace(
                trace_index,
                continuation_producer_error=repr(exc),
            )
            raise
        finally:
            recv_task.cancel()
            try:
                await tts.shutdown()
            except Exception:
                pass
            self._update_spec_trace(
                trace_index,
                continuation_producer_done_sec=self.audio_received_sec(),
            )
            await audio_queue.put(None)

    async def _drain_continuation_audio_queue(
        self,
        audio_queue: asyncio.Queue[np.ndarray | None],
        continuation_task: asyncio.Task[None],
        generating_message_i: int,
        trace_index: int | None,
        prefix_drain_finished_sec: float,
    ) -> None:
        """Drain continuation audio in order after committed speculative prefix."""
        first_forwarded_sec: float | None = None
        while True:
            if len(self.chatbot.chat_history) > generating_message_i:
                continuation_task.cancel()
                return

            try:
                item = await asyncio.wait_for(audio_queue.get(), timeout=0.05)
            except asyncio.TimeoutError:
                if continuation_task.done() and audio_queue.empty():
                    break
                continue

            if item is None:
                break

            now_sec = self.audio_received_sec()
            if first_forwarded_sec is None:
                first_forwarded_sec = now_sec
                self._update_spec_trace(
                    trace_index,
                    continuation_queue_first_audio_forwarded_sec=first_forwarded_sec,
                    continuation_wait_after_prefix_sec=(
                        first_forwarded_sec - prefix_drain_finished_sec
                    ),
                )

            await self.output_queue.put((SAMPLE_RATE, item))


        if continuation_task.done():
            exc = continuation_task.exception()
            if exc is not None:
                self._update_spec_trace(
                    trace_index,
                    continuation_producer_error=repr(exc),
                )
                raise exc
