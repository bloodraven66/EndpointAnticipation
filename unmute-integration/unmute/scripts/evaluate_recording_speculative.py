"""
evaluate_recording_speculative.py

Like evaluate_recording.py, but uses UnmuteHandlerSpeculative which adds
endpoint anticipation. Logs both normal VAD events AND speculation events
so latency comparison is easy.

Usage:
    PYTHONPATH=. python3.12 unmute/scripts/evaluate_recording_speculative.py \\
        /path/to/input.wav /path/to/output.wav

The anticipator server must be running at the address in kyutai_constants.ENDPOINTER_SERVER.
"""
import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import sphn

from unmute.kyutai_constants import SAMPLE_RATE, SAMPLES_PER_FRAME
from unmute.unmute_handler_speculative import UnmuteHandlerSpeculative
import unmute.openai_realtime_api_events as ora


async def main(input_path: Path, output_path: Path, voice: str) -> None:
    print(f"Loading input audio from {input_path}")
    data, _sr = sphn.read(input_path, sample_rate=SAMPLE_RATE)
    if data.ndim > 1:
        data = data.mean(axis=0)

    handler = UnmuteHandlerSpeculative()

    from unmute.llm.system_prompt import BaselineInstructions

    print("Configuring session for evaluation (speculative mode)...")
    await handler.update_session(
        ora.SessionConfig(
            instructions=BaselineInstructions(),
            voice=voice,
            allow_recording=False,
            modalities=["audio", "text"],
        )
    )

    async with handler:
        print("Starting UnmuteHandlerSpeculative (STT + anticipator)...")
        await handler.start_up()

        feed_done = asyncio.Event()

        timed_chunks: list[tuple[float, np.ndarray, int | None, int]] = []
        emitted_order_chunks: list[np.ndarray] = []
        transcription_log: list[str] = []
        response_log: list[str] = []
        cumulative_transcript = ""
        timing_trace: dict[str, Any] = {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "generated_at_unix_sec": time.time(),
            "vad_pause_detection_times_sec": [],
            "generations": [],
            "speculation_attempts": [],
            "alignment": {
                "mode": "generation_anchored_contiguous_with_committed_handoff_wait",
                "anchor_source": "generation_first_audio_chunk_received_sec",
                "preserved_silence_sec": 0.0,
                "real_gap_events": 0,
                "real_gap_sec": 0.0,
                "timestamp_backtrack_events": 0,
                "chunks_placed": 0,
                "first_chunk_anchor_sec": None,
                "final_bot_end_sec": None,
                "generation_regions": 0,
                "committed_handoff_wait_insertions": 0,
                "response_regions": 0,
            },
        }

        def ensure_generation(index_hint: int | None = None) -> tuple[int, dict[str, Any]]:
            generations = timing_trace["generations"]
            assert isinstance(generations, list)
            if index_hint is not None and 0 <= index_hint < len(generations):
                g = generations[index_hint]
                assert isinstance(g, dict)
                return index_hint, g

            g = {
                "generation_index": len(generations),
                "vad_pause_detected_sec": None,
                "start_trigger": "vad_pause",
                "status": "pending",
                "status_reason": None,
                "llm_generation_started_sec": None,
                "llm_first_token_sec": None,
                "tts_generation_started_sec": None,
                "tts_first_token_received_sec": None,
                "tts_first_audio_chunk_received_sec": None,
                "llm_text_generated": "",
                "llm_input_transcript_at_start": None,
            }
            generations.append(g)
            return len(generations) - 1, g

        # ── Timing markers for latency comparison ──────────────────────────
        # We log the same events as evaluate_recording.py so you can diff outputs.

        async def feed_audio() -> None:
            print(
                f"Feeding {len(data) / SAMPLE_RATE:.2f}s of audio in driftless real-time..."
            )
            next_feed_time = time.perf_counter()

            for i in range(0, len(data), SAMPLES_PER_FRAME):
                chunk = data[i : i + SAMPLES_PER_FRAME]
                if len(chunk) < SAMPLES_PER_FRAME:
                    chunk = np.pad(chunk, (0, SAMPLES_PER_FRAME - len(chunk)))

                await handler.receive((SAMPLE_RATE, chunk[np.newaxis, :]))

                next_feed_time += SAMPLES_PER_FRAME / SAMPLE_RATE
                sleep_duration = next_feed_time - time.perf_counter()
                if sleep_duration > 0:
                    await asyncio.sleep(sleep_duration)

            print("Finished feeding input audio.")

            # Feed silence until collector says it's done
            for _ in range(0, int(15 * SAMPLE_RATE), SAMPLES_PER_FRAME):
                if feed_done.is_set():
                    break
                chunk = np.zeros(SAMPLES_PER_FRAME, dtype=np.float32)
                await handler.receive((SAMPLE_RATE, chunk[np.newaxis, :]))

                next_feed_time += SAMPLES_PER_FRAME / SAMPLE_RATE
                sleep_duration = next_feed_time - time.perf_counter()
                if sleep_duration > 0:
                    await asyncio.sleep(sleep_duration)

            if not feed_done.is_set():
                print("Max silence margin reached. Stopping feeder.")
            feed_done.set()

        async def collect_audio() -> None:
            nonlocal cumulative_transcript
            print("Collecting server output...")
            ignore_bot_output = True
            has_seen_first_token = False
            input_end_wall_time: float | None = None
            active_generation_index: int | None = None
            active_response_region_id: int | None = None
            next_response_region_id = 0

            def _mark_tts_first_audio(timestamp: float) -> None:
                nonlocal active_generation_index
                active_generation_index, g = ensure_generation(active_generation_index)
                if g["tts_first_audio_chunk_received_sec"] is None:
                    g["tts_first_audio_chunk_received_sec"] = timestamp

            while True:
                try:
                    output = await asyncio.wait_for(handler.emit(), timeout=0.05)
                except asyncio.TimeoutError:
                    output = None

                if output is None:
                    if feed_done.is_set():
                        if input_end_wall_time is None:
                            input_end_wall_time = time.perf_counter()
                        elif time.perf_counter() - input_end_wall_time > 10.0:
                            print("10s passed since input finished. Stopping collector.")
                            break
                    continue

                # Reset post-input idle timer while we're still receiving
                if input_end_wall_time is not None:
                    input_end_wall_time = time.perf_counter()

                timestamp = handler.audio_received_sec()

                if isinstance(output, tuple):
                    if not ignore_bot_output:
                        _, chunk = output
                        if not timed_chunks:
                            print(f"\n[{timestamp:.2f}s] 🔈 FIRST TTS AUDIO CHUNK RECEIVED!")
                        _mark_tts_first_audio(timestamp)
                        if active_response_region_id is None:
                            active_response_region_id = next_response_region_id
                            next_response_region_id += 1
                        timed_chunks.append(
                            (
                                timestamp,
                                chunk,
                                active_generation_index,
                                active_response_region_id,
                            )
                        )
                        emitted_order_chunks.append(chunk)

                elif isinstance(output, ora.ServerEvent):
                    if output.type == "input_audio_buffer.speech_stopped":
                        ignore_bot_output = False
                        has_seen_first_token = False
                        active_generation_index, g = ensure_generation()
                        g["vad_pause_detected_sec"] = timestamp
                        vad_events = timing_trace["vad_pause_detection_times_sec"]
                        assert isinstance(vad_events, list)
                        vad_events.append(timestamp)
                        print(f"\n[{timestamp:.2f}s] 🛑 VAD PAUSE DETECTED (Speech Stopped)")
                    elif output.type == "response.created":
                        if not ignore_bot_output:
                            active_response_region_id = next_response_region_id
                            next_response_region_id += 1
                            active_generation_index, g = ensure_generation(active_generation_index)
                            if g["llm_generation_started_sec"] is None:
                                g["llm_generation_started_sec"] = timestamp
                                g["tts_generation_started_sec"] = timestamp
                                g["llm_input_transcript_at_start"] = cumulative_transcript
                            g["status"] = "started"
                            g["status_reason"] = "response_created"
                            tag = "🧠 LLM GENERATION STARTED"
                            print(f"[{timestamp:.2f}s] {tag}")
                    elif output.type == "unmute.response.text.delta.ready":
                        if not ignore_bot_output and not has_seen_first_token:
                            active_generation_index, g = ensure_generation(active_generation_index)
                            if g["llm_first_token_sec"] is None:
                                g["llm_first_token_sec"] = timestamp
                            if g["tts_first_token_received_sec"] is None:
                                g["tts_first_token_received_sec"] = timestamp
                            print(f"[{timestamp:.2f}s] ✍️  FIRST TEXT TOKEN FROM vLLM!")
                            has_seen_first_token = True
                        if not ignore_bot_output:
                            active_generation_index, g = ensure_generation(active_generation_index)
                            delta = output.delta
                            if delta:
                                g["llm_text_generated"] += delta
                    elif output.type in {
                        "response.audio.delta",
                        "unmute.response.audio.delta.ready",
                    }:
                        if not ignore_bot_output:
                            _mark_tts_first_audio(timestamp)
                    elif output.type == "conversation.item.input_audio_transcription.delta":
                        transcription_log.append(output.delta)
                        cumulative_transcript += output.delta
                    elif output.type == "response.text.done":
                        if not ignore_bot_output:
                            # Fallback: capture full text when no continuation
                            # tokens were emitted (e.g. committed spec with no
                            # continuation), so feed_done can fire on audio.done.
                            text = getattr(output, "text", "") or ""
                            active_generation_index, g = ensure_generation(active_generation_index)
                            if text.strip() and not g.get("llm_text_generated", "").strip():
                                g["llm_text_generated"] = text
                    elif output.type == "response.text.delta":
                        if not ignore_bot_output:
                            response_log.append(output.delta)
                    elif output.type == "unmute.interrupted_by_vad":
                        if not ignore_bot_output:
                            active_generation_index, g = ensure_generation(active_generation_index)
                            g["status"] = "interrupted"
                            g["status_reason"] = "early_start_from_vad_pause"
                            active_generation_index = None
                            active_response_region_id = None
                    elif output.type == "response.audio.done":
                        if not ignore_bot_output:
                            active_generation_index, g = ensure_generation(active_generation_index)
                            if g["status"] not in {"interrupted", "failed"}:
                                g["status"] = "completed"
                                g["status_reason"] = "response_audio_done"
                            print(f"\n[{timestamp:.2f}s] ✅ Response audio finished.")
                            active_generation_index = None
                            active_response_region_id = None
                            if g.get("llm_text_generated", "").strip() and g.get("vad_pause_detected_sec") is not None:
                                feed_done.set()

            feed_done.set()

        vad_times: list[float] = []
        vad_values: list[float] = []
        ep_times: list[float] = []
        ep_values: list[float] = []

        async def track_signals() -> None:
            """Track STT VAD prediction; anticipator signals are captured in handler trace."""
            last_t = -99.0
            while not feed_done.is_set():
                sim_time = handler.audio_received_sec()

                if handler.stt is not None:
                    curr_t = handler.stt.current_time
                    if curr_t > last_t and curr_t >= 0.0:
                        vad_times.append(sim_time)
                        vad_values.append(handler.stt.pause_prediction.value)
                        last_t = curr_t

                await asyncio.sleep(0.01)

        await asyncio.gather(feed_audio(), collect_audio(), track_signals())

        timing_trace["speculation_attempts"] = handler.get_speculation_trace()

        # ── Render output ───────────────────────────────────────────────────
        print("\nRendering output audio...")
        max_time = len(data) / SAMPLE_RATE
        if timed_chunks:
            last_ts, last_chunk, _, _ = timed_chunks[-1]
            max_time = max(max_time, last_ts + len(last_chunk) / SAMPLE_RATE)

        n_samples = int(np.ceil((max_time + 5.0) * SAMPLE_RATE))
        left_channel = np.zeros(n_samples, dtype=np.float32)
        right_channel = np.zeros(n_samples, dtype=np.float32)
        left_channel[: len(data)] = data

        aligned_preserved_silence_samples = 0
        real_gap_events = 0
        real_gap_sec_total = 0.0
        first_chunk_anchor_sec: float | None = None
        chunks_placed = 0
        prev_end_sec = 0.0
        committed_handoff_wait_insertions = 0

        speculation_attempts = timing_trace.get("speculation_attempts", [])
        committed_attempt_for_region: dict[int, dict[str, Any]] = {}

        regions_by_id: dict[int, dict[str, Any]] = {}
        region_order: list[int] = []
        for ts, chunk, generation_i, response_region_id in timed_chunks:
            if response_region_id not in regions_by_id:
                regions_by_id[response_region_id] = {
                    "response_region_id": response_region_id,
                    "generation_i": generation_i,
                    "chunks": [],
                    "first_ts": float(ts),
                }
                region_order.append(response_region_id)
            regions_by_id[response_region_id]["chunks"].append((ts, chunk, generation_i))

        regions = [regions_by_id[rid] for rid in region_order]

        if isinstance(speculation_attempts, list):
            for attempt in speculation_attempts:
                if not isinstance(attempt, dict):
                    continue
                if attempt.get("status") != "committed":
                    continue
                committed_time = attempt.get("committed_time_sec")
                if committed_time is None:
                    continue

                best_region_id: int | None = None
                best_diff = float("inf")
                for region in regions:
                    first_ts = region.get("first_ts")
                    if first_ts is None:
                        continue
                    diff = abs(float(first_ts) - float(committed_time))
                    if diff < best_diff:
                        best_diff = diff
                        best_region_id = int(region["response_region_id"])

                if best_region_id is not None and best_diff <= 2.0:
                    committed_attempt_for_region.setdefault(best_region_id, attempt)

        for region in regions:
            region_chunks = region["chunks"]
            if not region_chunks:
                continue

            region_id = int(region["response_region_id"])
            region_first_ts = float(region_chunks[0][0])
            if first_chunk_anchor_sec is None:
                first_chunk_anchor_sec = region_first_ts

            region_start_sec = max(region_first_ts, prev_end_sec)
            if region_start_sec > prev_end_sec:
                aligned_preserved_silence_samples += int(
                    round((region_start_sec - prev_end_sec) * SAMPLE_RATE)
                )

            cursor_sec = region_start_sec
            committed_attempt = committed_attempt_for_region.get(region_id)

            committed_prefix_chunks = 0
            continuation_wait_sec = 0.0
            if committed_attempt is not None:
                committed_prefix_chunks = int(
                    committed_attempt.get("committed_audio_chunks_enqueued") or 0
                )
                continuation_wait_sec = float(
                    committed_attempt.get("continuation_wait_after_prefix_sec") or 0.0
                )

            for chunk_i, (_ts, chunk, _generation_i) in enumerate(region_chunks):
                if (
                    committed_attempt is not None
                    and committed_prefix_chunks > 0
                    and continuation_wait_sec > 0.0
                    and chunk_i == committed_prefix_chunks
                ):
                    cursor_sec += continuation_wait_sec
                    real_gap_events += 1
                    real_gap_sec_total += continuation_wait_sec
                    aligned_preserved_silence_samples += int(
                        round(continuation_wait_sec * SAMPLE_RATE)
                    )
                    committed_handoff_wait_insertions += 1

                start_idx = int(round(cursor_sec * SAMPLE_RATE))
                end_idx = start_idx + len(chunk)
                if end_idx > len(right_channel):
                    pad = end_idx - len(right_channel) + SAMPLE_RATE * 5
                    right_channel = np.pad(right_channel, (0, pad))
                    left_channel = np.pad(left_channel, (0, pad))

                right_channel[start_idx:end_idx] = chunk
                cursor_sec += len(chunk) / SAMPLE_RATE
                chunks_placed += 1

            prev_end_sec = cursor_sec

        alignment = timing_trace.get("alignment")
        if isinstance(alignment, dict):
            alignment["preserved_silence_sec"] = (
                aligned_preserved_silence_samples / SAMPLE_RATE
            )
            alignment["real_gap_events"] = real_gap_events
            alignment["real_gap_sec"] = real_gap_sec_total
            alignment["timestamp_backtrack_events"] = 0
            alignment["chunks_placed"] = chunks_placed
            alignment["first_chunk_anchor_sec"] = first_chunk_anchor_sec
            alignment["final_bot_end_sec"] = prev_end_sec if timed_chunks else None
            alignment["generation_regions"] = len(regions)
            alignment["committed_handoff_wait_insertions"] = (
                committed_handoff_wait_insertions
            )
            alignment["response_regions"] = len(regions)

        active = np.where(np.abs(right_channel) + np.abs(left_channel) > 0.0)[0]
        if len(active) > 0:
            final_idx = active[-1] + int(1.0 * SAMPLE_RATE)
            left_channel = left_channel[:final_idx]
            right_channel = right_channel[:final_idx]

        right_channel = np.clip(right_channel, -1.0, 1.0)
        stereo = np.stack([left_channel, right_channel], axis=0)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        stereo_path = output_path.with_name("output.stereo.wav")
        user_path = output_path.with_name("user.wav")
        sphn.write_wav(stereo_path, stereo, SAMPLE_RATE)
        sphn.write_wav(user_path, left_channel[np.newaxis, :], SAMPLE_RATE)
        sphn.write_wav(output_path.with_name("output.wav"), right_channel[np.newaxis, :], SAMPLE_RATE)
        if emitted_order_chunks:
            emitted_order_audio = np.concatenate(emitted_order_chunks)
            emitted_order_path = output_path.with_name("output.emit_order.wav")
            sphn.write_wav(
                emitted_order_path,
                emitted_order_audio[np.newaxis, :],
                SAMPLE_RATE,
            )
            print(f"✓ Saved emit-order bot stream to {emitted_order_path}")
        print(f"✓ Saved stereo output to {stereo_path}")
        print(f"✓ Saved user stream to {user_path}")
        print(f"✓ Saved bot stream to {output_path.with_name('output.wav')}")

        anticipation_signals = handler.get_anticipation_signal_trace()

        ep_times = [
            float(entry["audio_time_sec"])
            for entry in anticipation_signals
            if entry.get("audio_time_sec") is not None
        ]
        ep_values = [
            float(entry["probability"])
            for entry in anticipation_signals
            if entry.get("probability") is not None
        ]

        timings_path = output_path.with_suffix(".timings.json")
        with timings_path.open("w", encoding="utf-8") as f:
            json.dump(timing_trace, f, indent=2)
        print(f"✓ Saved timing trace JSON to {timings_path}")

        print("\n===== Speculation Summary =====")
        attempts = timing_trace["speculation_attempts"]
        if attempts:
            for attempt in attempts:
                attempt_index = attempt.get("attempt_index")
                trigger_time = attempt.get("trigger_time_sec")
                status = attempt.get("status")
                reason = attempt.get("status_reason")
                llm_input = attempt.get("llm_input_transcript") or ""
                llm_output = attempt.get("generated_text") or ""

                print(f"[spec {attempt_index}] detected_at={trigger_time} status={status} reason={reason}")
                print(f"[spec {attempt_index}] llm_input: {llm_input}")
                print(f"[spec {attempt_index}] llm_output: {llm_output}")
        else:
            print("(no speculation attempts recorded)")
        print("===============================")

        # ── Plot ────────────────────────────────────────────────────────────
        try:
            import matplotlib.pyplot as plt
            if vad_times or ep_times:
                time_axis = np.linspace(0, len(left_channel) / SAMPLE_RATE, num=len(left_channel))
                max_plot_sec = 10.0
                plot_xmax = min(max_plot_sec, float(time_axis[-1])) if len(time_axis) > 0 else max_plot_sec

                _, ax1 = plt.subplots(figsize=(9.8, 4.32))

                ax1.plot(time_axis, left_channel, color="blue", alpha=0.35, label="User audio")
                ax1.plot(time_axis, right_channel, color="green", alpha=0.35, label="Bot audio")
                ax1.set_xlabel("Time (s)")
                ax1.set_ylabel("Amplitude")
                ax1.set_xlim(0, plot_xmax)

                from matplotlib.ticker import MultipleLocator, FuncFormatter
                ax1.xaxis.set_major_locator(MultipleLocator(1.0))
                ax1.xaxis.set_minor_locator(MultipleLocator(0.5))
                ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(round(x))}"))
                ax1.grid(which="major", axis="x", linestyle="-", linewidth=1.2, color="gray", alpha=0.6)
                ax1.grid(which="minor", axis="x", linestyle="--", linewidth=0.6, color="gray", alpha=0.4)
                ax1.grid(which="major", axis="y", alpha=0.2)

                ax2 = ax1.twinx()
                ax2.plot(vad_times, vad_values, color="red", linewidth=1.8, label="VAD pause pred.")
                if ep_times:
                    ax2.plot(ep_times, ep_values, color="orange", linewidth=1.8,
                             linestyle="--", label="Anticipator prob.")
                ax2.set_ylabel("Probability", color="dimgray")
                ax2.set_ylim(-0.05, 1.05)

                # VAD crossings
                crossings = [vad_times[i] for i in range(1, len(vad_values))
                             if vad_values[i-1] < 0.6 and vad_values[i] >= 0.6]
                for i, t in enumerate(crossings):
                    ax2.axvline(x=t, color="darkred", linestyle="-.", linewidth=1.5,
                                alpha=0.8, label="VAD trigger" if i == 0 else None)

                # Anticipator crossings (anticipation fires)
                from unmute.unmute_handler_speculative import ANTICIPATE_THRESHOLD
                if ep_times and ep_values:
                    ep_crossings = [ep_times[i] for i in range(1, len(ep_values))
                                    if ep_values[i-1] < ANTICIPATE_THRESHOLD
                                    and ep_values[i] >= ANTICIPATE_THRESHOLD]
                    for i, t in enumerate(ep_crossings):
                        ax2.axvline(x=t, color="darkorange", linestyle=":", linewidth=1.5,
                                    alpha=0.9, label="Anticipation fire" if i == 0 else None)

                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

                plt.title("Speculative Generation — Audio & VAD/Anticipator Signals")
                plt.tight_layout()

                plot_path = output_path.with_suffix(".vad_plot.png")
                plt.savefig(plot_path)
                print(f"✓ Saved plot to {plot_path}")
        except ImportError:
            print("⚠ matplotlib not installed, skipping plot.")

        print("\n===== Evaluation Summary =====")
        print("Transcription:", "".join(transcription_log))
        print("Response:     ", "".join(response_log))
        print("==============================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate UnmuteHandlerSpeculative with endpoint anticipation."
    )
    parser.add_argument("input_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument(
        "--voice",
        type=str,
        default="unmute-prod-website/ex04_narration_longform_00001.wav",
    )
    args = parser.parse_args()
    asyncio.run(main(args.input_path, args.output_path, args.voice))
