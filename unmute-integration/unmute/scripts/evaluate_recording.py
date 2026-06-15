import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import sphn

from fastrtc import CloseStream
from unmute.kyutai_constants import SAMPLE_RATE, SAMPLES_PER_FRAME
from unmute.unmute_handler import UnmuteHandler
import unmute.openai_realtime_api_events as ora


async def main(input_path: Path, output_path: Path, voice: str):
    print(f"Loading input audio from {input_path}")
    data, _sr = sphn.read(input_path, sample_rate=SAMPLE_RATE)
    if data.ndim > 1:
        data = data.mean(axis=0)  # Convert to mono if needed

    # The UnmuteHandler processes logic natively
    handler = UnmuteHandler()

    from unmute.llm.system_prompt import BaselineInstructions

    print("Configuring session for evaluation...")
    await handler.update_session(
        ora.SessionConfig(
            instructions=BaselineInstructions(),
            voice=voice,
            allow_recording=False,
            modalities=["audio", "text"],
        )
    )

    async with handler:
        print("Starting UnmuteHandler...")
        await handler.start_up()

        start_time = time.perf_counter()
        feed_done = asyncio.Event()

        timed_chunks = []
        transcription_log = []
        response_log = []
        cumulative_transcript = ""
        timing_trace: dict[str, Any] = {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "generated_at_unix_sec": time.time(),
            "vad_pause_detection_times_sec": [],
            "generations": [],
        }

        def ensure_generation(index_hint: int | None = None) -> tuple[int, dict[str, Any]]:
            generations = timing_trace["generations"]
            assert isinstance(generations, list)
            if index_hint is not None and 0 <= index_hint < len(generations):
                gen = generations[index_hint]
                assert isinstance(gen, dict)
                return index_hint, gen
            gen = {
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
            generations.append(gen)
            return len(generations) - 1, gen

        async def feed_audio():
            print(f"Feeding audio ({len(data) / SAMPLE_RATE:.2f}s) in driftless simulated real-time...")
            next_feed_time = time.perf_counter()

            for i in range(0, len(data), SAMPLES_PER_FRAME):
                chunk = data[i : i + SAMPLES_PER_FRAME]
                if len(chunk) < SAMPLES_PER_FRAME:
                    chunk = np.pad(chunk, (0, SAMPLES_PER_FRAME - len(chunk)))

                await handler.receive((SAMPLE_RATE, chunk[np.newaxis, :]))

                # Calculate mathematically perfect driftless sleep to keep it 1:1 real-time
                next_feed_time += SAMPLES_PER_FRAME / SAMPLE_RATE
                sleep_duration = next_feed_time - time.perf_counter()
                if sleep_duration > 0:
                    await asyncio.sleep(sleep_duration)

            print("Finished feeding input audio.")

            # Flush
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
                print("Max silence margin reached. Stopping.")
            
            feed_done.set()

        async def collect_audio():
            nonlocal cumulative_transcript
            print("Collecting server output...")
            ignore_bot_output = True
            has_seen_first_token = False
            input_end_wall_time: float | None = None
            active_generation_index: int | None = None

            def _mark_tts_first_audio(timestamp: float) -> None:
                nonlocal active_generation_index
                active_generation_index, gen = ensure_generation(active_generation_index)
                if gen["tts_first_audio_chunk_received_sec"] is None:
                    gen["tts_first_audio_chunk_received_sec"] = timestamp

            while True:
                # handler.emit() calls wait_for_item() which is a BLOCKING await —
                # it suspends until an item arrives in the output queue. We need
                # asyncio.wait_for() to periodically escape that block so we can
                # check if it's time to terminate (e.g. after TTS is fully done and
                # nothing more will arrive).
                #
                # The original bug was in the TimeoutError handler: it measured time
                # from session start instead of from when input feeding finished, so
                # the guard could fire too early or too late.
                try:
                    output = await asyncio.wait_for(handler.emit(), timeout=0.05)
                except asyncio.TimeoutError:
                    output = None

                if output is None:
                    # Check if we should terminate: only start countdown once
                    # feed_audio signals it has finished via feed_done.
                    if feed_done.is_set():
                        if input_end_wall_time is None:
                            input_end_wall_time = time.perf_counter()
                        elif time.perf_counter() - input_end_wall_time > 10.0:
                            print("10s passed since input finished. Stopping collector.")
                            break
                    continue

                # Got a real output — reset the post-input idle timer so we don't
                # terminate while TTS is still streaming audio.
                if input_end_wall_time is not None:
                    input_end_wall_time = time.perf_counter()

                # Use the handler's audio clock for timestamps (stays coherent with
                # the simulation even when processing faster/slower than real-time)
                timestamp = handler.audio_received_sec()

                if isinstance(output, tuple):
                    if not ignore_bot_output:
                        sr, chunk = output
                        if not timed_chunks:
                            print(f"\n[{timestamp:.2f}s] 🔈 FIRST TTS AUDIO CHUNK RECEIVED!")
                        _mark_tts_first_audio(timestamp)
                        timed_chunks.append((timestamp, chunk))

                elif isinstance(output, ora.ServerEvent):
                    if output.type == "input_audio_buffer.speech_stopped":
                        ignore_bot_output = False
                        has_seen_first_token = False
                        active_generation_index, gen = ensure_generation()
                        gen["vad_pause_detected_sec"] = timestamp
                        vad_events = timing_trace["vad_pause_detection_times_sec"]
                        assert isinstance(vad_events, list)
                        vad_events.append(timestamp)
                        print(f"\n[{timestamp:.2f}s] 🛑 VAD PAUSE DETECTED (Speech Stopped)")
                    elif output.type == "response.created":
                        if not ignore_bot_output:
                            active_generation_index, gen = ensure_generation(active_generation_index)
                            if gen["llm_generation_started_sec"] is None:
                                gen["llm_generation_started_sec"] = timestamp
                                gen["tts_generation_started_sec"] = timestamp
                                gen["llm_input_transcript_at_start"] = cumulative_transcript
                            gen["status"] = "started"
                            gen["status_reason"] = "response_created"
                            print(f"[{timestamp:.2f}s] 🧠 LLM GENERATION STARTED")
                    elif output.type == "unmute.response.text.delta.ready":
                        if not ignore_bot_output and not has_seen_first_token:
                            active_generation_index, gen = ensure_generation(active_generation_index)
                            if gen["llm_first_token_sec"] is None:
                                gen["llm_first_token_sec"] = timestamp
                            if gen["tts_first_token_received_sec"] is None:
                                gen["tts_first_token_received_sec"] = timestamp
                            print(f"[{timestamp:.2f}s] ✍️ FIRST TEXT TOKEN FROM vLLM!")
                            has_seen_first_token = True
                        if not ignore_bot_output:
                            active_generation_index, gen = ensure_generation(active_generation_index)
                            delta = output.delta
                            if delta:
                                gen["llm_text_generated"] += delta
                    elif output.type in {
                        "response.audio.delta",
                        "unmute.response.audio.delta.ready",
                    }:
                        if not ignore_bot_output:
                            _mark_tts_first_audio(timestamp)
                    elif output.type == "conversation.item.input_audio_transcription.delta":
                        transcription_log.append(output.delta)
                        cumulative_transcript += output.delta
                    elif output.type == "response.text.delta":
                        if not ignore_bot_output:
                            response_log.append(output.delta)
                    elif output.type == "unmute.interrupted_by_vad":
                        if not ignore_bot_output:
                            active_generation_index, gen = ensure_generation(active_generation_index)
                            gen["status"] = "interrupted"
                            gen["status_reason"] = "early_start_from_vad_pause"
                            active_generation_index = None
                    elif output.type == "response.audio.done":
                        if not ignore_bot_output:
                            active_generation_index, gen = ensure_generation(active_generation_index)
                            if gen["status"] not in {"interrupted", "failed"}:
                                gen["status"] = "completed"
                                gen["status_reason"] = "response_audio_done"
                            print(f"\n[{timestamp:.2f}s] ✅ Response audio finished.")
                            active_generation_index = None
                            if gen.get("llm_text_generated", "").strip():
                                feed_done.set()

            # feed_done.set()


        vad_times = []
        vad_values = []

        async def track_vad():
            last_t = -99.0
            while not feed_done.is_set():
                if handler.stt is not None:
                    curr_t = handler.stt.current_time
                    if curr_t > last_t and curr_t >= 0.0:
                        # Append the actual simulation clock time (shows real-time latency offset)
                        sim_time = handler.audio_received_sec()
                        vad_times.append(sim_time)
                        vad_values.append(handler.stt.pause_prediction.value)
                        last_t = curr_t
                await asyncio.sleep(0.01)

        # Run feeder, collector, and VAD tracker concurrently
        await asyncio.gather(feed_audio(), collect_audio(), track_vad())

        # Process the chunks into a smooth aligned track
        print("\nRendering stereo aligned output...")
        
        # Determine the necessary length for output buffers
        max_time = len(data) / SAMPLE_RATE
        if timed_chunks:
            last_timestamp, last_chunk = timed_chunks[-1]
            max_time = max(max_time, last_timestamp + (len(last_chunk) / SAMPLE_RATE))

        output_samples = int(np.ceil((max_time + 5.0) * SAMPLE_RATE)) # small 5s margin buffer
        left_channel = np.zeros(output_samples, dtype=np.float32)
        right_channel = np.zeros(output_samples, dtype=np.float32)

        # 1. Place the original input on the Left Channel
        left_channel[: len(data)] = data

        # 2. Reconstruct the bot audio on the Right Channel
        # We track current_cursor to avoid small jitter gaps within an ongoing utterance burst.
        current_cursor = 0.0
        for ts, chunk in timed_chunks:
            # If the timestamp is significantly far from the cursor, it's a new burst of speech
            if ts - current_cursor > 0.5:
                current_cursor = ts

            start_idx = int(current_cursor * SAMPLE_RATE)
            end_idx = start_idx + len(chunk)

            # Safeguard array bounds
            if end_idx > len(right_channel):
                pad_size = end_idx - len(right_channel) + (SAMPLE_RATE * 5)
                right_channel = np.pad(right_channel, (0, pad_size))
                left_channel = np.pad(left_channel, (0, pad_size))

            right_channel[start_idx:end_idx] += chunk
            
            # Advance the cursor smoothly by exactly the chunk duration
            current_cursor += len(chunk) / SAMPLE_RATE

        # Trim exact zeroes at the end to keep the file short
        active_indices = np.where(np.abs(right_channel) + np.abs(left_channel) > 0.0)[0]
        if len(active_indices) > 0:
            final_idx = active_indices[-1] + int(1.0 * SAMPLE_RATE)  # keep 1 sec after
            left_channel = left_channel[:final_idx]
            right_channel = right_channel[:final_idx]

        # Prevent digital clipping
        right_channel = np.clip(right_channel, -1.0, 1.0)

        # Merge stereo [L, R]
        stereo = np.stack([left_channel, right_channel], axis=0)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save stereo (both channels), user (left/input), and system (right/bot) separately
        stereo_path = output_path.with_name("output.stereo.wav")
        user_path = output_path.with_name("user.wav")
        sphn.write_wav(stereo_path, stereo, SAMPLE_RATE)
        sphn.write_wav(user_path, left_channel[np.newaxis, :], SAMPLE_RATE)
        sphn.write_wav(output_path.with_name("output.wav"), right_channel[np.newaxis, :], SAMPLE_RATE)
        print(f"✓ Saved stereo output to {stereo_path}")
        print(f"✓ Saved user stream to {user_path}")
        print(f"✓ Saved bot stream to {output_path.with_name('output.wav')}")

        timings_path = output_path.with_suffix(".timings.json")
        with timings_path.open("w", encoding="utf-8") as f:
            json.dump(timing_trace, f, indent=2)
        print(f"✓ Saved timing trace JSON to {timings_path}")


        try:
            import matplotlib.pyplot as plt
            if vad_times:
                times = vad_times
                vad = vad_values

                time_axis = np.linspace(0, len(left_channel) / SAMPLE_RATE, num=len(left_channel))
                max_plot_sec = 10.0
                plot_xmax = min(max_plot_sec, float(time_axis[-1])) if len(time_axis) > 0 else max_plot_sec

                _, ax1 = plt.subplots(figsize=(9.8, 4.32))

                # Left Y axis for waveforms
                ax1.plot(time_axis, left_channel, color="blue", alpha=0.4, label="Input Audio (User)")
                ax1.plot(time_axis, right_channel, color="green", alpha=0.4, label="Output Audio (Bot)")
                ax1.set_xlabel("Time (s)")
                ax1.set_ylabel("Audio Amplitude", color="black")
                ax1.tick_params(axis="y", labelcolor="black")
                ax1.set_xlim(0, plot_xmax)

                from matplotlib.ticker import MultipleLocator, FuncFormatter
                
                # Finer X-axis Configuration
                ax1.xaxis.set_major_locator(MultipleLocator(1.0))
                ax1.xaxis.set_minor_locator(MultipleLocator(0.5))
                ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(round(x))}"))
                ax1.tick_params(axis='x', which='major', labelsize=10)
                
                # Grid configuration (major and minor vertical lines)
                ax1.grid(which='major', axis='x', linestyle='-', linewidth=1.2, color='gray', alpha=0.6)
                ax1.grid(which='minor', axis='x', linestyle='--', linewidth=0.6, color='gray', alpha=0.4)
                ax1.grid(which='major', axis='y', alpha=0.2)

                # Right Y axis for VAD
                ax2 = ax1.twinx()
                ax2.plot(times, vad, color="red", linewidth=2.0, label="VAD Pause Prediction")
                ax2.set_ylabel("VAD Probability", color="red")
                ax2.tick_params(axis="y", labelcolor="red")
                ax2.set_ylim(-0.05, 1.05)
                
                # Add vertical red lines where VAD crosses threshold (0.7)
                crossings = [times[i] for i in range(1, len(vad)) if vad[i-1] < 0.6 and vad[i] >= 0.6]
                for i, cross_t in enumerate(crossings):
                    label = "VAD Trigger (0.6)" if i == 0 else None
                    ax2.axvline(x=cross_t, color='darkred', linestyle='-.', linewidth=1.5, alpha=0.8, label=label)
                
                # Combine legends from both axes
                lines_1, labels_1 = ax1.get_legend_handles_labels()
                lines_2, labels_2 = ax2.get_legend_handles_labels()
                ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

                plt.title("Audio Activity & VAD Tracking")
                plt.tight_layout()
                
                plot_path = output_path.with_suffix('.vad_plot.png')
                plt.savefig(plot_path)
                print(f"✓ Saved 3x1 audio & VAD prediction plot to {plot_path}")
        except ImportError:
            print("⚠ matplotlib is not installed. Skipping VAD plot generation.")

        print("\n===== Evaluation Summary =====")
        print("Transcription computed by Server Engine:")
        print("".join(transcription_log))
        print("\nResponse generated by LLM Engine:")
        print("".join(response_log))
        print("==============================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate UnmuteHandler locally bypassing WebSocket bottlenecks.")
    parser.add_argument("input_path", type=Path, help="Path to input .wav file (will be placed on Left Channel)")
    parser.add_argument("output_path", type=Path, help="Path to output stereo .wav file")
    parser.add_argument(
        "--voice",
        type=str,
        default="unmute-prod-website/ex04_narration_longform_00001.wav",
        help="The TTS voice reference file for cloning",
    )
    args = parser.parse_args()

    asyncio.run(main(args.input_path, args.output_path, args.voice))
