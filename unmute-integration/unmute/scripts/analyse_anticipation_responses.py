"""
Analyse per-sample latency for anticipation experiment results.

Latency definition mirrors eval_smooth_turn_taking.py:
  input_end_time   = turn_taking.json[0]["timestamp"][0]  (start of turn boundary marker)
  output_start_time = first output.json chunk whose timestamp[0] >= input_end_time
  latency          = output_start_time - input_end_time

VAD latency = vad_pause_detected_sec - input_end_time  (only reported when > 0)

Only the first completed (non-interrupted) generation is used per sample.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np


DATA_DIR = Path("/mnt/matylda4/udupa/data/Full-Duplex-Bench-Data/v1.0/candor_turn_taking")


def _stats(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "mean": round(float(np.mean(vals)), 3),
        "median": round(float(np.median(vals)), 3),
        "p90": round(float(np.percentile(vals, 90)), 3),
        "min": round(float(min(vals)), 3),
        "max": round(float(max(vals)), 3),
    }


def analyse(results_dir: Path, data_dir: Path = DATA_DIR, outlier_threshold: float = 3.0):
    rows = []

    sample_ids = sorted(
        [d.name for d in results_dir.iterdir() if d.is_dir()],
        key=lambda x: int(x) if x.isdigit() else x,
    )

    for sample_id in sample_ids:
        res_dir = results_dir / sample_id
        tt_file = data_dir / sample_id / "turn_taking.json"
        output_file = res_dir / "output.json"
        timings_file = res_dir / "output.timings.json"

        if not tt_file.exists() or not output_file.exists() or not timings_file.exists():
            continue

        with tt_file.open() as f:
            tt = json.load(f)
        if not tt:
            continue
        input_end_time = float(tt[0]["timestamp"][0])

        with output_file.open() as f:
            output_data = json.load(f)
        chunks = [c for c in output_data.get("chunks", []) if c["timestamp"][0] >= input_end_time]
        if not chunks:
            continue
        output_start_time = chunks[0]["timestamp"][0]
        latency = output_start_time - input_end_time
        if latency < 0:
            continue

        with timings_file.open() as f:
            timings = json.load(f)

        # Find first completed generation
        completed_gens = [
            g for g in timings.get("generations", [])
            if g.get("status") == "completed"
        ]
        if not completed_gens:
            continue
        gen = completed_gens[0]
        # Skip if this generation's VAD time is wildly off from input_end_time (>5s)
        vad_sec = gen.get("vad_pause_detected_sec")
        if vad_sec is not None and abs(vad_sec - input_end_time) > 5:
            gen = completed_gens[1] if len(completed_gens) > 1 else None
        if gen is None:
            continue

        vad_sec = gen.get("vad_pause_detected_sec")
        llm_start_sec = gen.get("llm_generation_started_sec")
        llm_first_token_sec = gen.get("llm_first_token_sec")
        tts_gen_sec = gen.get("tts_generation_started_sec")
        tts_first_token_sec = gen.get("tts_first_token_received_sec")
        tts_first_audio_sec = gen.get("tts_first_audio_chunk_received_sec")

        def _lat(t):
            return round(t - input_end_time, 3) if t is not None else None

        vad_lat = _lat(vad_sec)
        llm_start_lat = _lat(llm_start_sec)
        llm_first_token_lat = _lat(llm_first_token_sec)
        tts_gen_lat = _lat(tts_gen_sec)
        tts_first_token_lat = _lat(tts_first_token_sec)
        tts_first_audio_lat = _lat(tts_first_audio_sec)

        rows.append({
            "id": sample_id,
            "input_end": round(input_end_time, 3),
            "latency": round(latency, 3),
            "vad_lat": vad_lat,
            "llm_start_lat": llm_start_lat,
            "llm_first_token_lat": llm_first_token_lat,
            "tts_gen_lat": tts_gen_lat,
            "tts_first_token_lat": tts_first_token_lat,
            "tts_first_audio_lat": tts_first_audio_lat,
            "n_spec": len(timings.get("speculation_attempts", [])),
        })

    # Per-sample table (only rows with positive VAD latency)
    vad_positive = [r for r in rows if r["vad_lat"] is not None and r["vad_lat"] > 0]
    vad_positive_sorted = sorted(vad_positive, key=lambda x: x["vad_lat"], reverse=True)

    header = f"{'ID':>5}  {'end_t':>6}  {'latency':>8}  {'vad_lat':>8}  {'llm_start':>9}  {'llm_1tok':>8}  {'tts_1aud':>8}  {'n_spec':>6}"
    print(header)
    print("-" * len(header))
    for r in vad_positive_sorted:
        print(
            f"{r['id']:>5}  {r['input_end']:>6.2f}  {r['latency']:>8.3f}  "
            f"{r['vad_lat']:>8.3f}  "
            f"{str(r['llm_start_lat']):>9}  "
            f"{str(r['llm_first_token_lat']):>8}  "
            f"{str(r['tts_first_audio_lat']):>8}  "
            f"{r['n_spec']:>6}"
        )

    print()
    print("=" * 60)
    print(f"[Summary]  N samples with vad_lat > 0: {len(vad_positive)}")
    print()

    all_latencies = [r["latency"] for r in vad_positive]
    all_vad = [r["vad_lat"] for r in vad_positive if r["vad_lat"] is not None]
    all_llm_start = [r["llm_start_lat"] for r in vad_positive if r["llm_start_lat"] is not None]
    all_llm_tok = [r["llm_first_token_lat"] for r in vad_positive if r["llm_first_token_lat"] is not None]
    all_tts_audio = [r["tts_first_audio_lat"] for r in vad_positive if r["tts_first_audio_lat"] is not None]

    for label, vals in [
        ("Response latency (output_start - input_end)", all_latencies),
        ("VAD latency      (vad_fired  - input_end)",   all_vad),
        ("LLM start lat    (llm_start  - input_end)",   all_llm_start),
        ("LLM 1st token    (llm_1tok   - input_end)",   all_llm_tok),
        ("TTS 1st audio    (tts_audio  - input_end)",   all_tts_audio),
    ]:
        s = _stats(vals)
        if s["n"] == 0:
            print(f"  {label}: no data")
        else:
            print(f"  {label}:")
            print(f"    mean={s['mean']:.3f}s  median={s['median']:.3f}s  p90={s['p90']:.3f}s  min={s['min']:.3f}s  max={s['max']:.3f}s  (n={s['n']})")

    # Outlier-filtered summary (>3s excluded)
    filtered = [r for r in vad_positive if r["latency"] <= outlier_threshold]
    print()
    print(f"[Filtered > {outlier_threshold}s outliers]  N={len(filtered)}")
    f_lat = [r["latency"] for r in filtered]
    f_vad = [r["vad_lat"] for r in filtered if r["vad_lat"] is not None]
    f_llm = [r["llm_first_token_lat"] for r in filtered if r["llm_first_token_lat"] is not None]
    f_tts = [r["tts_first_audio_lat"] for r in filtered if r["tts_first_audio_lat"] is not None]
    for label, vals in [
        ("Response latency", f_lat),
        ("VAD latency     ", f_vad),
        ("LLM 1st token   ", f_llm),
        ("TTS 1st audio   ", f_tts),
    ]:
        s = _stats(vals)
        if s["n"]:
            print(f"  {label}: mean={s['mean']:.3f}s  median={s['median']:.3f}s  p90={s['p90']:.3f}s  (n={s['n']})")

    # Speculation breakdown
    with_spec = [r for r in rows if r["n_spec"] > 0]
    without_spec = [r for r in rows if r["n_spec"] == 0]
    print()
    print("[Speculation breakdown]")
    print(f"  With speculation:    N={len(with_spec)}  avg_latency={np.mean([r['latency'] for r in with_spec]):.3f}s" if with_spec else "  With speculation: none")
    print(f"  Without speculation: N={len(without_spec)}  avg_latency={np.mean([r['latency'] for r in without_spec]):.3f}s" if without_spec else "  Without speculation: none")

    # Raw diff table: latency - vad_lat for all vad_lat > 0 samples
    diff_rows = [
        (r["id"], round(r["vad_lat"] * 1000), round(r["latency"] * 1000), round((r["latency"] - r["vad_lat"]) * 1000))
        for r in vad_positive
    ]
    diff_rows.sort(key=lambda x: x[3], reverse=True)
    diffs_ms = [d[3] for d in diff_rows]
    print()
    print(f"[Raw diff: latency - vad_lat (all vad_lat > 0)]  N={len(diff_rows)}")
    print(f"{'ID':>5}  {'vad_lat_ms':>10}  {'lat_ms':>8}  {'diff_ms':>8}")
    print("-" * 38)
    for sid, vad_ms, lat_ms, diff_ms in diff_rows:
        print(f"{sid:>5}  {vad_ms:>10}  {lat_ms:>8}  {diff_ms:>8}")
    if diffs_ms:
        print()
        print(f"  median={np.median(diffs_ms):.0f}ms  mean={np.mean(diffs_ms):.0f}ms  p90={np.percentile(diffs_ms, 90):.0f}ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse anticipation experiment latency.")
    parser.add_argument("--results_dir", type=Path, required=True, help="Path to candor_turn_taking results directory.")
    parser.add_argument("--data_dir", type=Path, default=DATA_DIR, help="Path to benchmark data directory.")
    parser.add_argument("--outlier_threshold", type=float, default=3.0, help="Latency threshold (s) above which samples are excluded from filtered summary.")
    args = parser.parse_args()
    analyse(args.results_dir, args.data_dir, args.outlier_threshold)
