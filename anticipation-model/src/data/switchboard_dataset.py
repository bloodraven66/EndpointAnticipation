from src.utils import data_utils
import os, re
import torchaudio
from tqdm import tqdm
from pathlib import Path
from src.utils.logger import logger

def handle_channels(cfg, all_spn_files):
    if hasattr(cfg.data.datasets.switchboard, "channels") and cfg.data.datasets.switchboard.channels.separate:
        all_wav_files = {}
        if cfg.data.datasets.switchboard.channels.preserve == "all":
            logger.info("Preserving and saving all channels separately for Switchboard dataset.")
        else:
            raise NotImplementedError("Only 'all' option is implemented for channel preservation in Switchboard dataset.")

        for file_idx, file in enumerate(tqdm(all_spn_files, desc="Duplicating SPH files for separate channels")):
            file = Path(file)
            all_wav_files[file.stem] = []
            channel_paths = {}
            for ch_idx in range(2):  # Assuming max 2 channels for Switchboard
                channel_file_path = cfg.data.datasets.switchboard.channels.save_path.strip().format(
                    dump_path=cfg.data.save_paths.dump,
                    fname=f"{file.stem}_ch{ch_idx+1}.wav"
                )
                if file.stem not in channel_paths:
                    channel_paths[file.stem] = []
                channel_paths[file.stem].append(channel_file_path)
            if all(os.path.exists(p) for p in channel_paths[file.stem]):
                if "switchboard" not in cfg.data.override_preprocessed_data:
                    all_wav_files[file.stem] = [Path(p) for p in channel_paths[file.stem]]
                    continue

            multichannel_audio = data_utils.load_full_audio(str(file), sr=cfg.data.datasets.switchboard.sr, preserve_channels=True)
            for ch_idx in range(multichannel_audio.shape[0]):
                channel_file_path = channel_paths[file.stem][ch_idx]
                os.makedirs(os.path.dirname(channel_file_path), exist_ok=True)
                if file_idx == 0:
                    logger.info(f"Saving channel-separated file: {channel_file_path}")
                torchaudio.save(channel_file_path, multichannel_audio[ch_idx], cfg.data.datasets.switchboard.sr)
                all_wav_files[file.stem].append(Path(channel_file_path))
        return all_wav_files
    return all_spn_files

def preprocess_switchboard(cfg):
    data_folder = cfg.data.datasets.switchboard.raw_path
    train_folder = os.path.join(data_folder, "train_all")
    dev_folder = os.path.join(data_folder, "rt03")
    test_folder = os.path.join(data_folder, "eval2000")


    for mode, folder in zip(
        ["train", "val", "test"],
        [train_folder, dev_folder, test_folder],
    ):  
        wavscp = os.path.join(folder, "wav.scp")
        text = os.path.join(folder, "text")
        segments = os.path.join(folder, "segments")
        wavscp_data_ = data_utils.load_data_from_file(wavscp, reader="txt")
        text_data = data_utils.load_data_from_file(text, reader="txt")
        segments_data = data_utils.load_data_from_file(segments, reader="txt")
        wavscp_data = {}
        skipped_file_due_to_filtering = 0
        for line in wavscp_data_:
            file_path = line.strip().split()[-2]
            if cfg.data.datasets.switchboard.filter_out_keyword in file_path:
                skipped_file_due_to_filtering += 1
                continue
            wavscp_data[Path(file_path).stem] = file_path
        logger.info(f"removed {skipped_file_due_to_filtering} files due to filter - {cfg.data.datasets.switchboard.filter_out_keyword}")
        logger.info(f"preserved {len(wavscp_data)} audio files") 
        sph_dict = handle_channels(cfg, list(wavscp_data.values()))
        save_path = cfg.data.save_paths.preprocessed_data_path.strip().format(dataset="switchboard", mode=mode)
        save_path = os.path.join(cfg.data.save_paths.dump, save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if os.path.exists(save_path):
            if "switchboard" not in cfg.data.override_preprocessed_data:
                continue        
        speaker_segments = {}
        segments_data = {s.split(' ')[0]:s for s in segments_data}
        for line in tqdm(text_data, desc=f"Preprocessing switchboard {mode} data"):
            text_key = line.split(' ')[0]
            if cfg.data.datasets.switchboard.filter_keyword not in text_key: continue
            text_for_line = " ".join(line.split(' ')[1:])
            _, audio_speaker_key, start, stop = segments_data[text_key].split()
            audio_key = audio_speaker_key.split('-')[0]
            assert len(audio_speaker_key.split('-')) == 2, f"{audio_speaker_key.split('-')}"
            assert audio_speaker_key.split('-')[-1] in ['A', 'B'], f"{audio_speaker_key}"
            if audio_key not in speaker_segments:
                speaker_segments[audio_key] = {}
            if audio_speaker_key not in speaker_segments[audio_key]:
                speaker_segments[audio_key][audio_speaker_key] = []

            speaker_segments[audio_key][audio_speaker_key].append(
                {
                    "start": float(start),
                    "end": float(stop),
                    "text": text_for_line,   
                }
            )    
        preprocessed_json = {}
        for audio_key in speaker_segments:
            preprocessed_json[audio_key] = {}
            corresponding_sph_files = sph_dict[audio_key]
            for speaker in speaker_segments[audio_key]:
                preprocessed_json[audio_key][speaker] = {
                    "audio_filepath": str(corresponding_sph_files[0] if speaker.split('-')[-1] == "A" else corresponding_sph_files[1]),
                    "segments": []
                } 
                for segment in speaker_segments[audio_key][speaker]:
                    label_data = {
                        "turn": "user",
                        "text": segment["text"], 
                        "start_time": round(segment["start"], 4),
                        "end_time": round(segment["end"], 4),
                        "speaker": speaker
                    }
                    preprocessed_json[audio_key][speaker]["segments"].append(label_data)
        if len(preprocessed_json) == 0:
            logger.warning(f"No data found for switchboard {mode} set. Skipping saving preprocessed data.")
        else:
            data_utils.write_data_to_file(preprocessed_json, save_path, writer="json")
        # plot_diff_dist(diffs)
        total_duration, total_seg_duration = 0.0, 0.0
        for key in preprocessed_json:
            for speaker in preprocessed_json[key]:
                for segment in preprocessed_json[key][speaker]["segments"]:
                    total_seg_duration += segment["end_time"] - segment["start_time"]
                total_duration += preprocessed_json[key][speaker]["segments"][-1]["end_time"] - preprocessed_json[key][speaker]["segments"][0]["start_time"]
        logger.info(f"switchboard {mode} set: Processed {len(preprocessed_json)}, seg duration of {total_duration/3600.0:.2f} hours, total speech duration of {total_seg_duration/3600.0:.2f} hours.")


    