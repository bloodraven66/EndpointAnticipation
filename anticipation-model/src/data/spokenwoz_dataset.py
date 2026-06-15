import torch
import torchaudio
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
from src.utils import data_utils
from src.utils.run_utils import resample_audio

from src.utils.logger import logger

def handle_channels(cfg, all_wav_files):
    if hasattr(cfg.data.datasets.spokenwoz, "channels") and cfg.data.datasets.spokenwoz.channels.separate:
        separated_wav_files = {}
        if cfg.data.datasets.spokenwoz.channels.preserve == "all":
            logger.info("Preserving and saving all channels separately for SpokenWOZ dataset.")
        else:
            raise NotImplementedError("Only 'all' option is implemented for channel preservation in SpokenWOZ dataset.")

        for file_idx, file in enumerate(tqdm(all_wav_files, desc="Duplicating WAV files for separate channels")):
            file = Path(file)
            separated_wav_files[file.stem] = []
            channel_paths = {}
            for ch_idx in range(2):  # Assuming max 2 channels for SpokenWOZ
                channel_file_path = cfg.data.datasets.spokenwoz.channels.save_path.strip().format(
                    dump_path=cfg.data.save_paths.dump,
                    fname=f"{file.stem}_ch{ch_idx+1}.wav"
                )
                if file.stem not in channel_paths:
                    channel_paths[file.stem] = []
                channel_paths[file.stem].append(channel_file_path)
            if all(os.path.exists(p) for p in channel_paths[file.stem]):
                if "spokenwoz" not in cfg.data.override_preprocessed_data:
                    separated_wav_files[file.stem] = [Path(p) for p in channel_paths[file.stem]]
                    continue

            multichannel_audio = data_utils.load_full_audio(str(file), sr=cfg.data.datasets.spokenwoz.sr, preserve_channels=True)
            for ch_idx in range(multichannel_audio.shape[0]):
                channel_file_path = channel_paths[file.stem][ch_idx]
                os.makedirs(os.path.dirname(channel_file_path), exist_ok=True)
                if file_idx == 0:
                    logger.info(f"Saving channel-separated file: {channel_file_path}")
                torchaudio.save(channel_file_path, multichannel_audio[ch_idx], cfg.data.datasets.spokenwoz.sr)
                separated_wav_files[file.stem].append(Path(channel_file_path))
        return separated_wav_files, True
    return all_wav_files, False

def preprocess_spokenwoz(cfg):
    """
    Preprocess SpokenWOZ dataset to the required format
    Args:
        cfg (edict): Configuration dictionary
    Saves:
        Preprocessed data in the specified save path
    """
    data_folder = cfg.data.datasets.spokenwoz.raw_path
    train_audios = os.path.join(data_folder, "audio_5700_train_dev")
    val_audios = os.path.join(data_folder, "audio_5700_train_dev")
    test_audios = os.path.join(data_folder, "audio_5700_test")
    train_dev_json = os.path.join(data_folder, "text_5700_train_dev/data.json")
    test_json = os.path.join(data_folder, "text_5700_test/data.json")
    val_ids = os.path.join(data_folder, "text_5700_train_dev/valListFile.json")
    
    train_val_data = data_utils.load_data_from_file(train_dev_json, reader="json")
    val_ids = data_utils.load_data_from_file(val_ids, reader="txt")
    train_data = {k: v for k, v in train_val_data.items() if k not in val_ids}
    val_data = {k: v for k, v in train_val_data.items() if k in val_ids}
    test_data = data_utils.load_data_from_file(test_json, reader="json")
    modes = ["train", "val", "test"]
    for mode, data, audios in zip(
        modes,
        [train_data, val_data, test_data],
        [train_audios, val_audios, test_audios],
    ):  
        save_path = cfg.data.save_paths.preprocessed_data_path.strip().format(dataset="spokenwoz", mode=mode)
        save_path = os.path.join(cfg.data.save_paths.dump, save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if os.path.exists(save_path):
            if "spokenwoz" not in cfg.data.override_preprocessed_data:
                continue
        preprocessed_json = {}  
        all_audio_paths = []
        for key in data:
            audio_path = os.path.join(audios, key + ".wav")
            preprocessed_json[key] = {
                "audio_filepath": audio_path,
            }
            segments = []
            for turn_data in data[key]["log"]:
                segments.append({
                    "start_time": turn_data["words"][0]["BeginTime"] / 1000.0,
                    "end_time": turn_data["words"][-1]["EndTime"] / 1000.0,
                    "turn": turn_data["tag"],
                    "text": turn_data["text"],
                })
            all_audio_paths.append(audio_path)
            preprocessed_json[key]["segments"] = segments
        
        all_wav_files, status = handle_channels(cfg, all_audio_paths)
        if status:
            ## now we need to convert preprocessed_json to have channel-wise data
            updated_preprocessed_json = {}
            for key in preprocessed_json:
                ##user = "user_ch" - idx0 , system = "system_ch" - idx1
                ##we need to include user and system turns in respective channels
                updated_preprocessed_json[key] = {}
                for ch_idx, turn in enumerate(["user", "system"]):
                    updated_preprocessed_json[key][f"ch{ch_idx}"] = {
                        "audio_filepath": str(all_wav_files[key][ch_idx]),
                        "segments": []
                    }
                    for segment in preprocessed_json[key]["segments"]:
                        if segment["turn"] == turn:
                            updated_preprocessed_json[key][f"ch{ch_idx}"]["segments"].append(segment)
            preprocessed_json = updated_preprocessed_json
                    
        data_utils.write_data_to_file(preprocessed_json, save_path, writer="json")
    
