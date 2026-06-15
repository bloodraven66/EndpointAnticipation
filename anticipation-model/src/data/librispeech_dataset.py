import os
from pathlib import Path
import torch
from tqdm import tqdm
import numpy as np
from src.utils import data_utils
from pathlib import Path
from src.utils.logger import logger
import librosa, torchaudio

def preprocess_librispeech(cfg):
    data_folder = cfg.data.datasets.librispeech.raw_path
    train_folders = [os.path.join(data_folder, split) for split in cfg.data.datasets.librispeech.train_splits]
    dev_folders = [os.path.join(data_folder, split) for split in cfg.data.datasets.librispeech.eval_splits]
    test_folders = [os.path.join(data_folder, split) for split in cfg.data.datasets.librispeech.test_splits]
    for mode, folder in zip(
        ["train", "val", "test"],
        [train_folders, dev_folders, test_folders],
    ):
        save_path = cfg.data.save_paths.preprocessed_data_path.strip().format(dataset="librispeech", mode=mode)
        save_path = os.path.join(cfg.data.save_paths.dump, save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if os.path.exists(save_path):
            if "librispeech" not in cfg.data.override_preprocessed_data:
                continue
        preprocessed_json = {} 

        if not cfg.data.datasets.librispeech.get("create_session_recordings", False):

            audio_files, text_files = [], {}
            for folder_ in folder:
                audio_files = data_utils.get_files(Path(folder_), extension='.txt')
                audio_files.extend(audio_files)
            
            for audio_file in tqdm(audio_files, desc=f"Preprocessing LibriSpeech {mode} data"):
                key = audio_file.stem
                session = key.rsplit('-', 1)[0]
                if session not in text_files:
                    text_files[session] = {}
                    session_transcript_file = audio_file.parent / f"{session}.trans.txt"
                    text_data = data_utils.load_data_from_file(session_transcript_file, reader="txt")
                    for item in text_data:
                        parts = item.strip().split(' ', 1)
                        assert len(parts) == 2, f"Unexpected transcript format in {session_transcript_file}"
                        utt_id, transcript = parts
                        text_files[session][utt_id] = transcript
                transcript = text_files[session][key]
                duration = librosa.get_duration(filename=str(audio_file))
                start, end = 0.0, round(duration, 4)
                preprocessed_json[key] = {
                    "audio_filepath": str(audio_file),
                    "segments": [
                        {
                            "turn": "user",
                            "text": transcript,
                            "start_time": start,
                            "end_time": end
                        }
                    ]
                }
        
        else:
            os.makedirs(
                cfg.data.datasets.librispeech.session_recordings_path.format(
                    dump_path=cfg.data.save_paths.dump
                ),
                exist_ok=True
            )
            audio_files, text_files = [], []
            for folder_ in folder:
                text_files_ = data_utils.get_files(Path(folder_), extension='.txt')
                text_files.extend(text_files_)
            
            for text_file in tqdm(text_files, desc=f"Preprocessing LibriSpeech {mode} data"):
                audio_folder = text_file.parent
                text_data = data_utils.load_data_from_file(text_file, reader="txt")
                current_end_time = None
                segments = []
                for idx, line in enumerate(text_data):
                    parts = line.strip().split(' ', 1)
                    assert len(parts) == 2, f"Unexpected transcript format in {text_file}"
                    utt_id, transcript = parts
                    corresponding_audio_file = audio_folder / f"{utt_id}.flac"
                    y, sr = librosa.load(corresponding_audio_file, sr=cfg.data.datasets.librispeech.sr)
                    duration = round(len(y) / sr, 4)
                    if idx == 0:
                        start, end = 0.0, duration
                        current_end_time = end
                        concat_audio = y
                    else:
                        add_silence_between_utts = np.random.uniform(low=cfg.data.datasets.librispeech.join_silence_range[0],
                                                                     high=cfg.data.datasets.librispeech.join_silence_range[1])
                        start = round(current_end_time + add_silence_between_utts, 4)
                        end = round(start + duration, 4)
                        current_end_time = end
                        concat_audio = np.concatenate(
                            (
                                concat_audio,
                                np.zeros(int(sr * add_silence_between_utts)),
                                y
                            )
                        )
                    segments.append(
                        {
                            "turn": "user",
                            "text": transcript,
                            "start_time": start,
                            "end_time": end
                        }
                    )
                audio_save_path = os.path.join(
                    cfg.data.datasets.librispeech.session_recordings_path.format(
                        dump_path=cfg.data.save_paths.dump
                    ),
                    f"{text_file.stem}.wav"
                )
                assert abs(len(concat_audio) / sr - current_end_time) < 1, "Mismatch in audio duration and end time"
                torchaudio.save(audio_save_path, torch.from_numpy(concat_audio).unsqueeze(0), sr)
                preprocessed_json[text_file.stem] = {
                    "ch0": {
                        "audio_filepath": audio_save_path,
                        "segments": segments
                    }
                }

        data_utils.write_data_to_file(preprocessed_json, save_path, writer="json")

