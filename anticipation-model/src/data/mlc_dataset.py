import torch
import torchaudio
import os
from src.utils import textgrid
from pathlib import Path
from tqdm import tqdm
import numpy as np
from src.utils import data_utils
from pathlib import Path
from src.utils.logger import logger

def preprocess_mlc(cfg):
    data_folder = cfg.data.datasets.mlc.raw_path
    train_folder = os.path.join(data_folder, "MLC-SLM_Workshop-Training_Set_1/data/")
    dev_folder = os.path.join(data_folder, "MLC-SLM_Workshop-Development_Set/data/")

    lang_mapping = {
        "en": "English",
        "fr": "French",
        "de": "German",
        "it": "Italian",
        "kr": "Korean",
        "jn": "Japanese",
        "pt": "Portuguese",
        "ru": "Russian",
        "es": "Spanish",
        "th": "Thai",
        "vi": "Vietnamese"
    }

    for mode, folder in zip(
        ["train", "val"],
        [train_folder, dev_folder],
    ):  
        save_path = cfg.data.save_paths.preprocessed_data_path.strip().format(dataset="mlc", mode=mode)
        save_path = os.path.join(cfg.data.save_paths.dump, save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if os.path.exists(save_path):
            if "mlc" not in cfg.data.override_preprocessed_data:
                continue
        audio_files, text_files = [], []
        preprocessed_json = {} 
        ##collect all audio and text files for specified languages
        for language in cfg.data.datasets.mlc.languages:
            lang_name = lang_mapping.get(language, None)
            if lang_name is None:
                logger.warning(f"Language {language} not recognized. Quitting.")
                exit(1)
            folder_lang = os.path.join(folder, lang_name)
            lang_audio_files = data_utils.get_files(Path(folder_lang), extension='.wav')
            lang_text_files = data_utils.get_files(Path(folder_lang), extension='.txt')
            audio_files.extend(lang_audio_files)
            text_files.extend(lang_text_files)
        ##now we need to extract data from text files
        for text_file in tqdm(text_files, desc=f"Preprocessing MLC {mode} data"):
            key = text_file.stem
            corresponding_audio_file = text_file.with_suffix('.wav')
            parent_folder_for_text_file = str(text_file.parent)
            id = parent_folder_for_text_file.split('/')[-1].replace(' ', "__") + '_' + key
            text_data = data_utils.load_data_from_file(str(text_file), reader="txt")
            preprocessed_json[id] = {
                "ch0": {
                    "audio_filepath": str(corresponding_audio_file),
                    "segments": []
                }
            }
            for line in text_data:
                start_time, end_time, spk_tag, turn_text = line.strip().split('\t')
                start, end = round(float(start_time), 4), round(float(end_time),4)
                label_data = {
                    "turn": "user",
                    "text": turn_text,
                    "start_time": start,
                    "end_time": end,
                }
                preprocessed_json[id]["ch0"]["segments"].append(label_data)
        data_utils.write_data_to_file(preprocessed_json, save_path, writer="json")
            
