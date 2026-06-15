from src.utils import data_utils
import os, re
import torchaudio
from tqdm import tqdm
from pathlib import Path
from src.utils.logger import logger

def handle_channels(cfg, all_spn_files):
    if hasattr(cfg.data.datasets.fisher, "channels") and cfg.data.datasets.fisher.channels.separate:
        all_wav_files = {}
        if cfg.data.datasets.fisher.channels.preserve == "all":
            logger.info("Preserving and saving all channels separately for Fisher dataset.")
        else:
            raise NotImplementedError("Only 'all' option is implemented for channel preservation in Fisher dataset.")

        for file_idx, file in enumerate(tqdm(all_spn_files, desc="Duplicating SPH files for separate channels")):
            all_wav_files[file.stem] = []
            channel_paths = {}
            for ch_idx in range(2):  # Assuming max 2 channels for Fisher
                channel_file_path = cfg.data.datasets.fisher.channels.save_path.strip().format(
                    dump_path=cfg.data.save_paths.dump,
                    fname=f"{file.stem}_ch{ch_idx+1}.wav"
                )
                if file.stem not in channel_paths:
                    channel_paths[file.stem] = []
                channel_paths[file.stem].append(channel_file_path)
            if all(os.path.exists(p) for p in channel_paths[file.stem]):
                if "fisher" not in cfg.data.override_preprocessed_data:
                    all_wav_files[file.stem] = [Path(p) for p in channel_paths[file.stem]]
                    continue

            multichannel_audio = data_utils.load_full_audio(str(file), sr=cfg.data.datasets.fisher.sr, preserve_channels=True)
            for ch_idx in range(multichannel_audio.shape[0]):
                channel_file_path = channel_paths[file.stem][ch_idx]
                os.makedirs(os.path.dirname(channel_file_path), exist_ok=True)
                if file_idx == 0:
                    logger.info(f"Saving channel-separated file: {channel_file_path}")
                torchaudio.save(channel_file_path, multichannel_audio[ch_idx], cfg.data.datasets.fisher.sr)
                all_wav_files[file.stem].append(Path(channel_file_path))
        return all_wav_files
    return all_spn_files

def preprocess_fisher(cfg):
    data_folder = cfg.data.datasets.fisher.raw_path
    dev_folder = os.path.join(data_folder, "fe_03_p1_tran/data/trans/000/")
    test_folder = os.path.join(data_folder, "fe_03_p1_tran/data/trans/001/")

    all_txt_files = data_utils.get_files(Path(data_folder), extension='data/trans/*/*.txt')
    all_spn_files = data_utils.get_files(Path(data_folder), extension='.sph')

    for mode, folder in zip(
        ["train", "val", "test"],
        [None, dev_folder, test_folder],
    ):  
        # print(mode, len(all_txt_files), len(all_spn_files))
        if mode == "train":
            filter_out_pattern = re.compile(r'fe_03_p1_tran/data/trans/(000|001)/')
            txt_files = [f for f in all_txt_files if not filter_out_pattern.search(str(f))]
            # sph_files = [f for f in all_spn_files if not filter_out_pattern.search(str(f))]
        else:
            txt_files = [f for f in all_txt_files if str(folder) in str(f)]
        txt_stems = set([f.stem for f in txt_files])
        sph_files = [f for f in all_spn_files if f.stem in txt_stems]
            # sph_files = [f for f in all_spn_files if str(folder) in str(f)]
            # sph_stems = set([f.stem for f in sph_files])
            # txt_files = [f for f in all_txt_files if f.stem in sph_stems]
        # print(mode, len(sph_files), len(txt_files))
        sph_dict = handle_channels(cfg, sph_files)
        
        save_path = cfg.data.save_paths.preprocessed_data_path.strip().format(dataset="fisher", mode=mode)
        save_path = os.path.join(cfg.data.save_paths.dump, save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if os.path.exists(save_path):
            if "fisher" not in cfg.data.override_preprocessed_data:
                continue
        diffs = []
        preprocessed_json = {}
        for text_file in tqdm(txt_files, desc=f"Preprocessing Fisher {mode} data"):
            key = text_file.stem
            corresponding_sph_files = sph_dict[key]
            text_data = data_utils.load_data_from_file(str(text_file), reader="txt")

            assert text_data[1].startswith("# Transcribed"), f"Unexpected header in file {text_file}, {text_data[:5]}"
            speaker_segments = {}
            for line in text_data[2:]:
                if line.strip() == "":
                    continue
                start_time, end_time, speaker, utterance = line.split(' ', 3)
                speaker = speaker.strip(":")

                ##we cannot use the segments as it is, because they can be very short, with very short silences between segments
                ##These do not correspond to actual pauses o endpoints

                ##lets use a simple heuristic:
                ##if a speaker speaks within 1 second of their previous utterance, we merge the segments
                ##for that, we track speaker_specific_segments
                if speaker not in speaker_segments:
                    speaker_segments[speaker] = []
                if len(speaker_segments[speaker]) == 0:
                    speaker_segments[speaker].append({
                        "start": float(start_time),
                        "end": float(end_time),
                        "text": utterance.strip(),
                        "original_segments": [[float(start_time), float(end_time), utterance.strip()]]
                    })
                else:
                    last_segment = speaker_segments[speaker][-1]
                    diff = float(start_time) - last_segment["end"]
                    # if diff < 0:
                        # print(speaker_segments[speaker][-1], start_time, end_time, utterance)
                        # exit()
                    diffs.append(diff)
                    if diff <= 1.0:
                        #merge segments
                        last_segment["end"] = float(end_time)
                        last_segment["text"] += " " + utterance.strip()
                        last_segment["original_segments"].append([float(start_time), float(end_time), utterance.strip()])
                    else:
                        speaker_segments[speaker].append({
                            "start": float(start_time),
                            "end": float(end_time),
                            "text": utterance.strip(),
                            "original_segments": [[float(start_time), float(end_time), utterance.strip()]]
                        })
            #now we have merged segments for each speaker
            preprocessed_json[key] = {}
            for speaker_idx, (speaker, segments) in enumerate(speaker_segments.items()):
                # speaker_file_id = f"{key}_spk{speaker}"
                speaker_audio_file = corresponding_sph_files[0] if speaker == "A" else corresponding_sph_files[1] 
                preprocessed_json[key][speaker] = {
                    "audio_filepath": str(speaker_audio_file),
                    "segments": []
                }
                for segment_idx, segment in enumerate(segments):
                    label_data = {
                        "turn": "user",
                        "text": segment["text"], 
                        "start_time": round(segment["start"], 4),
                        "end_time": round(segment["end"], 4),
                        "original_segments": segment["original_segments"] if len(segment["original_segments"]) > 1 else None,
                        "speaker": speaker
                    }
                    preprocessed_json[key][speaker]["segments"].append(label_data)
                    # preprocessed_json[speaker_file_id]["segments"].append(label_data)
                
                # print(preprocessed_json)
                # for key in preprocessed_json:
                    # print(preprocessed_json[key]["audio_files"])
                    # texts = ""
                    # for segment in preprocessed_json[key]["segments"]:
                        # print(segment)
                        # texts += segment["text"] + " "
                    # print("Full text:", texts.strip())
                    # exit()
        if len(preprocessed_json) == 0:
            logger.warning(f"No data found for Fisher {mode} set. Skipping saving preprocessed data.")
        else:
            data_utils.write_data_to_file(preprocessed_json, save_path, writer="json")
        # plot_diff_dist(diffs)
        total_duration, total_seg_duration = 0.0, 0.0
        for key in preprocessed_json:
            for speaker in preprocessed_json[key]:
                for segment in preprocessed_json[key][speaker]["segments"]:
                    total_seg_duration += segment["end_time"] - segment["start_time"]
                total_duration += preprocessed_json[key][speaker]["segments"][-1]["end_time"] - preprocessed_json[key][speaker]["segments"][0]["start_time"]
        logger.info(f"Fisher {mode} set: Processed {len(preprocessed_json)}, seg duration of {total_duration/3600.0:.2f} hours, total speech duration of {total_seg_duration/3600.0:.2f} hours.")

                
            # exit()

    # exit()

def plot_diff_dist(diffs):
    diffs = [d for d in diffs if d < 2.0 and d > -1.0]
    import matplotlib.pyplot as plt
    plt.hist(diffs, bins=100)
    plt.xlabel("Time difference between segments of same speaker (s)")
    plt.ylabel("Count")
    plt.title("Distribution of time differences between segments of same speaker in Fisher dataset")
    plt.grid()
    plt.savefig("plots/fisher_speaker_segment_diffs_lim.png")
    exit()



    