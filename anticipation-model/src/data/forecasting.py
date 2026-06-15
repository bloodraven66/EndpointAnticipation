import torch, torchaudio
import os
import copy
from tqdm import tqdm
import numpy as np
from src.utils import data_utils
from src.utils.logger import logger
from src.data.data_processing import handle_length_filtering, endpointing_dataset

def vectorised_forecast_label_generation(frame_labels, min_turn_length_frames, turn_idx, turn_end_idx):
    subsequence = np.array([turn_idx] * min_turn_length_frames + [turn_end_idx])  # e.g., [3, 3, 3, 1]
    indices = np.arange(len(frame_labels) - len(subsequence) + 1)
    matrix = np.array([frame_labels[i:i+len(subsequence)] for i in indices]) # shape (T - len(subsequence) + 1, len(subsequence))
    comparisions = (matrix == subsequence[None, :])
    valid_transitions = np.where(np.all(comparisions, axis=1))[0] # indices where the subsequence matches
    return valid_transitions, subsequence

class forecasting_dataset(endpointing_dataset):
    """
    Dataset class for forecasting tasks
    Args:
        cfg: Configuration object
        mode: Mode of the dataset (e.g., train, val, test)
        dataset: Name of the dataset
        feat_extractor: Feature extractor for audio features
    Returns:
        Dataset object for endpointing tasks
    """
    def __init__(self, cfg, mode, dataset, feat_extractor=None):
        super().__init__(cfg, mode, dataset, feat_extractor)
        forecast_intervals_ms = self.cfg.data.label_params.forecast_intervals_ms
        freq = self.cfg.data.audio_params.freq
        self.forecast_intervals = np.array([int((interval / 1000) * freq) for interval in forecast_intervals_ms])
        self.turn_end_idx = self.label_mapping[self.cfg.data.special_tokens.user_end]
        if dataset == "fisher":
            ignore_keys = ["fe_03_00160", "fe_03_00141"]
            prev_length = len(self.keys)
            self.keys = [k for k in self.keys if k not in ignore_keys]
            logger.info(f"Reduced dataset keys from {prev_length} to {len(self.keys)}")
        if hasattr(cfg.data.datasets[dataset], "min_words_per_turn") and cfg.data.datasets[dataset].min_words_per_turn is not None:
            self.trim_short_text_turns()
    
    def trim_short_text_turns(self):
        total_removed, total_removed_for_neg, total_preserved = 0, 0, 0
        for key in self.data_json:
            first_channel = list(self.data_json[key].keys())[0]
            label_data = self.data_json[key][first_channel]["segments"]
            # print(label_data)
            # print("----")
            new_segments = []
            for turn in label_data:
                text = " ".join(turn["text"].split())
                if len(text.split()) >= self.cfg.data.datasets[self.dataset].min_words_per_turn:
                    if turn["end_time"] > turn["start_time"]:
                        total_preserved += 1
                    else:
                        turn["turn"] = "<user_end>"
                        total_removed_for_neg += 1
                else:
                    turn["turn"] = "<user_end>"
                    total_removed += 1
                new_segments.append(turn)
            # print(new_segments)
            # exit()
                

            self.data_json[key][first_channel]["segments"] = new_segments
                
        logger.info(f"Removed {total_removed} turns due to short text, {total_removed_for_neg} for negative turn dur, {total_preserved} preserved!")

    def generate_forecast_labels(self, frame_labels):
        # print(frame_labels)
        T = len(frame_labels)
        Y_h = np.zeros((T, len(self.forecast_intervals)), dtype=np.float32)

        # Find transitions to turn-end (where label becomes 1 from non-1)
        transitions = []
        turn_lengths = []
        turn_labels = list(self.cfg.data.special_tokens.values())
        turn_end_idx = turn_labels.index(self.cfg.data.special_tokens.user_end)
        turn_idx = turn_labels.index(self.cfg.data.special_tokens.user)


        ##i need to mark frames where there is 3/1 transition - vectorise
        if hasattr(self.cfg.model, "remove_forecast_on_short_frames") and self.cfg.model.remove_forecast_on_short_frames.apply:
            min_turn_length_frames = int((self.cfg.model.remove_forecast_on_short_frames.min_segment_ms / 1000) * self.cfg.data.audio_params.freq)
        else:
            min_turn_length_frames = 1
        ### find if min_turn_length_frames == 3, find 3 3 3 1   
        
        valid_transitions, subsequence = vectorised_forecast_label_generation(frame_labels, min_turn_length_frames, turn_idx, turn_end_idx)
        valid_transitions += len(subsequence)

        for i, h in enumerate(self.forecast_intervals):
            transition_for_forecast, subsequence = vectorised_forecast_label_generation(frame_labels, h, turn_idx, turn_end_idx)
            transition_for_forecast += len(subsequence)
            ##include any transition_for_forecast_end in valid_transitions
            forecast_transitions = [(t-len(subsequence), t) for t in transition_for_forecast if t in valid_transitions] 
            for interval in forecast_transitions:
                start = max(0, interval[0])
                end = interval[1]
                Y_h[start:end, i] = 1.0
        # shifted_frame_labels = np.concatenate(([-1], frame_labels)) #0, turn-labels shifted by 1 ## -1 3 3 3 1 1  1
        # original_frame_labels = np.concatenate((frame_labels, [-1])) #1, no shift                ##  3 3 3 1 1 1 -1
        # print([(i, frame_labels[i]) for i in range(len(frame_labels))], valid_transitions, forecast_transitions, self.forecast_intervals)
        # transitions = np.where((shifted_frame_labels == turn_idx) & (original_frame_labels == turn_end_idx))[:-1] ##0 0 0 1 0 0 
        # user_turn_locations = np.where(frame_labels == turn_idx)
        # print(user_turn_locations)
        # 
            
        # exit()
        # for t in range(1, T):
            # if frame_labels[t] == 1 and frame_labels[t-1] != 1:
                # transitions.append(t)
                ##also track the length of turn before the turn-end


        # For each transition, mark the preceding frames within each horizon
        # for trans_idx in transitions:
            # for i, h in enumerate(self.forecast_intervals):
                # start = max(0, trans_idx - h)
                # Y_h[start:trans_idx, i] = 1.0
        # for idx in range(T):
            # print(idx, frame_labels[idx], Y_h[idx])
        return Y_h

        # exit()

    def __getitem__(self, idx):
        """
        Load frame-level turn labels for a fixed length segment, along with corresponding audio features.
        """
        key = self.keys[idx]
        first_channel = list(self.data_json[key].keys())[0]
        label_data = self.data_json[key][first_channel]["segments"]
        if self.cfg.data.label_params.use_fixed_context_training:
            fixed_context_labels, start_time, end_time = data_utils.convert_continous_labels_to_fixed_context_frames(self.cfg, label_data, key)    

        ys = []
        for ch in self.data_json[key]:
            y_ch = data_utils.load_audio_segment(self.data_json[key][ch]["audio_filepath"], start_time, end_time, self.sr, preserve_channels=True)
            ys.append(y_ch.unsqueeze(0))
        # melspec = self.audio_feature(ys)
        # print(melspec.shape, start_time, end_time, ys[0].shape)
        # preserve_channels = False
        # if hasattr(self.cfg.data, "multi_audio_stream"):
            # preserve_channels = self.cfg.data.multi_audio_stream
        # if hasattr(self.cfg.data, "zero_system"):
            # if self.cfg.data.zero_system:
                # preserve_channels = True
        # print("Loading audio segment:", self.data_json[key]["audio_filepath"], start_time, end_time, "Preserve channels:", preserve_channels, self.sr)
        # y = data_utils.load_audio_segment(self.data_json[key]["audio_filepath"], start_time, end_time, self.sr, preserve_channels=True)
        # if hasattr(self.cfg.data, "zero_system"):
        #     if self.cfg.data.zero_system:
        #         y = y[0]
        #         preserve_channels = False
        # with torch.no_grad():
        #     yd = y
        #     if self.cfg.data.audio_params.audio_feature not in ["logmel", "logmel-v2"] and preserve_channels:
        #         yd = y.numpy()
        #         yd = [yd[0], yd[1]]
        #     melspec = self.audio_feature(yd)
        mel_length = int(round((end_time - start_time) * self.cfg.data.audio_params.freq))
        aligned_labels, texts = data_utils.align_labels_with_frames(fixed_context_labels, mel_length, self.label_mapping, key)
        fc_aligned_labels = self.generate_forecast_labels(aligned_labels)
        # print(aligned_labels.shape, fc_aligned_labels.shape)
        # exit()
        texts = self.cfg.data.text_delim.join(texts)    
        fc_aligned_labels = torch.from_numpy(np.array(fc_aligned_labels)).long() 
        aligned_labels = torch.from_numpy(np.array(aligned_labels)).long()       
        # assert y.shape[-1] == round(end_time - start_time) * self.sr, f"Shape mismatch: {y.shape[-1]} != {round(end_time - start_time) * self.cfg.data.datasets[self.dataset].sr}"
        # assert melspec.shape[-1] == aligned_labels.shape[-1], f"Shape mismatch: {melspec.shape[-1]} != {aligned_labels.shape[-1]}"
        # print(melspec.shape, aligned_labels.shape, fc_aligned_labels.shape, start_time, end_time)
        # if melspec.dim() == 2:
            # melspec = melspec.unsqueeze(0)
        # print(fc_aligned_labels.shape)
        if self.max_length is not None:
            if fc_aligned_labels.shape[0] > self.max_length:
                # if preserve_channels:
                # print(melspec.shape)
                # melspec = melspec[:, :, :self.max_length]
                # else:
                    # melspec = melspec[:, :self.max_length]
                fc_aligned_labels = fc_aligned_labels[:self.max_length, :]
                aligned_labels = aligned_labels[:self.max_length]   
        ys = torch.cat(ys, dim=0)
        # print("loader", ys.shape)
        return ys, fc_aligned_labels, (self.data_json[key][first_channel]["audio_filepath"], 0, texts, start_time, end_time, key, aligned_labels)

class CollateForecasting:
    def __init__(self, cfg, encoder):
        self.cfg = cfg
        self.encoder = encoder
    
    def __call__(self, batch):
        audios = [item[0] for item in batch] 
        audios = torch.stack(audios, dim=0)  # bs x num_channels x T
        audios = audios.reshape(-1, audios.shape[-1])  # (bs * num_channels) x T
        # print(audios.shape)
        fc_aligned_labels = [item[1] for item in batch]
        meta_data = [item[2] for item in batch]
        # melspec = self.encoder(audios) # bs * num_channels x feat_dim x T_frames
        # melspec = melspec.reshape(len(batch), -1, melspec.shape[1], melspec.shape[2])  # bs x num_channels x feat_dim x T_frames
        fc_aligned_labels = torch.stack(fc_aligned_labels, dim=0)  # bs x T_frames x num_forecast_intervals
        # print(fc_aligned_labels.shape, melspec.shape)
        # melspec = melspec[:, :, :, :fc_aligned_labels.shape[1]]  # Align melspec length with labels
        return audios, fc_aligned_labels, meta_data


class forecasting_dataset_full_context(forecasting_dataset):
    def __init__(self, cfg, mode, dataset, feat_extractor=None):
        super().__init__(cfg, mode, dataset, feat_extractor)
        IGNORE_KEYS = [
            "SNG0601", 
            "SNG0646", 
            "SNG0653",
            "SNG0877", 
            "SNG0885", 
            "SNG0890", 
            "SNG0897", 
            "SNG0901",
            "SNG0903",
            "MUL0363",
        ]
        for k in IGNORE_KEYS:
            if k in self.keys:
                self.keys.remove(k)
            
    def __getitem__(self, idx):
        key = self.keys[idx]
        first_channel = list(self.data_json[key].keys())[0]
        label_data = self.data_json[key][first_channel]["segments"]
        label_data, start_time, end_time = data_utils.convert_continous_labels_to_list(self.cfg, label_data)
        ys = []
        # print(start_time, end_time)
        for ch in self.data_json[key]:
            y_ch = data_utils.load_audio_segment(self.data_json[key][ch]["audio_filepath"], start_time, end_time, self.sr, preserve_channels=True)
            ys.append(y_ch.unsqueeze(0))
        mel_length = int(round((end_time - start_time) * self.cfg.data.audio_params.freq))
        # print(ys[0].shape)
        aligned_labels, texts = data_utils.align_labels_with_frames(label_data, mel_length, self.label_mapping, key)
        fc_aligned_labels = self.generate_forecast_labels(aligned_labels)
        texts = self.cfg.data.text_delim.join(texts)    
        fc_aligned_labels = torch.from_numpy(np.array(fc_aligned_labels)).long() 
        aligned_labels = torch.from_numpy(np.array(aligned_labels)).long()
        # if ys[0].shape[-1] > round(round(end_time - start_time, 3) * self.sr):
            # ys_ = []
            # ys_.append(ys[0][:, :round(round(end_time - start_time, 3) * self.sr)])
            # ys_.append(ys[1][:, :round(round(end_time - start_time, 3) * self.sr)])  
            # ys = ys_  
        # assert ys[0].shape[-1] == round(round(end_time - start_time) * self.sr), f"Shape mismatch: {ys[0].shape[-1]} != {round(end_time - start_time) * self.sr,  round(end_time - start_time, 3) * self.sr, start_time, end_time, key, mel_length, aligned_labels.shape[-1], self.sr, ys[0].shape}"
        assert mel_length == aligned_labels.shape[-1], f"Shape mismatch: {melspec.shape[-1]} != {aligned_labels.shape[-1],  start_time, end_time}"
        return ys, fc_aligned_labels, (self.data_json[key][first_channel]["audio_filepath"], label_data, texts, start_time, end_time, key, aligned_labels)
