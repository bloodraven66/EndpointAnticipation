import torch
import matplotlib.pyplot as plt

def find_all_valid_cutoffs(non_interval_predictions, intervals):
    working_mask = non_interval_predictions.clone()
    valid_cutoffs_mask = torch.zeros_like(working_mask, dtype=torch.bool) ##init solution
    _, turn_length, num_horizons = working_mask.shape
    time_indices = torch.arange(turn_length, device=working_mask.device).view(1, turn_length, 1) # time_indices shape: (1, turn_length, 1)
    interval_lengths = intervals.view(1, 1, -1) # intervals shape: (1, 1, num_horizons)
    while working_mask.any():
        has_trigger = working_mask.any(dim=1) # Find which (threshold, horizon) pairs still have triggers left
        first_idx = working_mask.float().argmax(dim=1) # Find the frame index of the FIRST remaining trigger
        first_idx_exp = first_idx.unsqueeze(1) # Shape: (num_thresholds, 1, num_horizons)
        has_trigger_exp = has_trigger.unsqueeze(1) 
        new_triggers = torch.zeros_like(valid_cutoffs_mask)
        new_triggers.scatter_(
            dim=1, 
            index=first_idx_exp, 
            src=has_trigger_exp
        )
        valid_cutoffs_mask |= new_triggers #RECORD THE TRIGGER - place a True at the exact time index
        in_cooldown = (time_indices >= first_idx_exp) & \
                  (time_indices < (first_idx_exp + interval_lengths)) & \
                  has_trigger_exp
        working_mask &= ~in_cooldown #ignore the cooldown frames
    total_early_cutoffs = valid_cutoffs_mask.sum(dim=1)
    return total_early_cutoffs, valid_cutoffs_mask

def evaluate_latency_parallel(cfg, sigmoid_out, label, turn_id, min_threshold, label_data, aligned_labels):
    sigmoid_out = sigmoid_out.detach().cpu()
    thresholds = torch.linspace(cfg.infer_params.threshold_range[0], cfg.infer_params.threshold_range[1], cfg.infer_params.threshold_range[2])
    thresholded_sigmoid_out = sigmoid_out > thresholds[:, None, None] #num_thresholds x num_frames x num_horizons
    latency_list, index_list = [], []

    ##ok need better algorithm, basically i need a list of tuples indicating start,end of each turn = turn_id
    match_turns = aligned_labels == turn_id
    turn_segments = []
    in_turn = False
    for i in range(len(match_turns)):
        if match_turns[i] and not in_turn:
            turn_start = i
            in_turn = True
        elif not match_turns[i] and in_turn:
            turn_end = i - 1
            turn_segments.append((turn_start, turn_end))
            in_turn = False
    if in_turn:
        turn_end = len(match_turns) - 1
        turn_segments.append((turn_start, turn_end))
    
    ##[(37, 40), (86, 108), (151, 201), ...
    intervals = (torch.tensor(cfg.data.label_params.forecast_intervals_ms) * cfg.data.audio_params.freq // 1000).long()
    #tensor([ 2.,  4.,  8., 16., 24., 32.])
    interval_mask = torch.zeros(intervals.max().item(), intervals.shape[0])
    # print(interval_mask.shape, intervals.shape)
    for i in range(intervals.shape[0]):
        interval_mask[-intervals[i]:, i] = 1 ## for 2 x 4 [ [0, 0, 1, 1], [1, 1, 1, 1] ]
    interval_mask = interval_mask.unsqueeze(0) #1 x 32 x num_horizons
    # print(interval_mask.shape)
    results = []
    skipped, used = 0,0

    # fig, ax = plt.subplots(7, 1, figsize=(20, 8))
    # current_start = 0

    for turn_idx, (turn_start, turn_end) in enumerate(turn_segments):
        turn_length = turn_end - turn_start + 1 ##turn_end - first, turn_start - last turn ids eg: 20, 28 - 9 vals
        # print(turn_length, intervals) #5 tensor([ 2,  4,  8, 16, 32])
        # exit()
        turn_length_sec = (turn_length / cfg.data.audio_params.freq)
        if turn_length_sec < cfg.infer_params.min_turn_length:
            skipped += 1
            continue
        used += 1
        # turn_length_sec = 
        turn_predictions = thresholded_sigmoid_out[:, turn_start:turn_end+1, :] #num_thresholds x turn_length x num_horizons
        ##now let us have 2 tensors based on masking - one where intervals are masked out from the end and one where they are not
        ##let us first design interval mask, it should be num_thresholds x num_turn_frames x num_horizons where the last k[i] horizons are masked for each frame i
        ##if mask is longer than turn length, we need to crop it
        turn_interval_mask = interval_mask[:, -turn_length:, :] #1 x turn_length x num_horizons - truncate when interval is larger than turn
        if turn_interval_mask.shape[1] < turn_predictions.shape[1]: #when segment is larger than mask
            ##need to pad
            pad_length = turn_predictions.shape[1] - turn_interval_mask.shape[1]
            padded_mask = torch.zeros((1, pad_length, turn_interval_mask.shape[2]))
            turn_interval_mask = torch.cat([padded_mask, turn_interval_mask], dim=1) #if segment length is 5 [ [0, 0, 0, 1, 1], [0, 1, 1, 1, 1] ]
        possible_error_frames = turn_length - turn_interval_mask.sum(1).squeeze()
        no_of_possible_error_frames = torch.ceil(possible_error_frames / intervals) # num_horizons
        # print(turn_length, intervals, possible_error_frames, no_of_possible_error_frames)
        # 23 tensor([ 2,  4,  8, 16, 32]) tensor([ 2.,  4.,  8., 16., 23.])
        # 50 tensor([ 2,  4,  8, 16, 32]) tensor([ 2.,  4.,  8., 16., 32.])
        # 49 tensor([ 2,  4,  8, 16, 32]) tensor([ 2.,  4.,  8., 16., 32.])
        # 50 tensor([ 2,  4,  8, 16, 32]) tensor([ 2.,  4.,  8., 16., 32.])
        # 42 tensor([ 2,  4,  8, 16, 32]) tensor([ 2.,  4.,  8., 16., 32.])
        # 24 tensor([ 2,  4,  8, 16, 32]) tensor([ 2.,  4.,  8., 16., 24.])
        non_interval_predictions = turn_predictions & (~turn_interval_mask.bool()) ##mask out interval predictions - this is where cutoff occurs ##num_thresh x turn_len x num_horizons
        interval_predictions = turn_predictions & turn_interval_mask.bool() ## num_thresh x turn_length x num_horizons ##interval regions
        total_early_cutoffs, working_mask = find_all_valid_cutoffs(non_interval_predictions, intervals)
        first_non_interval_has_true = non_interval_predictions.any(dim=1) #first cutiff

        # x_vals = torch.arange(current_start, current_start+turn_length)
        
        # ax[1].plot(x_vals, turn_predictions[0, :, 0])
        # ax[0].plot(x_vals, turn_interval_mask[0, :, 0])
        # ax[2].plot(x_vals, non_interval_predictions[0, :, 0])
        # ax[3].plot(x_vals, working_mask[0, :, 0])
        
        # print(first_non_interval_has_true[0])
        first_non_interval = non_interval_predictions.float().argmax(dim=1) #num_thresholds x num_horizons
        # ax[2].scatter(current_start+first_non_interval[0][0], non_interval_predictions[0, :, 0][current_start+first_non_interval[0][0]])
        # ax[4].plot(x_vals, first_non_interval_has_true[0, :, 0])
        # print(first_non_interval[0])
        first_non_interval = torch.where(first_non_interval_has_true, first_non_interval, torch.full_like(first_non_interval, -1)) ##identify first trigger location
        # print(first_non_interval)
        # print(first_non_interval != -1)
        denom = no_of_possible_error_frames[None, :]
        proportion_of_total_early_cutoffs = torch.where(
            denom != 0, 
            total_early_cutoffs / denom, 
            torch.zeros_like(total_early_cutoffs)
        )
        # proportion_of_total_early_cutoffs = total_early_cutoffs / no_of_possible_error_frames[None, :]
        # print(total_early_cutoffs, proportion_of_total_early_cutoffs, no_of_possible_error_frames)
        # ax[5].plot(x_vals, first_non_interval_has_true[0, :, 0])
        # ax[6].plot(x_vals, interval_predictions[0, :, 0])
        first_interval_has_true = interval_predictions.any(dim=1)    
        first_interval = interval_predictions.float().argmax(dim=1) #num_thresholds x num_horizons
        first_interval = torch.where(first_interval_has_true, first_interval, torch.full_like(first_interval, -1))
        # print(turn_length, turn_start, turn_end)
        # print(first_interval.squeeze())
        # current_start += turn_length + 2
        # print(first_non_interval.min(), first_non_interval.max(), first_interval.min(), first_interval.max(), turn_predictions.shape)
        # print()
        # print(first_interval.squeeze(), turn_length)
        interval_forecast = turn_predictions.shape[1] - (first_interval)
        # print(interval_forecast.squeeze())
        # print(interval_forecast.squeeze())
        interval_forecast[interval_forecast > turn_predictions.shape[1]] = 0
        # print(interval_forecast.squeeze())
        # print('---')
        # print(interval_forecast.squeeze())
        # print(interval_forecast)
        # print(first_interval)
        # print(first_non_interval)
        # print(turn_predictions.shape)
        # print('---')
        results.append({
            "turn_idx": turn_idx,
            "turn_start": turn_start,
            "turn_end": turn_end,
            "first_non_interval": first_non_interval,
            "first_non_interval_binary": first_non_interval != -1,
            "first_interval": first_interval,
            "turn_shape": turn_predictions.shape,
            "interval_forecast": interval_forecast,
            "total_early_cutoffs": total_early_cutoffs,
            "proportion_of_total_early_cutoffs": proportion_of_total_early_cutoffs,
        })
    # plt.tight_layout()
    # plt.savefig("plots/fc_metrics.png")
    # plt.close()
    # exit()
    # print("Skipped: ", skipped, "Used: ", used)
    return results
    