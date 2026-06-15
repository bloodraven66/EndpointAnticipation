import os
import random
import json
import torch
import numpy as np
from tqdm import tqdm
from src.utils import fc_metrics
from src.utils.logger import logger
from sklearn.metrics import confusion_matrix
from src.utils.wandb_logger import WandbLogger
from .default_trainer import DefaultTrainer

from src.utils.run_utils import (
    load_checkpoint, 
    save_checkpoint, 
    setup_save_folder, 
    prediction_visualization,
    prediction_visualization_forecasting,
    plot_and_save_fc_metrics,
)

from src.utils.data_utils import (
    get_id_to_token_mapping, 
    get_token_to_id_mapping, 
)

class ForecastingTrainer(DefaultTrainer):
    def __init__(self, cfg, model, config_paths):
        super(ForecastingTrainer, self).__init__(cfg, model, config_paths)
    
    def handle_start_of_epoch(self, mode):
        self.mode = mode
        if mode == "train": 
            self.model.train()
        else: 
            self.model.eval()
        self.epoch_metrics = {}
        num_horizons = len(self.cfg.data.label_params.forecast_intervals_ms)
        self.cm = {"gnd": [[] for _ in range(num_horizons)], 
               "pred": [[] for _ in range(num_horizons)]}
    
    def handle_system(self, label, aligned_labels):
        if hasattr(self.cfg.model, "mask_system_tokens"):
            if self.cfg.model.mask_system_tokens:
                system_id = get_token_to_id_mapping(self.cfg)[self.cfg.data.special_tokens.system]
                system_id_mask = aligned_labels == system_id
                label[system_id_mask] = -1
                system_end_id = get_token_to_id_mapping(self.cfg)[self.cfg.data.special_tokens.system_end]
                system_end_id_mask = aligned_labels == system_end_id
                label[system_end_id_mask] = -1
        return label
        
    
    def train(self, mode, loader):
        self.handle_start_of_epoch(mode)
        system_ids = None
        pbar = tqdm(loader, desc=f"{mode}")
        for idx, data in enumerate(pbar):
            # if idx > 10:
            #     break
            data, label, metadata = data
            data, label = self.to_device([data, label])
            if hasattr(self.cfg.model, "use_system_ip_embed") and self.cfg.model.use_system_ip_embed:
                system_ids = self.get_system_ids(label)
            # print(self.cfg.data.label_params)
            if self.model.single_output_head:
                
                if hasattr(self.cfg.data.label_params, "continual") and self.cfg.data.label_params.continual is not None:
                    assert self.cfg.data.label_params.continual in ["increasing", "decreasing"], f"{self.cfg.data.label_params}"
                    choices = min(self.epoch, len(self.cfg.data.label_params.forecast_intervals_ms)-1) #0, 1, 2, 3, 4, 4, 4, 4
                    if self.cfg.data.label_params.continual == "increasing":
                        self.model.current_forecast_interval_index = random.randint(0, choices)
                    else:
                        self.model.current_forecast_interval_index = random.randint(len(self.cfg.data.label_params.forecast_intervals_ms)-1-choices, len(self.cfg.data.label_params.forecast_intervals_ms)-1) #(4,4), (3, 4), .. (0, 4) 
                    # print(choices, self.model.current_forecast_interval_index)
                else:
                    self.model.current_forecast_interval_index = random.randint(0, len(self.cfg.data.label_params.forecast_intervals_ms)-1)
                label = label[:, :, self.model.current_forecast_interval_index].unsqueeze(-1)

            output = self.model(data, label, init_hidden=True, system_ids=system_ids)
            label_, delay_frames = self.handle_delay(label)
            label_ = self.handle_system(label_, metadata[-1])
            loss, label_info = self.model.loss(output, label_, delay_frames)
            if mode == "train":
                self.optimizer.zero_grad()
                loss["total"].backward()   
                self.optimizer.step()      
            pbar.set_description(f"Mode: {mode}, Loss: {loss['total'].item():.4f}, Acc: {loss['accuracy'].item():.4f}")  
            self.handle_batch_loss(loss, label_info)
            # break
        self.handle_end_of_epoch(data, label, output, metadata)
        return {"acc": self.avg_user_accuracy, "loss": self.epoch_metrics[f"{mode}_total"]}

    def handle_end_of_epoch(self, data, label, output, metadata, idx=0):
        for key in self.epoch_metrics:
            self.epoch_metrics[key] = round(sum(self.epoch_metrics[key]) / len(self.epoch_metrics[key]), 4)
        logger.info(f"{self.mode} epoch: {self.epoch_metrics}")
        self.wandb_logger.log(self.epoch_metrics)  
        self.avg_user_accuracy = self.epoch_metrics[f"{self.mode}_accuracy"]

        if self.mode != "val":
            return
        
        prediction_visualization_forecasting(
            self.cfg, 
            data, 
            label, 
            output, 
            metadata,
            self.save_folder, 
            self.wandb_logger,
            idx=idx
        )
    def infer_loop(self, loader, ):
        interval_forecast, cutoffs, total_cutoffs, total_cutoff_proportions = {}, {}, {}, {}
        system_ids = None
        thresholds = torch.linspace(self.cfg.infer_params.threshold_range[0], self.cfg.infer_params.threshold_range[1], self.cfg.infer_params.threshold_range[2])
        for idx, data in enumerate(tqdm(loader, desc="Infer")):
            # if idx > 5:
                # break
            if len(data) == 3:
                data, label, metadata = data
                codec_probs, codec_entropy = None, None
            elif len(data) == 4:
                data, label, metadata, (codec_probs, codec_entropy) = data
            else:
                raise NotImplementedError(f"Data length not implemented: {len(data)}")
            data, label = self.to_device([data, label])
            if hasattr(self.cfg.model, "use_system_ip_embed") and self.cfg.model.use_system_ip_embed:
                system_ids = self.get_system_ids(label)
            if self.model.single_output_head:
                output = self.model.infer(data, label, init_hidden=True, system_ids=system_ids, infer_forecast_labels=range(len(self.cfg.data.label_params.forecast_intervals_ms)))
                output = output.squeeze().T[None, :, :]
            else:
                output = self.model.infer(data, label, init_hidden=True, system_ids=system_ids)
            assert output.size(-1) == len(self.cfg.data.label_params.forecast_intervals_ms), f"Output size mismatch: {output.size(-1)} != {len(self.cfg.data.label_params.forecast_intervals_ms)}"
            audio, label_data, texts, start_time, end_time, key, aligned_labels = metadata
            total_time = end_time - start_time
            num_output_frames = output.size(1)
            num_frames_per_second = round(num_output_frames / total_time.item())
            min_forecast = 0
            assert len(self.cfg.infer_params.score_turns) == 1, f"Score turns not implemented for more than 1 turn - {self.cfg.infer_params.score_turns}, {len(self.cfg.infer_params.score_turns)}"
            turn = self.cfg.infer_params.score_turns[0]
            turn_id = get_token_to_id_mapping(self.cfg)[turn]
            system_ids = self.get_system_ids(label)
            forecast_data = fc_metrics.evaluate_latency_parallel(
                self.cfg, 
                output, 
                label.squeeze(), 
                turn_id, 
                min_forecast, 
                label_data,
                aligned_labels.squeeze()
            )
            
            if forecast_data is None:
                continue
            
            if idx < self.cfg.infer_params.num_infer_pred_imgs:
                prediction_visualization_forecasting(
                    self.cfg, 
                    data, 
                    label, 
                    output, 
                    metadata,
                    self.cfg.infer_folder, 
                    None,
                    idx=0,
                    forecast_data=forecast_data,
                    fig_size=(120, 10),
                    save_name=f"infer_prediction{idx+1}.png",
                    save_data_name=f"infer_prediction_data{idx+1}.pt",
                    plot_pitch=False,
                    probs_and_entropy=(codec_probs, codec_entropy),
                )
            # exit()
            for forecast_data_item in forecast_data:
                for threshold_idx, threshold in enumerate(thresholds):
                    threshold = round(threshold.item(), 4)
                    if threshold not in interval_forecast:
                        interval_forecast[threshold] = []
                        cutoffs[threshold] = []
                        total_cutoffs[threshold] = []
                        total_cutoff_proportions[threshold] = []
                    # print(forecast_data_item.keys())
                    #turn_idx', 'turn_start', 'turn_end', 'first_non_interval', 'first_non_interval_binary', 'first_interval', 'turn_shape', 'interval_forecast'
                    # print(forecast_data_item["interval_forecast"][threshold_idx])
                    interval_forecast[threshold].append(forecast_data_item["interval_forecast"][threshold_idx].cpu().numpy())
                    cutoffs[threshold].append(forecast_data_item["first_non_interval_binary"][threshold_idx].cpu().numpy())
                    total_cutoffs[threshold].append(forecast_data_item["total_early_cutoffs"][threshold_idx].cpu().numpy())
                    total_cutoff_proportions[threshold].append(forecast_data_item["proportion_of_total_early_cutoffs"][threshold_idx].cpu().numpy())
            # exit()
        total_ep_cutoffs_mean, total_ep_cutoffs_median, ep_cutoff, median_forecast, worst_case_forecast, total_cutoff_proportions_mean, accuracies = {}, {}, {}, {}, {}, {}, {}
        all_horizons = np.array(self.cfg.data.label_params.forecast_intervals_ms) / (1000 / self.cfg.data.audio_params.freq)
        all_horizons_with_collar = (all_horizons - self.cfg.infer_params.infer_accuracy_collar_frames)[None, :] # 1 x num_horizons
        logger.info(f"{self.cfg.data.label_params.forecast_intervals_ms}, {all_horizons}")
        for threshold in interval_forecast:
            raw_forecasts = np.array(interval_forecast[threshold])
            correct_with_collar = raw_forecasts >= all_horizons_with_collar
            accuracies[threshold] = (correct_with_collar.sum(0) / len(correct_with_collar)).round(2)
            # print(accuracy)
            forecast = np.round(raw_forecasts / self.cfg.data.audio_params.freq, 4)


            cutoff = np.round(np.array(cutoffs[threshold]) * 100, 4)
            total_cutoff = np.array(total_cutoffs[threshold])
            total_cutoff_proportion = np.array(total_cutoff_proportions[threshold])
            # print(total_cutoff.shape)
            # exit()
            total_ep_cutoffs_mean[threshold] = np.mean(total_cutoff, axis=0).round(4)
            total_ep_cutoffs_median[threshold] = np.median(total_cutoff, axis=0)
            total_cutoff_proportions_mean[threshold] = np.mean(total_cutoff_proportion, axis=0).round(4)
            ep_cutoff[threshold] = cutoff.sum(0) / cutoff.shape[0]
            median_forecast[threshold] = (np.median(forecast, axis=0) * 1000).round()
            # worst_case_forecast[threshold] = np.percentile(forecast, 10, axis=0)
            print(f"Threshold: {threshold}, Cutoff%: {ep_cutoff[threshold]}", f"Median Latency (ms): {median_forecast[threshold]}, Total Cutoff Mean: {total_ep_cutoffs_mean[threshold]}, Total Cutoff median: {total_ep_cutoffs_median[threshold]}")
        # exit()
        return ep_cutoff, median_forecast, total_ep_cutoffs_mean, total_ep_cutoffs_median, total_cutoff_proportions_mean, accuracies, self.cfg.data.label_params.forecast_intervals_ms
        

    def infer(self, loader, chk_folder):        
        
        self.model = load_checkpoint(
            path=os.path.join(chk_folder, self.cfg.infer_params.infer_checkpoint_name),
            model=self.model
        )

        self.model.eval()

        stream_configs = [(None, "")]
        if hasattr(self.cfg.model_params, "stream_drop"):
            if self.cfg.model_params.stream_drop is not None:
                stream_configs = [
                    ("both", self.cfg.infer_folder + "_use_both"),
                    ("single", self.cfg.infer_folder + "_use_one")
                ]
            
        logger.info(f"Using configs - {stream_configs}")
        
        for stream_config in stream_configs:
            stream_status, infer_folder_path = stream_config
            logger.info(f"Using inference stream config - {stream_status}")
            if stream_status is not None:
                if stream_status == "both":
                    self.model.stream_drop = None
                elif stream_status == "single":
                    self.model.stream_drop = 1
                else:
                    raise NotImplementedError
                self.cfg.infer_folder = infer_folder_path
            
            os.makedirs(self.cfg.infer_folder, exist_ok=True)
            

            with torch.no_grad():
                ep_cutoff, median_forecast, total_ep_cutoffs_mean, total_ep_cutoffs_median, total_cutoff_proportions_mean, accuracies, forecast_intervals = self.infer_loop(loader)
            score_metrics = {
                "median_forecast": median_forecast,
                "ep_cutoff": ep_cutoff,
                "total_ep_cutoffs_mean": total_ep_cutoffs_mean,
                "total_ep_cutoffs_median": total_ep_cutoffs_median,
                "forecast_intervals": forecast_intervals,
                "total_cutoff_proportions_mean": total_cutoff_proportions_mean,
                "accuracies_with_collar": accuracies,
            }
                
            plot_and_save_fc_metrics(
                metrics=score_metrics,
                cfg=self.cfg
            )

            

def load_forecasting_trainer(cfg, model, loaders, config_paths):
    
    trainer = ForecastingTrainer(cfg, model, config_paths)
    
    if hasattr(cfg, "infer_params"):
        infer_folder = cfg.infer_folder
        for (name, test_loader) in loaders["test"]:
            cfg.infer_folder = infer_folder + "__" + name
            logger.info(f"Starting infer for {name} - {cfg.infer_folder}")
            trainer.infer(loader=test_loader, chk_folder=infer_folder)
        return
        
    train_loader = loaders["train"]
    dev_loader = loaders["val"]
    
    if cfg.run_params.load_checkpoint:
        load_checkpoint(
            path=cfg.run_params.load_checkpoint_path,
            model=trainer.model,
        )
    results = []
    for epoch in tqdm(range(cfg.run_params.epochs)):
        trainer.epoch = epoch
        r = {"epoch": epoch}
        # print(r)
        trainer.train(mode="train", loader=train_loader)
        r = {**r, **trainer.epoch_metrics}
        # print(r)
        val_metric = trainer.train(mode="val", loader=dev_loader)
        r = {**r, **trainer.epoch_metrics}
        # print(r)
        results.append(r)
        trainer.checkpoint_handler(val_metric)
        # print(results)
        with open(os.path.join(trainer.save_folder, "train.json"), "w") as f:
            json.dump(results, f, indent=4)