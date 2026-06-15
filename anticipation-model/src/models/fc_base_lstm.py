import torchaudio
import torch.nn as nn
import torch
from src.utils.logger import logger
from .model import Model
import torch.nn.functional as F
from .base_lstm import LSTM_Model

class FC_LSTM_Model(LSTM_Model):
    def __init__(
        self, 
        feat_extractor=None,
        use_system_embed=False,
        num_infer_chunk_samples=None,
        forecast_intervals=None,
        **kwargs
    ):
        super(FC_LSTM_Model, self).__init__(**kwargs)
        self.model1 = self.make_lstm(kwargs)
        self.two_stream = False
        if kwargs.get("two_stream", False):
            logger.info("Using two stream LSTM model for FC_LSTM_Model")
            self.model2 = self.make_lstm(kwargs)
            self.two_stream = True
            self.linear = nn.Linear(kwargs["hidden_size"]*2, kwargs["output_size"])
        else:
            self.linear = nn.Linear(kwargs["hidden_size"], kwargs["output_size"])
        if use_system_embed:
            self.system_embed = nn.Embedding(2, kwargs["input_size"])
        if kwargs.get("loss_fn", "") == "frame_level_binary_cross_entropy":
            logger.info("Using Sigmoid activation for binary cross-entropy loss")
            self.activation = nn.Sigmoid()
            self.loss_fn = nn.BCELoss()
        self.loss_weight_factors = kwargs.get("loss_weight_factors", None)
        if self.loss_weight_factors is not None:
            logger.info(f"Using loss weight factors: {self.loss_weight_factors}")
            self.loss_fn = nn.BCELoss(reduction='none')
        self.asymmetric_loss_weight_factors = kwargs.get("asymmetric_loss_weight_factors", None)
        if self.asymmetric_loss_weight_factors is not None:
            logger.info(f"Using asymmetric loss weight factors: {self.asymmetric_loss_weight_factors}")
            self.loss_fn = nn.BCELoss(reduction='none')
        self.feat_extractor = feat_extractor
        if feat_extractor is not None:
            logger.info("Using feature extractor in FC_LSTM_Model")
            ##detach gradients from feature extractor
            for param in self.feat_extractor.model.parameters():
                param.requires_grad = False
        self.num_infer_chunk_samples = num_infer_chunk_samples
        self.forecast_intervals = torch.from_numpy(forecast_intervals)
        self.stream_drop = kwargs.get("stream_drop", None)
        if self.stream_drop is not None:
            self.drop_replace_params = nn.Parameter(torch.randn(kwargs["hidden_size"]))


    def forward(self, x, label, h=None, c=None, init_hidden=False, system_ids=None):
        if self.feat_extractor is not None:
            with torch.no_grad():
                x = x.reshape(-1, x.size(2))  # (bs x num_channels) xxw T_frames
                x = self.feat_extractor(x) # (bs x 2) x feat_dim x T_frames
                x = x[:, :, :label.size(1)]  # Align feature length with labels
                x = x.reshape(label.size(0), -1, x.size(1), x.size(2))  # bs x num_channels x feat_dim x T_frames
        x = self._forward(x, h, c, init_hidden, system_ids)
        if hasattr(self, "activation"):
            x = self.activation(x)
        return x

    def infer(self, x, label, h=None, c=None, init_hidden=False, system_ids=None):
        if self.feat_extractor is not None:
            with torch.no_grad():
                x = torch.cat(x, dim=0)
                x = x.reshape(-1, x.size(2))  # (bs x num_channels) xxw T_frames
                x = self.feat_extractor(x, chunk_length=self.num_infer_chunk_samples) # (bs x 2) x feat_dim x T_frames
                x = x[:, :, :label.size(1)]  # Align feature length with labels
                x = x.reshape(label.size(0), -1, x.size(1), x.size(2))  # bs x num_channels x feat_dim x T_frames
        x = self._infer(x, h, c, init_hidden, system_ids)
        if hasattr(self, "activation"):
            x = self.activation(x)
        return x

    def apply_temporal_window(self, gnd):
        ### NOTE: Test correctness of this function in edgecases (multiple turns within temporal window, start/end of sequence)
        B, num_horizons, T = gnd.shape
        print("gnd shape in temporal window:", gnd.shape)
        temporal_mask = torch.zeros_like(gnd, dtype=torch.bool)
        
        # For each sample and horizon, find forecast frames and create window
        for b in range(B):
            for h in range(num_horizons):
                # Find frames where forecast is active (gnd == 1)
                forecast_frames = torch.where((gnd[b, h] == 1) & (gnd[b, h] != -1))[0]
                
                if len(forecast_frames) > 0:
                    # Create window around each forecast frame
                    for frame_idx in forecast_frames:
                        start = max(0, frame_idx - self.temporal_window)
                        end = min(T, frame_idx + self.temporal_window + 1)
                        temporal_mask[b, h, start:end] = True
        
        # Combine valid mask with temporal mask
        valid_mask = valid_mask & temporal_mask
        return valid_mask
    
    def apply_asymmetric_loss_weight_factors(self, gnd, valid_mask, loss):
        ##so we need to create a weight matrix where
        ##f labels are 0 0 0 0 1    1   1    1 0 0 0 0
        ## matrix is 1 1 1 1 2 1.75 1.5 1.25 1 1 1 1 1
        ##where weights are uniformly decreasing from max_weight to min_weight for forecast, and a constant weight for non-forecast frames
        forecast_mask = (gnd == 1)
        non_forecast_weights = torch.ones_like(gnd) * self.asymmetric_loss_weight_factors["non_forecast_weight"]
        ## so we use 0 0 0 0 1 1 1 1 0 0 0 0 and we need a transmation to get 0 0 0 0 2 1.75 1.5 1.25 1 0 0 0 0
        forecast_loss_weights = torch.ones_like(gnd)
        for i in range(self.forecast_intervals.shape[0]):
            ramp = torch.linspace(
                self.asymmetric_loss_weight_factors["max_forecast_weight"], 
                self.asymmetric_loss_weight_factors["min_forecast_weight"], 
                self.forecast_intervals[i]
            )
            current_interval_mask = gnd[:, :, i] == 1
            # print(gnd.shape, current_interval_mask.shape, ramp.shape)
            # exit()
            kernel = ramp.view(1, 1, -1).to(gnd.device)
            padded_mask = F.pad(current_interval_mask.float(), (1, 0), value=0)
            diff = padded_mask[:, 1:] - padded_mask[:, :-1]
            starts = (diff == 1).float()
            output = F.conv_transpose1d(
                starts.unsqueeze(1),  # Add channel dim: [Batch, 1, Length]
                kernel, 
                stride=1
            )
            outputs = output.squeeze(1)[:, :current_interval_mask.shape[1]]

            # for j in range(500):
                # print(outputs[0, j], gnd[0, j].squeeze())

            forecast_loss_weights[:, :, i] = torch.where(forecast_mask[:, :, i], outputs, non_forecast_weights[:, :, i])
        forecast_loss_weights = forecast_loss_weights[valid_mask]
        loss = (loss * forecast_loss_weights).mean()
        return loss
        # for j in range(500):
            # print(forecast_loss_weights[0, j].squeeze(), gnd[0, j].squeeze())
        # exit()

        ##so we need to create a weight matrix where
    def apply_loss_weights(self, gnd, valid_mask, loss):
        if self.loss_weight_factors is not None:
            loss = self.apply_loss_weight_factors(gnd, valid_mask, loss)
        elif self.asymmetric_loss_weight_factors is not None:
            loss = self.apply_asymmetric_loss_weight_factors(gnd, valid_mask, loss)
        return loss

    def apply_loss_weight_factors(self, gnd, valid_mask, loss):
        forecast_mask = (gnd == 1)
        forecast_weights = torch.ones_like(gnd) * self.loss_weight_factors["forecast_frames"]
        non_forecast_weights = torch.ones_like(gnd) * self.loss_weight_factors["non_forecast_frames"]
        weight_matrix = torch.where(forecast_mask, forecast_weights, non_forecast_weights)
        valid_weights = weight_matrix[valid_mask]
        loss = (loss * valid_weights).mean()
        return loss
    

    def loss(self, pred, gnd, delay_frames=None):
        """
        pred: (B, num_horizons, T) - logits for each forecast horizon
        gnd: (B, T, num_horizons) - binary targets for each horizon
        """
        # print(gnd.shape)
        # exit()
        # pred is (B, num_horizons, T), need to permute to (B, T, num_horizons)
        pred = pred.permute(0, 2, 1)  # Now (B, T, num_horizons)
        
        if delay_frames is not None:
            pred = pred[:, delay_frames:, :]  # (B, T-delay, num_horizons)
            gnd = gnd[:, delay_frames:, :]     # (B, T-delay, num_horizons)
        pred = pred.permute(0, 2, 1)  # (B, T, num_horizons)
        gnd = gnd.float()
        valid_mask = (gnd != -1)
        if hasattr(self, 'temporal_window') and self.temporal_window is not None:
            valid_mask = self.apply_temporal_window(gnd)

        valid_pred = pred[valid_mask]  # (num_valid,)
        valid_gnd = gnd[valid_mask]

        # BCE loss expects logits, will apply sigmoid internally
        loss = self.loss_fn(valid_pred, valid_gnd)  # Should be nn.BCELoss
        loss = self.apply_loss_weights(gnd, valid_mask, loss)
        
        # Apply sigmoid for predictions
        pred_binary = (pred > 0.5).float()  # Threshold at 0.5

        num_horizons = pred.shape[2]    
        pred_per_horizon = []
        gnd_per_horizon = []
        acc_ = {}
        for h in range(num_horizons):
            mask_h = valid_mask[:, :, h]
            pred_h = pred_binary[:, :, h][mask_h].detach().cpu().tolist()
            gnd_h = gnd[:, :, h][mask_h].detach().cpu().tolist()
            pred_per_horizon.append(pred_h)
            gnd_per_horizon.append(gnd_h)
            horizon_accuracy = (pred_binary[:, :, h][mask_h] == gnd[:, :, h][mask_h]).float().mean()
            acc_[f"{h}"] = horizon_accuracy
        accuracy = sum(acc_[f"{h}"] for h in range(num_horizons)) / num_horizons
        loss_dict = {"total": loss, "accuracy": accuracy}
        loss_dict.update(acc_)
        return loss_dict, (gnd_per_horizon, pred_per_horizon)
    