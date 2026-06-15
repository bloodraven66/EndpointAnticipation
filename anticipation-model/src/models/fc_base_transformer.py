import torchaudio
import torch.nn as nn
import torch
from src.utils.logger import logger
from .model import Model
import torch.nn.functional as F
from .fc_base_lstm import FC_LSTM_Model

class FC_Transformer_Model(FC_LSTM_Model):
    def __init__(
        self, 
        feat_extractor=None,
        use_system_embed=False,
        num_infer_chunk_samples=None,
        forecast_intervals=None,
        **kwargs
    ):
        super(FC_LSTM_Model, self).__init__(**kwargs)
        self.model1 = self.make_transformer(kwargs)
        self.two_stream = False
        if kwargs.get("two_stream", False):
            logger.info("Using two stream LSTM model for FC_LSTM_Model")
            self.model2 = self.make_transformer(kwargs)
            self.two_stream = True
            self.linear = nn.Linear(kwargs["hidden_size"]*2, kwargs["output_size"])
        else:
            self.linear = nn.Linear(kwargs["hidden_size"], kwargs["output_size"])
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
            logger.info("Using feature extractor in FC_Transformer_Model")
            ##detach gradients from feature extractor
            for param in self.feat_extractor.model.parameters():
                param.requires_grad = False
        self.num_infer_chunk_samples = num_infer_chunk_samples
        self.forecast_intervals = torch.from_numpy(forecast_intervals)
        self.stream_drop = kwargs.get("stream_drop", None)
        if self.stream_drop is not None:
            self.drop_replace_params = nn.Parameter(torch.randn(kwargs["hidden_size"]))
        self.single_output_head = False
        self.current_forecast_interval_index = None
        if "single_head" in kwargs and kwargs["single_head"]:
            logger.info(f"Setting single head and forecast embedding in model for intervals {forecast_intervals}")
            self.single_output_head = True
            self.forecast_embedding = nn.Embedding(len(forecast_intervals), kwargs["hidden_size"])

    def forward(self, x, label, h=None, c=None, init_hidden=False, system_ids=None):
        if self.feat_extractor is not None:
            with torch.no_grad():
                # print("incoming", x.shape)
                x = x.reshape(-1, x.size(2))  # (bs x num_channels) xxw T_frames
                # print("reshaped", x.shape)
                x = self.feat_extractor(x) # (bs x 2) x feat_dim x T_frames
                # print(x.shape)
                x = x[:, :, :label.size(1)]  # Align feature length with labels
                # print(x.shape)
                x = x.reshape(label.size(0), -1, x.size(1), x.size(2))  # bs x num_channels x feat_dim x T_frames
                # print(x.shape)
        x = self._forward(x, h, c, init_hidden=False, system_ids=system_ids, cross_attention_src=None)
        if hasattr(self, "activation"):
            x = self.activation(x)
        return x

    def infer(self, x, label, h=None, c=None, init_hidden=False, system_ids=None, infer_forecast_labels=None):
        if self.feat_extractor is not None:
            with torch.no_grad():
                x = torch.cat(x, dim=0)
                x = x.reshape(-1, x.size(2))  # (bs x num_channels) xxw T_frames
                x = self.feat_extractor(x, chunk_length=self.num_infer_chunk_samples) # (bs x 2) x feat_dim x T_frames
                x = x[:, :, :label.size(1)]  # Align feature length with labels
                x = x.reshape(label.size(0), -1, x.size(1), x.size(2))  # bs x num_channels x feat_dim x T_frames
        x = self._infer(x, h, c, init_hidden=False, system_ids=system_ids, cross_attention_src=None, infer_forecast_labels=infer_forecast_labels)
        if hasattr(self, "activation"):
            x = self.activation(x)
        return x
