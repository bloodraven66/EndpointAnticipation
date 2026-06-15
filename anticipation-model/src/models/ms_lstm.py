import torchaudio
import torch.nn as nn
import torch
from src.utils.logger import logger
from .model import Model

class MS_LSTM_Model(Model):
    def __init__(
        self, 
        use_mel_embed=False,
        **kwargs
    ):
        super(MS_LSTM_Model, self).__init__(**kwargs)
        self.model1 = self.make_lstm(kwargs)
        self.model2 = self.make_lstm(kwargs)

        self.merge_method = "concat"
        self.merge_residual = False
        outsize = None
        if "merge" in kwargs:
            logger.info("Using merge method: {}".format(kwargs["merge"]))
            self.merge_method = kwargs["merge"]
        if "merge_residual" in kwargs:
            logger.info("Using merge residual: {}".format(kwargs["merge_residual"]))
            self.merge_residual = kwargs["merge_residual"]
            self.project_lstm_to_linear = nn.Linear(kwargs["hidden_size"], kwargs["input_size"])
            outsize = kwargs["input_size"]
        if outsize is None:
            outsize = kwargs["hidden_size"] * 2 if self.merge_method == "concat" else kwargs["hidden_size"]
        self.linear = nn.Linear(outsize, kwargs["output_size"])

    def forward(self, x, h=None, c=None, init_hidden=False):
        if init_hidden:
            h, c = self.init_lstm_hidden(x.size(0), x.device)
        x = x.permute(0, 1, 3, 2)
        x_residual = x[:, 0, :, :].clone()
        x1, (_, _) = self.model1(x[:, 0, :, :], (h, c))
        x2, (_, _) = self.model2(x[:, 1, :, :], (h, c))
        if self.merge_method == "concat":
            x = torch.cat([x1, x2], dim=-1)
        elif self.merge_method == "add":
            x = x1 + x2
        if self.merge_residual:
            x = self.project_lstm_to_linear(x)
            x = x + x_residual
        x = self.linear(x)
        return x
    
    def infer(self, x, h=None, c=None, init_hidden=False):
        if init_hidden:
            h, c = self.init_lstm_hidden(x.size(0), x.device)
        x = x.permute(0, 1, 3, 2)
        x_residual = x[:, 0, :, :].clone()
        
        x1, (_, _) = self.model1(x[:, 0, :, :], (h, c))
        x2, (_, _) = self.model2(x[:, 1, :, :], (h, c))
        if self.merge_method == "concat":
            x = torch.cat([x1, x2], dim=-1)
        elif self.merge_method == "add":
            x = x1 + x2
        if self.merge_residual:
            x = self.project_lstm_to_linear(x)
            x = x + x_residual
        x = self.linear(x)
        return x

    def infer_ar(self, x, h=None, c=None, init_hidden=False):
        raise NotImplementedError
    