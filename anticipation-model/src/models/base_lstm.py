import torchaudio
import random
import torch.nn as nn
import torch, random
from src.utils.logger import logger
from .model import Model

class LSTM_Model(Model):
    def __init__(
        self, 
        use_system_embed=False,
        **kwargs
    ):
        super(LSTM_Model, self).__init__(**kwargs)
        # self.model1 = self.make_lstm(kwargs)
        # self.linear = nn.Linear(kwargs["hidden_size"], kwargs["output_size"])
        if use_system_embed:
            self.system_embed = nn.Embedding(2, kwargs["input_size"])
    
    def forward(self, x, h=None, c=None, init_hidden=False, system_ids=None):
        return self._forward(x, h, c, init_hidden, system_ids)
    
    def _forward(self, x, h=None, c=None, init_hidden=False, system_ids=None, **kwargs):
        # print(x.shape)
        # exit()
        forecast_embed = None
        if self.current_forecast_interval_index is not None:
            forecast_embed = self.forecast_embedding(torch.tensor(self.current_forecast_interval_index).to(x.device))[None, None, :].repeat(x.shape[0], x.shape[3], 1)

        if init_hidden:
            h, c = self.init_lstm_hidden(x.size(0), x.device)
        if self.two_stream:
            x1 = x[:, 0, :, :].permute(0, 2, 1)
            if self.stream_drop is not None:
                if random.random() > self.stream_drop:
                    x2 = x[:, 1, :, :].permute(0, 2, 1)
                else:
                    x2 = None
            else:
                x2 = x[:, 1, :, :].permute(0, 2, 1)
        else:
            x1 = x
            x = x.permute(0, 2, 1)
        if system_ids is not None and self.system_embed is not None:
            x = x + self.system_embed(system_ids)
        if forecast_embed is not None:
            x1 = forecast_embed + x1
        # x1, (_, _) = self.model1(x1, (h, c))
        x1, (_, _) = self.model_forward("model1", x1, (h, c))
        if self.two_stream:
            if x2 is not None:
                # x2, (_, _) = self.model2(x2, (h, c))
                x2, (_, _) = self.model_forward("model2", x2, (h, c))
            else:
                
                x2 = self.drop_replace_params[None, None, :].expand(x1.shape[0], x1.shape[1], -1)
            x = torch.cat([x1, x2], dim=2)
        else:
            x = x1
        x = self.linear(x)
        return x

    def _infer(self, x, h=None, c=None, init_hidden=False, system_ids=None, **kwargs):
        assert self.stream_drop in [1, None], f"Inference time stream drop should either be None or 1 - {self.stream_drop}"
        # assert x.size(0) == 1, "Inference only supports batch size of 1"
        forecast_embed = None
        # print(kwargs)
        if "infer_forecast_labels" in kwargs and hasattr(self, "forecast_embedding"):
            forecast_embed = self.forecast_embedding(torch.tensor(kwargs["infer_forecast_labels"]).to(x.device))
            # print(forecast_embed.shape, x.shape)
            forecast_embed = forecast_embed[:, None, :].repeat(1, x.shape[3], 1)
        # print(forecast_embed.shape, x.shape)
        if init_hidden:
            h, c = self.init_lstm_hidden(x.size(0), x[0].device)
        if self.two_stream:
            x1 = x[:, 0, :, :].permute(0, 2, 1)
            if self.stream_drop == 1:
                x2 = None
            else:
                x2 = x[:, 1, :, :].permute(0, 2, 1)
        else:
            x1 = x
            x = x.permute(0, 2, 1)
        if system_ids is not None and self.system_embed is not None:
            x = x + self.system_embed(system_ids)
        # print(x1.shape, h.shape, c.shape)
        if forecast_embed is not None:
            x1 = x1.repeat(forecast_embed.shape[0], 1, 1)
            x2 = x2.repeat(forecast_embed.shape[0], 1, 1)
            x1 = forecast_embed + x1
        x1, (_, _) = self.model_forward("model1", x1, (h, c))
        if self.two_stream:
            if x2 is not None:  
                x2, (_, _) = self.model_forward("model2", x2, (h, c))
            else:
                x2 = self.drop_replace_params[None, None, :].expand(x1.shape[0], x1.shape[1], -1)
            x = torch.cat([x1, x2], dim=2)
        else:
            x = x1
        x = self.linear(x)
        return x

    def infer(self, x, h=None, c=None, init_hidden=False):
        return self._infer(x, h, c, init_hidden)

    def infer_ar(self, x, h=None, c=None, init_hidden=False):
        assert x.size(0) == 1, "Inference only supports batch size of 1"
        num_decode_steps = x.size(2)
        if init_hidden:
            h, c = self.init_lstm_hidden(x.size(0), x.device)
        full_output = None
        for decode_idx in range(num_decode_steps):
            x_ = x[:, :, decode_idx].unsqueeze(2)
            x_ = x_.permute(0, 2, 1)
            x_, (h, c) = self.model(x_, (h, c))
            out = self.linear(x_)
            if full_output is None:
                full_output = out
            else:
                full_output = torch.cat([full_output, out], dim=1)
        return full_output
    