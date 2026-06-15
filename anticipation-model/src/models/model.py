import torchaudio
import torch.nn as nn
import torch
from src.utils.logger import logger
from moshi.modules.transformer import StreamingTransformer

class Model(nn.Module):
    def __init__(
        self, **kwargs
    ):
        super(Model, self).__init__()
        self.mel_embed = None
        self.system_embed = None
        self.loss_fn = None
        if kwargs["loss_fn"] == "frame_level_cross_entropy":
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
            if hasattr(kwargs, "loss_weights"):
                logger.info(f"Using loss weights - {kwargs['loss_weights']}")
                self.loss_fn = nn.CrossEntropyLoss(weight=kwargs["loss_weights"], ignore_index=-1)
                
    def forward(self, x):
        raise NotImplementedError
    
    def infer(self, x):
        return NotImplementedError

    def infer_ar(self, x):
        return NotImplementedError

    def make_lstm(self, kwargs):
        model = nn.LSTM(
            input_size=kwargs["input_size"] if "project" not in kwargs else kwargs["project"],
            hidden_size=kwargs["hidden_size"],
            num_layers=kwargs["num_layers"],
            batch_first=True,
            dropout=kwargs["dropout"],
            bidirectional=kwargs["bidirectional"],
        )
        self.modelname = "lstm"
        return model
    
    def make_transformer(self, kwargs):
        model = StreamingTransformer(
            d_model=kwargs["hidden_size"],
            num_heads=kwargs["num_heads"],
            num_layers=kwargs["num_layers"],
            dim_feedforward=kwargs["dim_feedforward"],
            causal=True,
            context=kwargs["context"],
            positional_embedding=kwargs["positional_embedding"],
            max_period=kwargs["max_period"],
        )
        self.modelname = "transformer"
        return model
    
    def model_forward(self, model_, x, args):
        if self.modelname == "lstm":
            a = x, *args
        elif self.modelname == "transformer":
            a = x
        else:
            raise NotImplementedError
        if model_ == "model1":
            out = self.model1(a)
        elif model_ == "model2":
            out = self.model2(a)
        else:
            raise NotImplementedError
        if self.modelname == "lstm":
            return out
        else:
            return out, (None, None)
        


    def init_lstm_hidden(self, batch_size, device):
        h_o = torch.zeros(self.model1.num_layers, batch_size, self.model1.hidden_size)
        c_0 = torch.zeros(self.model1.num_layers, batch_size, self.model1.hidden_size)
        return h_o.to(device), c_0.to(device)
        
    def loss(self, pred, gnd, delay_frames=None):
        pred = pred.permute(0, 2, 1)
        loss = self.loss_fn(pred, gnd)
        if delay_frames is not None:
            pred = pred[:, :, delay_frames:]
            gnd = gnd[:, delay_frames:]
        pred_label = torch.argmax(pred, dim=1)
        accuracy = (pred_label == gnd).float().mean()
        pred_label = pred_label.view(-1).detach().cpu().tolist()
        gnd_label = gnd.contiguous().view(-1).detach().cpu().tolist()
        loss = {"total": loss, "accuracy": accuracy}
        return loss, (gnd_label, pred_label)
    
    