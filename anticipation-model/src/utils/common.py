from easydict import EasyDict as edict
from src.training.default_trainer import load_default_trainer
from src.training.forecasting_trainer import load_forecasting_trainer
import os
import torch
import yaml
import math
import numpy as np
import torchaudio
from src.utils.logger import logger
from huggingface_hub import snapshot_download
from torch.nn import functional as F
from src.models.registry import FEATURE_EXTRACTORS

def load_config(yamlFiles):
    """
    Load config from yaml file(s)
    Args:
        yamlFiles (list): List of paths to yaml config files
    Returns:
        cfg (edict): EasyDict containing the merged configurations
    """
    assert isinstance(yamlFiles, list) and len(yamlFiles) == 1, "Please provide a list with one config file path"
    cfg = {}
    for yamlFile in yamlFiles:
        assert os.path.exists(yamlFile), f"Config file not found: {yamlFile}"
        with open(yamlFile) as f:
            cfg = yaml.load(f, Loader=yaml.SafeLoader)

    if 'data_config' in cfg:
        data_yaml = cfg['data_config']
        assert os.path.exists(data_yaml), f"Data config file not found: {data_yaml}"
        logger.info(f"Loading data config from: {cfg['data_config']}")
        with open(data_yaml) as f:
            data_cfg = yaml.load(f, Loader=yaml.SafeLoader)
        data_cfg_fname = os.path.basename(data_yaml).rstrip(".yaml")
        logger.info(f"Data config {data_cfg_fname} loaded successfully")
        ##merge data_cfg.data into cfg.data
        if 'data' not in cfg:
            cfg['data'] = {}
        for key, value in data_cfg['data'].items():
            cfg['data'][key] = value
    else:
        if not "infer_params" in cfg:
            raise NotImplementedError(f"Only supporting data config based datasets - {cfg}")
    cfg = edict(cfg)
    if 'data_config' in cfg:
        cfg.run_name = data_cfg_fname + "__" + os.path.splitext(os.path.basename(yamlFiles[0]))[0]
        logger.info(f"Run name set to: {cfg.run_name}")
    return cfg    

def load_run(cfg):
    """
    Load model, trainer, and feature extractor from config
    Args:
        cfg (edict): EasyDict containing the configurations
    Returns:
        model: Loaded model
        cfg (edict): Updated config
        trainer: Trainer function
        feat_extractor: Feature extractor
    """
    feat_extractor = None
    if hasattr(cfg, "infer_params"):
        logger.info("Loading inference config")
        infer_folder = os.path.join(cfg.infer_params.root_path, cfg.infer_params.checkpoint_folder, cfg.wandb.run_name)
        infer_cfg_files = [os.path.join(infer_folder, files) for files in os.listdir(infer_folder) if files.endswith(".yaml")]
        infer_cfg = load_config(infer_cfg_files)
        infer_modes = cfg.data.modes
        infer_datasets = cfg.data.datasets
        infer_wandb_mode = cfg.wandb.use_wandb
        infer_device = cfg.infer_params.device
        path = cfg.infer_params.root_path
        cfg = edict({**cfg, **infer_cfg})
        cfg.data.modes = infer_modes
        cfg.data.datasets = infer_datasets
        cfg.infer_folder = infer_folder
        cfg.run_params.infer = True
        cfg.run_params.device = infer_device
        cfg.run_params.batch_size = cfg.infer_params.batch_size
        cfg.wandb.use_wandb = infer_wandb_mode
    if cfg.data.audio_params.audio_feature  not in  ["logmel", "logmel-v2"]:
        feat_extractor = FEATURE_EXTRACTORS[cfg.data.audio_params.audio_feature](cfg)
    logger.info(f"Using {cfg.data.audio_params.audio_feature} as feature extractor")
    model = globals()[cfg.model.name](cfg, feat_extractor)
    trainer = get_trainer(cfg)

    if cfg.run_params.get('compile_model', False) and torch.__version__ >= '2.0.0':
        logger.info("Compiling model with torch.compile")
        model = torch.compile(model)
    return model, cfg, trainer, feat_extractor

def get_trainer(cfg):
    """
    Get trainer function based on model name
    Args:
        cfg (edict): EasyDict containing the configurations
    Returns:
        trainer: Trainer function
    """
    if cfg.model.name in ["base_lstm", "ms_lstm_vap", "ms_lstm", "linear", "reformer", "mamba"]:
        return load_default_trainer
    elif cfg.model.name in ["fc_base_lstm", "fc_base_transformer"]:
        return load_forecasting_trainer
    else:
         raise NotImplementedError(f"Trainer not implemented for {cfg.model.name}")

def get_feat_size(cfg):
    """
    Get feature size based on audio feature type
    Args:
        cfg (edict): EasyDict containing the configurations
    Returns:
        cfg (edict): Updated config with input_size
    """
    if cfg.data.audio_params.audio_feature  in ["logmel", "logmel-v2"]:
        cfg.model_params.input_size = cfg.data.audio_params.n_mels
    else:
        cfg.model_params.input_size = cfg.data.audio_params.feat_size
    return cfg

def ms_lstm(cfg):
    """
    Get MS-LSTM model
    Args:
        cfg (edict): EasyDict containing the configurations
    Returns:
        model: MS-LSTM model
    """
    from src.models.ms_lstm import MS_LSTM_Model
    cfg = get_feat_size(cfg)
    cfg.model_params.output_size = len(cfg.data.special_tokens.keys())
    cfg.model_params.loss_fn = cfg.run_params.loss_fn
    if cfg.model.use_loss_weights:
        cfg.model_params.loss_weights = torch.tensor(cfg.model.loss_weight_factors)
    if hasattr(cfg.data.audio_params, "lookahead_frames"):
        cfg.model_params.lookahead_frames = cfg.data.audio_params.lookahead_frames
    return MS_LSTM_Model(
        use_mel_embed=cfg.model.use_mel_embed,
        **cfg.model_params
    )
def ms_lstm_vap(cfg):
    """
    Get MS-LSTM for Voice Activity Prediction model
    Args:
        cfg (edict): EasyDict containing the configurations
    Returns:
        model: MS-LSTM VAP model
    """
    from src.models.vap_lstm import MS_LSTM_Model
    cfg = get_feat_size(cfg)
    cfg.model_params.output_size = len(cfg.data.special_tokens.keys())
    cfg.model_params.loss_fn = cfg.run_params.loss_fn
    if cfg.model.use_loss_weights:
        cfg.model_params.loss_weights = torch.tensor(cfg.model.loss_weight_factors)
    return MS_LSTM_Model(
        use_mel_embed=cfg.model.use_mel_embed,
        **cfg.model_params
    )

def base_lstm(cfg):
    """
    Get Base LSTM model
    Args:
        cfg (edict): EasyDict containing the configurations
    Returns:
        model: Base LSTM model
    """
    from src.models.base_lstm import LSTM_Model
    cfg = get_feat_size(cfg)
    cfg.model_params.output_size = len(cfg.data.special_tokens.keys())
    cfg.model_params.loss_fn = cfg.run_params.loss_fn
    if hasattr(cfg.model, "use_loss_weights") and cfg.model.use_loss_weights:
        cfg.model_params.loss_weights = torch.tensor(cfg.model.loss_weight_factors)
    return LSTM_Model(
        use_mel_embed=cfg.model.use_mel_embed,
        use_system_embed=cfg.model.use_system_ip_embed, 
        **cfg.model_params
    )

def fc_base_lstm(cfg, feat_extractor=None):
    """
    Get FC Base LSTM model
    Args:
        cfg (edict): EasyDict containing the configurations
    Returns:
        model: FC Base LSTM model
    """
    
    from src.models.fc_base_lstm import FC_LSTM_Model
    cfg = get_feat_size(cfg)
    cfg.model_params.output_size = len(cfg.data.label_params.forecast_intervals_ms)
    if hasattr(cfg.data.label_params, "use_single_prediction_head"):
        logger.info("Using single prediction head")
        cfg.model_params.output_size = 1
        cfg.model_params.single_head = True
    cfg.model_params.loss_fn = cfg.run_params.loss_fn
    if hasattr(cfg.model, "loss_weight_factors"):
        cfg.model_params.loss_weight_factors = cfg.model.loss_weight_factors
    infer_chunk_samples = 10 * cfg.data.audio_params.target_sr
    forecast_intervals = np.array([int((interval / 1000) * cfg.data.audio_params.freq) for interval in cfg.data.label_params.forecast_intervals_ms])
    if hasattr(cfg.model, "asymmetric_loss_weight_factors"):
        cfg.model_params.asymmetric_loss_weight_factors = cfg.model.asymmetric_loss_weight_factors
    if hasattr(cfg.model, "stream_drop") and cfg.model.stream_drop is not None:
        cfg.model_params.stream_drop = cfg.model.stream_drop
    return FC_LSTM_Model(
        feat_extractor=feat_extractor,
        use_mel_embed=hasattr(cfg.model, "use_mel_embed") and cfg.model.use_mel_embed,
        use_system_embed=cfg.model.use_system_ip_embed, 
        num_infer_chunk_samples=infer_chunk_samples,
        forecast_intervals=forecast_intervals,
        **cfg.model_params
    )

def fc_base_transformer(cfg, feat_extractor=None):
    """
    Get FC Base LSTM model
    Args:
        cfg (edict): EasyDict containing the configurations
    Returns:
        model: FC Base LSTM model
    """
    
    from src.models.fc_base_transformer import FC_Transformer_Model
    cfg = get_feat_size(cfg)
    cfg.model_params.output_size = len(cfg.data.label_params.forecast_intervals_ms)
    if hasattr(cfg.data.label_params, "use_single_prediction_head"):
        logger.info("Using single prediction head")
        cfg.model_params.output_size = 1
        cfg.model_params.single_head = True
    cfg.model_params.loss_fn = cfg.run_params.loss_fn
    if hasattr(cfg.model, "loss_weight_factors"):
        cfg.model_params.loss_weight_factors = cfg.model.loss_weight_factors
    infer_chunk_samples = 10 * cfg.data.audio_params.target_sr
    forecast_intervals = np.array([int((interval / 1000) * cfg.data.audio_params.freq) for interval in cfg.data.label_params.forecast_intervals_ms])
    if hasattr(cfg.model, "asymmetric_loss_weight_factors"):
        cfg.model_params.asymmetric_loss_weight_factors = cfg.model.asymmetric_loss_weight_factors
    if hasattr(cfg.model, "stream_drop") and cfg.model.stream_drop is not None:
        cfg.model_params.stream_drop = cfg.model.stream_drop
    return FC_Transformer_Model(
        feat_extractor=feat_extractor,
        use_mel_embed=hasattr(cfg.model, "use_mel_embed") and cfg.model.use_mel_embed,
        num_infer_chunk_samples=infer_chunk_samples,
        forecast_intervals=forecast_intervals,
        **cfg.model_params
    )

def linear(cfg):
    """
    Get Linear model
    Args:
        cfg (edict): EasyDict containing the configurations
    Returns:
        model: Linear model
    """
    from src.models.linear import Linear_Model
    cfg = get_feat_size(cfg)
    cfg.model_params.output_size = len(cfg.data.special_tokens.keys())
    cfg.model_params.loss_fn = cfg.run_params.loss_fn
    if cfg.model.use_loss_weights:
        cfg.model_params.loss_weights = torch.tensor(cfg.model.loss_weight_factors)
        
    return Linear_Model(
        use_mel_embed=cfg.model.use_mel_embed,
        use_system_embed=cfg.model.use_system_ip_embed, 
        **cfg.model_params
    )



def AudioDec(cfg):
    """
    Get AudioDec model 
    Args:
        cfg (edict): EasyDict containing the configurations
    Returns:
        codec_encode: AudioDec encoder function
    """
    from AudioDec.utils.audiodec import AudioDec, assign_model
    sample_rate, encoder_checkpoint, decoder_checkpoint = assign_model(cfg.data.audio_params.model_name)  
    audiodec = AudioDec(tx_device=cfg.run_params.device, rx_device=cfg.run_params.device)
    audiodec.load_transmitter(encoder_checkpoint)
    audiodec.load_receiver(encoder_checkpoint, decoder_checkpoint)
    
    class encode():
        def __init__(
            self, 
            model, 
            nq, 
            sr, 
            device, 
            reduction,
            downsample,
            kernal_size,
            stride
        ):
            self.model = model
            self.device = device
            self.nq = nq
            self.reduction = reduction
            self.sr = sr
            self.kernal_size = kernal_size
            self.stride = stride
            self.downsample = downsample
            assert reduction in ["sum"]
            
        def __call__(self, wav_24kHz):
            wav_24kHz = wav_24kHz[None, None, :].to(self.device)
            z = audiodec.tx_encoder.encode(wav_24kHz)
            idx, (probs, entropy) = audiodec.tx_encoder.quantize(z, return_probs=True)
            if self.nq != 8:
                assert self.nq < 8, "Invalid number of quantisers"
                idx = idx[:self.nq, :]
            code_vectors = audiodec.rx_encoder.lookup(idx).squeeze()
            assert self.reduction in ["sum"], "Only sum reduction is supported - inbuilt"
            code_vectors = code_vectors.squeeze().permute(1, 0)
            if self.downsample:
                code_vectors = self.causal_avg_pool(code_vectors.unsqueeze(0), self.kernal_size, self.stride)
            return code_vectors.squeeze()
    
        def causal_avg_pool(self, input, kernel_size, stride):
            padding = kernel_size - 1
            input_padded = F.pad(input, (padding, 0), mode='constant', value=0)
            return F.avg_pool1d(input_padded, kernel_size, stride=stride)
        
    codec_encode = encode(
        audiodec, 
        nq=cfg.data.audio_params.nq,
        sr=cfg.data.audio_params.target_sr, 
        device=cfg.run_params.device,
        reduction=cfg.data.audio_params.reduction,
        downsample=cfg.data.audio_params.downsample,
        kernal_size=cfg.data.audio_params.kernel_size,
        stride=cfg.data.audio_params.stride
    )
    return codec_encode

    
def Encodec(cfg):
    """
    Get Encodec model (https://huggingface.co/docs/transformers/en/model_doc/encodec)
    Args:
        cfg (edict): EasyDict containing the configurations
    Returns:
        codec_encode: Encodec encoder function
    """
    from transformers import EncodecModel, AutoProcessor
    model = EncodecModel.from_pretrained(cfg.data.audio_params.model_repo)
    processor = AutoProcessor.from_pretrained(cfg.data.audio_params.model_repo)
    num_quantizer2bw = {
        2: 1.5,
        4: 3,
        8: 6,
        16: 12,
        32: 24
    }
    bw = num_quantizer2bw[cfg.data.audio_params.num_quantisers]
    class encode():
        def __init__(
            self, 
            model, 
            processor, 
            bw, 
            sr, 
            device, 
            reduction,
            downsample,
            kernal_size,
            stride
        ):
            self.model = model
            self.model.to(device)
            self.model.eval()
            self.device = device
            self.processor = processor
            self.bw = bw
            self.reduction = reduction
            self.sr = sr
            self.kernal_size = kernal_size
            self.stride = stride
            self.downsample = downsample
            assert reduction in ["sum"]
            
        def __call__(self, wav_24kHz):
            inputs = processor(raw_audio=wav_24kHz, return_tensors="pt", sampling_rate=self.sr)
            inputs["input_values"] = inputs["input_values"].to(self.device)
            encoder_outputs = self.model.encode(inputs["input_values"], inputs["padding_mask"], bandwidth=self.bw)
            code_vectors = self.model.quantizer.decode(encoder_outputs.audio_codes.squeeze(0))
            if self.reduction == "sum":
                code_vectors = code_vectors.sum(dim=0)
            if self.downsample:
                code_vectors = self.causal_avg_pool(code_vectors.unsqueeze(0), self.kernal_size, self.stride)
            return code_vectors.squeeze()
    
        def causal_avg_pool(self, input, kernel_size, stride):
            padding = kernel_size - 1
            input_padded = F.pad(input, (padding, 0), mode='constant', value=0)
            return F.avg_pool1d(input_padded, kernel_size, stride=stride)
        
    codec_encode = encode(
        model, 
        processor, 
        bw, 
        sr=cfg.data.audio_params.target_sr, 
        device=cfg.run_params.device,
        reduction=cfg.data.audio_params.reduction,
        downsample=cfg.data.audio_params.downsample,
        kernal_size=cfg.data.audio_params.kernel_size,
        stride=cfg.data.audio_params.stride
    )
    return codec_encode    


    

    