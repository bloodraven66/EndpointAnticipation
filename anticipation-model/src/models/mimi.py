from .registry import register_feature
import torch
from src.utils.logger import logger

@register_feature("mimi")
def mimi(cfg):
    """
    Get Mimi model (https://huggingface.co/kyutai/mimi)
    Args:
        cfg (edict): EasyDict containing the configurations
    Returns:
        mimi_encode: Mimi encoder function
    """
    from transformers import MimiModel, AutoFeatureExtractor
    model = MimiModel.from_pretrained(cfg.data.audio_params.model_repo)
    feature_extractor = AutoFeatureExtractor.from_pretrained(cfg.data.audio_params.model_repo)
    
    class encode():
        def __init__(
            self, 
            model, 
            processor, 
            sr, 
            device, 
            upsample=False,
            num_quantizers=2,
            no_quantisation=False,
            downsample=False,
            use_transformer=False,
        ):  
            model.config.num_quantizers = num_quantizers
            self.model = model
            self.feature_extractor = processor
            self.model.eval()
            self.device = cfg.run_params.device
            if not torch.cuda.is_available():
                if self.device != 'cpu':
                    logger.warning("CUDA not available, switching to CPU")
                self.device = 'cpu'
            self.model.to(self.device)
            self.model.quantizer.to(self.device)
            self.upsample = upsample
            self.sr = sr
            self.num_quantizers = num_quantizers
            self.no_quantisation = no_quantisation
            self.downsample = downsample
            self.use_transformer = use_transformer
            assert self.sr == self.feature_extractor.sampling_rate
            
        def __call__(self, wav, chunk_length=None):
            # print(wav.shape)
            # wav_ = wav.clone()
            # if torch.is_tensor(wav):
                # wav = wav.cpu().numpy().tolist()
            # inputs = self.feature_extractor(raw_audio=wav, sampling_rate=self.sr, return_tensors="pt")
            # print((inputs["input_values"].squeeze() - wav_.cpu().squeeze()).sum())
            # inputs["input_values"] = inputs["input_values"].to(self.device)
            inputs = wav.unsqueeze(1).to(self.device)
            if self.no_quantisation:
                assert self.upsample == False
                # padding_mask = torch.ones_like(inputs["input_values"]).bool()
                encoder_past_key_values = None
                return_dict = False
                embeddings = model.encoder(inputs)
                if self.use_transformer:
                    embeddings = self.model.encoder_transformer(
                        embeddings.transpose(1, 2), 
                        past_key_values=encoder_past_key_values, 
                        return_dict=return_dict
                    )
                    embeddings = embeddings[0].transpose(1, 2)
                    if self.downsample:
                        embeddings = self.model.downsample(embeddings)
                else:
                    assert self.downsample == False

            else:
                codes = self.model.encode(inputs, num_quantizers=self.num_quantizers).audio_codes
                embeddings = self.model.quantizer.decode(codes)
                if self.upsample:
                    embeddings = self.model.upsample(embeddings)
            return embeddings.squeeze()

    no_quantisation = False
    if hasattr(cfg.data.audio_params, "no_quantisation"):
        if cfg.data.audio_params.no_quantisation:
            no_quantisation = True
    downsample = False
    if hasattr(cfg.data.audio_params, "downsample"):
        if cfg.data.audio_params.downsample:
            downsample = True
    use_transformer = False
    if hasattr(cfg.data.audio_params, "use_transformer"):
        if cfg.data.audio_params.use_transformer:
            use_transformer = True
            
    mimi_encode = encode(
        model, 
        feature_extractor, 
        sr=cfg.data.audio_params.target_sr, 
        device=cfg.run_params.device,
        upsample=cfg.data.audio_params.upsample,
        num_quantizers=cfg.data.audio_params.num_quantisers,
        no_quantisation=no_quantisation,
        downsample=downsample,
        use_transformer=use_transformer,
    )
    return mimi_encode