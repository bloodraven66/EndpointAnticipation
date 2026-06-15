from .registry import register_feature
import torch
import logging

import os
os.environ['NEMO_LOG_LEVEL'] = 'ERROR'
os.environ['HYDRA_FULL_ERROR'] = '0'

@register_feature("fastconformer_streaming")
def fastconformer_streaming(cfg):
    logging.getLogger('nemo_logger').setLevel(logging.ERROR)
    nemo_logger = logging.getLogger('nemo_logger')
    nemo_logger.setLevel(logging.ERROR)
    
    # Also suppress PyTorch Lightning logs that NeMo uses
    logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)
    logging.getLogger('lightning').setLevel(logging.ERROR)
    

    import nemo.collections.asr as nemo_asr

    from nemo.utils import logging as nemo_logging
    nemo_logging.setLevel(logging.ERROR)

    asr_model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(model_name=cfg.data.audio_params.model_repo)
    asr_model.encoder.set_default_att_context_size(cfg.data.audio_params.att_context_size)
    asr_model.change_decoding_strategy(decoder_type='ctc')
    encoder_module = asr_model.encoder
    preprocessor = asr_model.preprocessor
    del asr_model.decoder, asr_model.joint
    del asr_model.ctc_decoder, asr_model.ctc_loss, asr_model.ctc_wer
    del asr_model.loss, asr_model.spec_augmentation, asr_model.wer
    del asr_model

    

    class encode():
        def __init__(
            self,
            model,
            preprocessor,
            device
        ):
            self.model = model
            self.preprocessor = preprocessor
            self.device = cfg.run_params.device
            if not torch.cuda.is_available():
                if self.device != 'cpu':
                    logger.warning("CUDA not available, switching to CPU")
                self.device = 'cpu'
            self.model.to(self.device)
            self.preprocessor.to(self.device)
            self.model.eval()
        
        def set_device(self, device):
            self.device = device
            self.model.to(self.device)
            self.preprocessor.to(self.device)

        def __call__(self, wav, chunk_length=None):
            if isinstance(wav, list):
                wav = torch.cat(wav, 0)
            if len(wav.shape) == 1:
                wav = wav.unsqueeze(0)
            if chunk_length is not None:
                encoded = []
                num_chunks = wav.shape[-1] // chunk_length + 1
                for i in range(num_chunks):
                    wav_ = wav[:, i*chunk_length:(i+1)*chunk_length]
                    wav_ = wav_.to(self.device)
                    length = torch.tensor([wav_.shape[1]]).to(self.device)
                    with torch.no_grad():
                        processed_signal, processed_signal_length = self.preprocessor.get_features(input_signal=wav_, length=length)
                        encoded_, _ = self.model(audio_signal=processed_signal, length=processed_signal_length)
                        encoded.append(encoded_.cpu())
                encoded = torch.cat(encoded, dim=-1)
                encoded = encoded.to(self.device)
            else:
                wav = wav.to(self.device)
                length = torch.tensor([wav.shape[1]]).to(self.device)
                # print(wav.device, self.model.device)
                with torch.no_grad():
                    processed_signal, processed_signal_length = self.preprocessor.get_features(input_signal=wav, length=length)
                    encoded, _ = self.model(audio_signal=processed_signal, length=processed_signal_length)
            return encoded.squeeze(0)
    nemo_encode = encode(encoder_module, preprocessor, cfg.run_params.device)
    return nemo_encode