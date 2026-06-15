
import torch

from src.data.data_processing import (
    process_vad, 
    handle_and_add_turns, 
    endpointing_dataset, 
    endpointing_dataset_full_context
)

import multiprocessing
from src.utils.logger import logger

def load_data(cfg, feat_extractor):
    """
    Load data loaders based on the dataset specified in the config
    Args:
        cfg: Configuration object
        feat_extractor: Feature extractor object
    Returns:
        loaders (dict): Dictionary containing data loaders for each mode
    """

    supported_datasets = ['spokenwoz', 'humdial', 'mlc', 'librispeech', 'fisher', 'switchboard']
    dataset_instances, loaders = {}, {}
    for dataset in cfg.data.datasets:
        logger.info(f"Preparing dataset: {dataset}")
        assert dataset in supported_datasets, f"Dataset {dataset} not supported. Supported datasets: {supported_datasets}"
        preproc_fn_name = f"preprocess_{dataset}"
        dataset_file = f"src.data.{dataset}_dataset"
        from importlib import import_module
        dataset_module = import_module(dataset_file)
        ### We need to convert the raw dataset to a fixed format first
        ### We expect the dataset to have the following format after preprocessing:
        ### Each audio file will have a corresponding .json file with the same name
        ### json format:
        ### {
        ###   "audio_filepath": "path/to/audio/file.wav",
        ###   "segments": [
        ###       {"start": xx.yy, "end": zz.ww, "turn": "...", "text": "..."},
        ###       ...
        ###   ],
        ### Where "segments" contain the turn-level information including start and end times
        ### Turn corresponds to label such as user, system, etc
        ### We will save these preprocessed files for further processing
        getattr(dataset_module, preproc_fn_name)(cfg) #here we preprocess and standardize the dataset
        process_vad(cfg, dataset) #here we use VAD to trim beginning and end silences for each segment
        handle_and_add_turns(cfg, dataset) #here we add all missing turns to the segments
        for mode in cfg.data.modes:
            dataset_class = endpointing_dataset
            if hasattr(cfg, "forecast") and cfg.forecast:
                from . import forecasting
                dataset_class = forecasting.forecasting_dataset
            if hasattr(cfg, "infer_params"):
                # dataset_class = getattr(dataset_module, f"{dataset}_dataset_infer")(cfg, mode, feat_extractor)
                dataset_class = endpointing_dataset_full_context
                if hasattr(cfg, "forecast") and cfg.forecast:
                    dataset_class = forecasting.forecasting_dataset_full_context
            dataset_instance = dataset_class(cfg, mode, dataset, feat_extractor)
            dataset_instances.setdefault(mode, []).append((dataset, dataset_instance))
        ##logger line separator
        logger.info("-" * 50)
    collate_fn = None
    # if hasattr(cfg, "forecast") and cfg.forecast:
    #     from . import forecasting
    #     collate_fn = forecasting.CollateForecasting(cfg, feat_extractor)

    for mode in dataset_instances:
        if mode != "test":
            dataset_instance = torch.utils.data.ConcatDataset([dset[1] for dset in dataset_instances[mode]])
            loaders[mode] = torch.utils.data.DataLoader(
                dataset_instance,
                batch_size=cfg.run_params.batch_size,
                shuffle=True if mode == 'train' else False,
                collate_fn=collate_fn, # We train with fixed length segments, so no need for custom collate_fn
                num_workers=8,
                pin_memory=True
            )
        else:
            test_loaders = []
            for (name, dset) in dataset_instances[mode]:
                test_loader = torch.utils.data.DataLoader(
                    dset,
                    batch_size=cfg.run_params.batch_size,
                    shuffle=False,
                    collate_fn=collate_fn, # We train with fixed length segments, so no need for custom collate_fn
                    num_workers=8,
                    pin_memory=True
                )
                test_loaders.append((name, test_loader))
            loaders[mode] = test_loaders
    return loaders

