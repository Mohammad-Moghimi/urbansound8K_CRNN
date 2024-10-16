# main.py

from encoder import ManyHotEncoder
from collections import OrderedDict
import pandas as pd
import numpy as np




def get_encoder(config):
    """
    Initialize the ManyHotEncoder for the UrbanSound8K dataset.

    Args:
        config: dict, configuration parameters including audio length, frame length, etc.

    Returns:
        ManyHotEncoder instance
    """
    encoder = ManyHotEncoder(
        labels=list(classes_labels_urbansound8k.keys()),
        audio_len=config["data"]["audio_max_len"],
        frame_len=config["feats"]["frame_len"],
        frame_hop=config["feats"]["frame_hop"],
        net_pooling=config["data"].get("net_pooling", 1),
        fs=config["data"]["fs"],
    )
    return encoder
