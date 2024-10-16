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

def get_embeddings_name(config, name):
    devtest_embeddings = (
        None
        if config["pretrained"]["e2e"]
        else os.path.join(
            config["pretrained"]["extracted_embeddings_dir"],
            config["pretrained"]["model"],
            f"{name}.hdf5",
        )
    )

    return devtest_embeddings
def single_run(
    config,
    log_dir,
    gpus,
    checkpoint_resume=None,
    test_state_dict=None,
    fast_dev_run=False,
    evaluation=False,
    callbacks=None,
):
    """
    Running sound event detection baselin

    Args:
        config (dict): the dictionary of configuration params
        log_dir (str): path to log directory
        gpus (int): number of gpus to use
        checkpoint_resume (str, optional): path to checkpoint to resume from. Defaults to "".
        test_state_dict (dict, optional): if not None, no training is involved. This dictionary is the state_dict
            to be loaded to test the model.
        fast_dev_run (bool, optional): whether to use a run with only one batch at train and validation, useful
            for development purposes.
    """
    config.update({"log_dir": log_dir})

    # handle seed
    seed = config["training"]["seed"]
    if seed:
        pl.seed_everything(seed, workers=True)

    ##### data prep test ##########
    encoder = get_encoder(config)
        if not evaluation:
        devtest_df = pd.read_csv(config["data"]["test_tsv"], sep="\t")

# Training dataset
train_dataset = StronglyAnnotatedSet(
    audio_folder=config_data['audio_train_folder'],
    annotations_df=pd.read_csv(config_data['train_full_csv']),
    encoder=encoder,
    pad_to=config["data"]["audio_max_len"],
    fs=config["data"]["fs"],
    return_filename=False,
    random_channel=False,
    multisrc=False,
    feats_pipeline=None,  # Replace with your feature extraction pipeline if you have one
    test=False,
)

# Validation dataset
val_dataset = StronglyAnnotatedSet(
    audio_folder=config_data['audio_val_folder'],
    annotations_df=pd.read_csv(config_data['val_full_csv']),
    encoder=encoder,
    pad_to=config["data"]["audio_max_len"],
    fs=config["data"]["fs"],
    return_filename=False,
    random_channel=False,
    multisrc=False,
    feats_pipeline=None,
    test=True,
)

# Test dataset
test_dataset = StronglyAnnotatedSet(
    audio_folder=config_data['test_folder'],
    annotations_df=pd.read_csv(config_data['test_full_csv']),
    encoder=encoder,
    pad_to=config["data"]["audio_max_len"],
    fs=config["data"]["fs"],
    return_filename=True,  # Set to True if you want to get the filename when accessing samples
    random_channel=False,
    multisrc=False,
    feats_pipeline=None,
    test=True,
)
