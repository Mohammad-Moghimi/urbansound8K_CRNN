import os
import random
from pathlib import Path
import glob

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset


def to_mono(mixture, random_ch=False):
    if mixture.ndim > 1:  # multi-channel
        if not random_ch:
            mixture = torch.mean(mixture, 0)
        else:  # randomly select one channel
            indx = np.random.randint(0, mixture.shape[0])
            mixture = mixture[indx]
    return mixture

def pad_audio(audio, target_len, fs, test=False):
    target_len = int(target_len)  # Ensure target_len is an integer
    if audio.shape[-1] < target_len:
        pad_amount = int(target_len - audio.shape[-1])
        audio = torch.nn.functional.pad(
            audio, (0, pad_amount), mode="constant"
        )
        padded_indx = [target_len / len(audio)]
        onset_s = 0.000
    elif len(audio) > target_len:
        if test:
            clip_onset = 0
        else:
            clip_onset = random.randint(0, len(audio) - target_len)
        audio = audio[clip_onset: clip_onset + target_len]
        onset_s = round(clip_onset / fs, 3)
        padded_indx = [target_len / len(audio)]
    else:
        onset_s = 0.000
        padded_indx = [1.0]
    offset_s = round(onset_s + (target_len / fs), 3)
    return audio, onset_s, offset_s, padded_indx

def process_labels(df, onset, offset, pad_to_sec):
    df["onset"] = df["onset"] - onset
    df["offset"] = df["offset"] - onset
    df["onset"] = df["onset"].clip(lower=0, upper=pad_to_sec)
    df["offset"] = df["offset"].clip(lower=0, upper=pad_to_sec)
    df = df[df["onset"] < df["offset"]]
    return df

def read_audio(file, multisrc, random_channel, target_len, test=False):
    mixture, fs = torchaudio.load(file)
    if not multisrc:
        mixture = to_mono(mixture, random_channel)
    mixture, onset_s, offset_s, padded_indx = pad_audio(mixture, target_len, fs, test=test)
    mixture = mixture.float()
    return mixture, onset_s, offset_s, padded_indx

class StronglyAnnotatedSet(Dataset):
    def __init__(
        self,
        audio_folder,
        annotations_df,
        encoder,
        pad_to=4,  # UrbanSound8K clips are 4 seconds
        fs=16000,
        return_filename=False,
        random_channel=False,
        multisrc=False,
        feats_pipeline=None,
        embeddings_hdf5_file=None,
        embedding_type=None,
        mask_events_other_than=None,
        test=False,
    ):
        self.encoder = encoder
        self.fs = fs

        # Ensure that self.pad_to_sec is defined before it's used
        self.pad_to_sec = pad_to  # pad_to in seconds
        self.pad_to = int(pad_to * fs)  # pad_to in samples

        self.return_filename = return_filename
        self.random_channel = random_channel
        self.multisrc = multisrc
        self.feats_pipeline = feats_pipeline
        self.embeddings_hdf5_file = embeddings_hdf5_file
        self.embedding_type = embedding_type
        self.test = test

        if mask_events_other_than is not None:
            self.mask_events_other_than = torch.ones(len(encoder.labels))
            for indx, cls in enumerate(encoder.labels):
                if cls not in mask_events_other_than:
                    self.mask_events_other_than[indx] = 0
        else:
            self.mask_events_other_than = torch.ones(len(encoder.labels))
        self.mask_events_other_than = self.mask_events_other_than.bool()

        annotations_df = annotations_df.dropna()

        # Process annotations
        self.examples = {}
        for _, row in annotations_df.iterrows():
            filename = row["slice_file_name"]
            fold = f"fold{row['fold']}"  # Include fold number
            if filename not in self.examples:
                self.examples[filename] = {
                    "mixture": os.path.join(audio_folder, fold, filename),
                    "events": [],
                }
            self.examples[filename]["events"].append(
                {
                    "event_label": row["class"],
                    "onset": 0.0,
                    "offset": self.pad_to_sec,
                }
            )

        self.examples_list = list(self.examples.keys())

        # Handle embeddings if provided
        if self.embeddings_hdf5_file is not None:
            assert (
                self.embedding_type is not None
            ), "If you use embeddings you need to specify the type ('global' or 'frame')"
            self.ex2emb_idx = {}
            with h5py.File(self.embeddings_hdf5_file, "r") as f:
                for i, fname in enumerate(f["filenames"]):
                    self.ex2emb_idx[fname.decode("UTF-8")] = i
            self._opened_hdf5 = None

    def __len__(self):
        return len(self.examples_list)

    @property
    def hdf5_file(self):
        if self._opened_hdf5 is None:
            self._opened_hdf5 = h5py.File(self.embeddings_hdf5_file, "r")
        return self._opened_hdf5

    def __getitem__(self, index):
        c_ex = self.examples[self.examples_list[index]]
        mixture, onset_s, offset_s, padded_indx = read_audio(
            c_ex["mixture"],
            self.multisrc,
            self.random_channel,
            self.pad_to,
            self.test,
        )

        # Labels
        labels = c_ex["events"]
        labels_df = pd.DataFrame(labels)

        # Adjust labels if necessary
        labels_df = process_labels(labels_df, onset_s, offset_s, self.pad_to_sec)

        # Encode labels
        if not len(labels_df):
            max_len_targets = self.encoder.n_frames
            strong = torch.zeros(max_len_targets, len(self.encoder.labels)).float()
        else:
            strong = self.encoder.encode_strong_df(labels_df)
            strong = torch.from_numpy(strong).float()

        # Prepare output arguments
        out_args = [mixture, strong.transpose(0, 1), padded_indx]

        # Feature extraction
        if self.feats_pipeline is not None:
            feats = self.feats_pipeline(mixture)
            out_args.append(feats)

        if self.return_filename:
            out_args.append(c_ex["mixture"])

        # Embeddings
        if self.embeddings_hdf5_file is not None:
            name = Path(c_ex["mixture"]).stem
            index = self.ex2emb_idx.get(name)
            if index is not None:
                if self.embedding_type == "global":
                    embeddings = torch.from_numpy(
                        self.hdf5_file["global_embeddings"][index]
                    ).float()
                elif self.embedding_type == "frame":
                    embeddings = torch.from_numpy(
                        np.stack(self.hdf5_file["frame_embeddings"][index])
                    ).float()
                else:
                    raise NotImplementedError
                out_args.append(embeddings)
            else:
                raise KeyError(f"Embedding for {name} not found.")

        if self.mask_events_other_than is not None:
            out_args.append(self.mask_events_other_than)

        return out_args
