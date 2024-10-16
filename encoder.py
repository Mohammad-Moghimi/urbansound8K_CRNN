import numpy as np
import pandas as pd
from collections import OrderedDict
from dcase_util.data import DecisionEncoder  # Ensure you have dcase_util installed

class ManyHotEncoder:
    """
    Encode labels into numpy arrays where 1 corresponds to the presence of the class and 0 absence.
    Supports strong labels (with onset and offset times) for multi-label problems.

    Args:
        labels: list, the classes to encode
        audio_len: float, length of the audio in seconds
        frame_len: int, frame length in samples
        frame_hop: int, hop length in samples
        net_pooling: int, pooling factor of the network (default: 1)
        fs: int, sampling frequency (default: 16000)
    """

    def __init__(
        self, labels, audio_len, frame_len, frame_hop, net_pooling=1, fs=16000
    ):
        if isinstance(labels, (np.ndarray, np.array)):
            labels = labels.tolist()
        elif isinstance(labels, (dict, OrderedDict)):
            labels = list(labels.keys())
        self.labels = labels
        self.audio_len = audio_len
        self.frame_len = frame_len
        self.frame_hop = frame_hop
        self.fs = fs
        self.net_pooling = net_pooling
        n_frames = self.audio_len * self.fs
        self.n_frames = int(int((n_frames / self.frame_hop)) / self.net_pooling)

    def _time_to_frame(self, time):
        samples = time * self.fs
        frame = samples / self.frame_hop
        return int(np.clip(frame / self.net_pooling, a_min=0, a_max=self.n_frames))

    def _frame_to_time(self, frame):
        frame = frame * self.net_pooling * self.frame_hop / self.fs
        return np.clip(frame, a_min=0, a_max=self.audio_len)

    def encode_strong_df(self, label_df):
        """
        Encode strong labels into a numpy array.

        Args:
            label_df: pandas DataFrame containing 'onset', 'offset', and 'event_label' columns

        Returns:
            numpy.array
            Encoded labels of shape (n_frames, n_classes)
        """
        y = np.zeros((self.n_frames, len(self.labels)))
        if not label_df.empty and {"onset", "offset", "event_label"}.issubset(label_df.columns):
            for _, row in label_df.iterrows():
                if not pd.isna(row["event_label"]):
                    label = row["event_label"]
                    if label in self.labels:
                        i = self.labels.index(label)
                        onset = self._time_to_frame(row["onset"])
                        offset = self._time_to_frame(row["offset"])
                        y[onset:offset, i] = 1  # Mark the presence of the event
        return y

    def decode_strong(self, labels):
        """
        Decode the encoded strong labels.

        Args:
            labels: numpy.array, the encoded labels to be decoded

        Returns:
            list
            Decoded labels, list of [label, onset, offset]
        """
        result_labels = []
        for i, label_column in enumerate(labels.T):
            change_indices = DecisionEncoder().find_contiguous_regions(label_column)
            # Append [label, onset, offset] to the result list
            for row in change_indices:
                onset_time = self._frame_to_time(row[0])
                offset_time = self._frame_to_time(row[1])
                result_labels.append([self.labels[i], onset_time, offset_time])
        return result_labels

    def state_dict(self):
        return {
            "labels": self.labels,
            "audio_len": self.audio_len,
            "frame_len": self.frame_len,
            "frame_hop": self.frame_hop,
            "net_pooling": self.net_pooling,
            "fs": self.fs,
        }

    @classmethod
    def load_state_dict(cls, state_dict):
        return cls(
            labels=state_dict["labels"],
            audio_len=state_dict["audio_len"],
            frame_len=state_dict["frame_len"],
            frame_hop=state_dict["frame_hop"],
            net_pooling=state_dict["net_pooling"],
            fs=state_dict["fs"],
        )
