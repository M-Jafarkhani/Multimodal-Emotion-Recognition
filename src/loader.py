import pickle
import torch
import numpy as np
from torch.nn import functional as F
from typing import *
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

np.seterr(divide="ignore", invalid="ignore")


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def drop_entry(dataset):
    """Drop entries where there's no text in the data."""
    drop = []
    for ind, k in enumerate(dataset["text"]):
        if k.sum() == 0:
            drop.append(ind)

    for modality in list(dataset.keys()):
        dataset[modality] = np.delete(dataset[modality], drop, 0)
    return dataset


def z_norm(dataset, max_seq_len=50):
    """Normalize data in the dataset."""
    processed = {}
    text = dataset["text"][:, :max_seq_len, :]
    vision = dataset["vision"][:, :max_seq_len, :]
    audio = dataset["audio"][:, :max_seq_len, :]
    for ind in range(dataset["text"].shape[0]):
        vision[ind] = np.nan_to_num(
            (vision[ind] - vision[ind].mean(0, keepdims=True))
            / (np.std(vision[ind], axis=0, keepdims=True))
        )
        audio[ind] = np.nan_to_num(
            (audio[ind] - audio[ind].mean(0, keepdims=True))
            / (np.std(audio[ind], axis=0, keepdims=True))
        )
        text[ind] = np.nan_to_num(
            (text[ind] - text[ind].mean(0, keepdims=True))
            / (np.std(text[ind], axis=0, keepdims=True))
        )

    processed["vision"] = vision
    processed["audio"] = audio
    processed["text"] = text
    processed["labels"] = dataset["labels"]
    return processed


class SentimentDataset(Dataset):
    """Implements Sentiment Dataset as a torch dataset."""

    def __init__(
        self,
        data: Dict,
        flatten_time_series: bool,
        aligned: bool = True,
        task: str = None,
        max_pad=False,
        max_pad_num=50,
        z_norm=False,
    ) -> None:
        """Instantiate SentimentDataset

        Args:
            data (Dict): Data dictionary
            flatten_time_series (bool): Whether to flatten time series or not
            aligned (bool, optional): Whether to align data or not across modalities. Defaults to True.
            task (str, optional): What task to load. Defaults to None.
            max_pad (bool, optional): Whether to pad data to max_pad_num or not. Defaults to False.
            max_pad_num (int, optional): Maximum padding number. Defaults to 50.
            z_norm (bool, optional): Whether to normalize data along the z-axis. Defaults to False.
        """
        self.dataset = data
        self.flatten = flatten_time_series
        self.aligned = aligned
        self.task = task
        self.max_pad = max_pad
        self.max_pad_num = max_pad_num
        self.z_norm = z_norm
        self.dataset["audio"][self.dataset["audio"] == -np.inf] = 0.0

    def __getitem__(self, ind):
        """Get item from dataset."""
        vision = torch.tensor(self.dataset["vision"][ind])
        audio = torch.tensor(self.dataset["audio"][ind])
        text = torch.tensor(self.dataset["text"][ind])

        if self.aligned:
            try:
                start = text.nonzero(as_tuple=False)[0][0]
            except:
                print(text, ind)
                exit()
            vision = vision[start:].float()
            audio = audio[start:].float()
            text = text[start:].float()
        else:
            vision = vision[vision.nonzero()[0][0] :].float()
            audio = audio[audio.nonzero()[0][0] :].float()
            text = text[text.nonzero()[0][0] :].float()

        if self.z_norm:
            vision = torch.nan_to_num(
                (vision - vision.mean(0, keepdims=True))
                / (torch.std(vision, axis=0, keepdims=True))
            )
            audio = torch.nan_to_num(
                (audio - audio.mean(0, keepdims=True))
                / (torch.std(audio, axis=0, keepdims=True))
            )
            text = torch.nan_to_num(
                (text - text.mean(0, keepdims=True))
                / (torch.std(text, axis=0, keepdims=True))
            )

        def _get_class(flag):
            return [[1]] if flag > 0 else [[0]]

        tmp_label = self.dataset["labels"][ind]

        label = (
            torch.tensor(_get_class(tmp_label)).long()
            if self.task == "classification"
            else torch.tensor(tmp_label).float()
        )

        if self.flatten:
            return [
                vision.flatten(),
                audio.flatten(),
                text.flatten(),
                ind,
                label,
            ]
        else:
            if self.max_pad:
                tmp = [vision, audio, text, label]
                for i in range(len(tmp) - 1):
                    tmp[i] = tmp[i][: self.max_pad_num]
                    tmp[i] = F.pad(
                        tmp[i], (0, 0, 0, self.max_pad_num - tmp[i].shape[0])
                    )
            else:
                tmp = [vision, audio, text, ind, label]
            return tmp

    def __len__(self):
        """Get length of dataset."""
        return self.dataset["vision"].shape[0]


def get_dataloader(
    filepath: str,
    batch_size: int = 32,
    max_seq_len=50,
    max_pad=False,
    train_shuffle: bool = True,
    num_workers: int = 2,
    flatten_time_series: bool = False,
    task=None,
    z_norm=False,
) -> DataLoader:
    """Get dataloaders for affect data.

    Args:
        filepath (str): Path to datafile
        batch_size (int, optional): Batch size. Defaults to 32.
        max_seq_len (int, optional): Maximum sequence length. Defaults to 50.
        max_pad (bool, optional): Whether to pad data to max length or not. Defaults to False.
        train_shuffle (bool, optional): Whether to shuffle training data or not. Defaults to True.
        num_workers (int, optional): Number of workers. Defaults to 2.
        flatten_time_series (bool, optional): Whether to flatten time series data or not. Defaults to False.
        task (str, optional): Which task to load in. Defaults to None.
        z_norm (bool, optional): Whether to normalize data along the z dimension or not. Defaults to False.

    Returns:
        DataLoader: tuple of train dataloader, validation dataloader, test dataloader
    """

    with open(filepath, "rb") as f:
        alldata = pickle.load(f)
    processed_dataset = {"train": {}, "test": {}, "valid": {}}
    alldata["train"] = drop_entry(alldata["train"])
    alldata["valid"] = drop_entry(alldata["valid"])
    alldata["test"] = drop_entry(alldata["test"])
    process = eval("_process_2") if max_pad else eval("_process_1")
    for dataset in alldata:
        processed_dataset[dataset] = alldata[dataset]
    train = DataLoader(
        SentimentDataset(
            processed_dataset["train"],
            flatten_time_series,
            task=task,
            max_pad=max_pad,
            max_pad_num=max_seq_len,
            z_norm=z_norm,
        ),
        shuffle=train_shuffle,
        num_workers=num_workers,
        batch_size=batch_size,
        collate_fn=process,
    )
    valid = DataLoader(
        SentimentDataset(
            processed_dataset["valid"],
            flatten_time_series,
            task=task,
            max_pad=max_pad,
            max_pad_num=max_seq_len,
            z_norm=z_norm,
        ),
        shuffle=False,
        num_workers=num_workers,
        batch_size=batch_size,
        collate_fn=process,
    )

    test = DataLoader(
        SentimentDataset(
            processed_dataset["test"],
            flatten_time_series,
            task=task,
            max_pad=max_pad,
            max_pad_num=max_seq_len,
            z_norm=z_norm,
        ),
        shuffle=False,
        num_workers=num_workers,
        batch_size=batch_size,
        collate_fn=process,
    )

    return train, valid, test


def _process_1(inputs: List):
    processed_input = []
    processed_input_lengths = []
    inds = []
    labels = []

    for i in range(len(inputs[0]) - 2):
        feature = []
        for sample in inputs:
            feature.append(sample[i])
        processed_input_lengths.append(
            torch.as_tensor([v.size(0) for v in feature])
        )
        pad_seq = pad_sequence(feature, batch_first=True)
        processed_input.append(pad_seq)

    for sample in inputs:
        inds.append(sample[-2])
        if sample[-1].shape[1] > 1:
            labels.append(
                sample[-1].reshape(sample[-1].shape[1], sample[-1].shape[0])[0]
            )
        else:
            labels.append(sample[-1])

    return (
        processed_input,
        processed_input_lengths,
        torch.tensor(inds).view(len(inputs), 1),
        torch.tensor(labels).view(len(inputs), 1),
    )


def _process_2(inputs: List):
    processed_input = []
    processed_input_lengths = []
    labels = []

    for i in range(len(inputs[0]) - 1):
        feature = []
        for sample in inputs:
            feature.append(sample[i])
        processed_input_lengths.append(
            torch.as_tensor([v.size(0) for v in feature])
        )
        processed_input.append(torch.stack(feature))

    for sample in inputs:
        if sample[-1].shape[1] > 1:
            labels.append(
                sample[-1].reshape(sample[-1].shape[1], sample[-1].shape[0])[0]
            )
        else:
            labels.append(sample[-1])

    return (
        processed_input[0],
        processed_input[1],
        processed_input[2],
        torch.tensor(labels).view(len(inputs), 1),
    )
