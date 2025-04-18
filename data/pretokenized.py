import math
import numpy as np
import os
import torch
import lightning as L
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import Any, List, Tuple, Union


class PretokenizedDataModule:
    """
    PyTorch Lightning style DataModule for a pretokenized dataset
    """

    def __init__(
        self,
        fabric: L.Fabric,
        config,
    ):
        self.fabric = fabric
        self.seq_length = config.model.block_size
        self.batch_size = config.data.device_batch_size
        self.num_workers = config.data.num_workers

        root_path = config.data.pretokenized_data_path
        train_data = config.data.pretokenized_train_data
        val_data = config.data.pretokenized_val_data
        self.train_dataset = self._load_data_files(root_path, train_data)
        self.val_dataset = self._load_data_files(root_path, val_data)

        tokenizer_name_or_path = config.data.tokenizer_name_or_path
        if tokenizer_name_or_path:
            self.fabric.print(f"Loading tokenizer {tokenizer_name_or_path}")
            with fabric.rank_zero_first():
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        else:
            self.fabric.print(
                f"No tokenizer specified in config, using dummy tokenizer with EOS={config.data.tokenizer_eos_id}"
            )
            self.tokenizer = DummyTokenizer(config.data.tokenizer_eos_id)

        self.fabric.print(f"Total size of training dataset: {len(self.train_dataset)}")
        self.fabric.print(f"Total size of validation dataset: {len(self.val_dataset)}")

    def _load_data_files(
        self, root_path: str, data_list: List[Union[str, Tuple[str, float]]]
    ):
        datasets = []
        data_paths = []
        weights = []

        for data_spec in data_list:
            # Default weight is 1.0 if not specified
            if isinstance(data_spec, str):
                data_file, data_weight = data_spec, 1.0
            elif len(data_spec) == 2:
                data_file, data_weight = data_spec
            else:
                raise ValueError(
                    f"Invalid data file specification {data_spec}. Expected a type of str or (str, float)"
                )

            # Load the dataset
            data_path = os.path.join(root_path, data_file)
            dataset = LMDataset(data_path, self.seq_length)
            datasets.append(dataset)
            data_paths.append(data_path)
            weights.append(data_weight)

        concat_dataset = WeightedConcatDataset(datasets, weights)

        # Print dataset info
        self.fabric.print(f"Loading {len(datasets)} data files")
        for i in range(len(datasets)):
            path = data_paths[i]
            weight = weights[i]
            actual_length = len(datasets[i])
            effective_length = concat_dataset.effective_lengths[i]
            self.fabric.print(
                f"  {path}: {actual_length} samples, {effective_length} effective samples (weight={weight})"
            )

        return concat_dataset

    def train_dataloader(self):
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )
        return self.fabric.setup_dataloaders(dataloader)

    def val_dataloader(self):
        dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
        )
        return self.fabric.setup_dataloaders(dataloader)

    def update_config(self, config):
        pass

    def get_tokenizer(self):
        return self.tokenizer


class DummyTokenizer:
    """
    Dummy tokenizer that only stores the EOS token id.
    Can be used for training on a pretokenized dataset, but not for inference.
    """

    def __init__(self, eos_token_id: int):
        self.eos_token_id = eos_token_id

    def __call__(self, batch):
        raise NotImplementedError

    def encode(self, *args):
        raise NotImplementedError

    def decode(self, *args):
        raise NotImplementedError


class WeightedConcatDataset(Dataset):
    """Weighted concatenation of datasets."""

    def __init__(self, datasets: List[Dataset], weights: List[float]) -> None:
        """
        Initialize the dataset by concatenating the given datasets according to the given weights.
        For a given dataset, the number of resulting samples is computed as `ceil(len(dataset) * weight)`.

        If the resulting number of samples is greater than the length of the dataset,
        the dataset is duplicated multiple times to reach the desired number of samples.

        If the resulting number of samples is less than the length of the dataset,
        the dataset is truncated to the desired number of samples, without shuffling.

        Args:
            datasets: List of datasets.
            weights: List of weights.

        """

        assert len(datasets) == len(
            weights
        ), "`datasets` and `weights` must have the same length."
        self.datasets = datasets
        self.weights = weights

        self.actual_lengths = np.array([len(dataset) for dataset in self.datasets])
        self.effective_lengths = np.array(
            [math.ceil(l * w) for l, w in zip(self.actual_lengths, self.weights)]
        )
        self.cumulative_lengths = np.cumsum(self.effective_lengths)

        if len(self.cumulative_lengths) == 0:
            self.total_length = 0
        else:
            self.total_length = self.cumulative_lengths[-1]

    def __len__(self) -> int:
        return self.total_length

    def __getitem__(self, idx: int) -> Any:
        # Handle out-of-bounds indices
        if idx >= self.total_length:
            raise IndexError(
                f"Absolute index {idx} out of range for dataset with size of {self.total_length}."
            )
        # Handle negative indices
        if idx < 0:
            if -idx > self.total_length:
                raise IndexError(
                    f"Absolute index {idx} out of range for dataset with size of {self.total_length}."
                )
            idx = self.total_length + idx

        # Find the dataset index and sample index
        dataset_idx = np.searchsorted(self.cumulative_lengths, idx, side="right")
        if dataset_idx == 0:
            # No previous dataset, so sample index is just the index
            sample_idx = idx
        else:
            # Subtract the cumulative length of the previous dataset
            sample_idx = idx - self.cumulative_lengths[dataset_idx - 1]

        # Map the sample index to the actual dataset length
        sample_idx = sample_idx % self.actual_lengths[dataset_idx]

        sample = self.datasets[dataset_idx][sample_idx]
        return sample


class LMDataset(Dataset):
    """
    Dataset that loads a numpy file from disk. The file is assumed to contain
    packed sequences of pretokenized text data.

    Each index returned from the dataset is a 1-D tensor of length `seq_len`.
    Any additional data at the end of file that does not evenly divide `seq_len` is truncated.

    The returned tensors are memory mapped and read only.

    Args:
        file_path: Path to the numpy file.
        seq_len: Length of each sequence.
    """

    def __init__(self, file_path: str, seq_len: int):
        assert os.path.isfile(file_path), f"File {file_path} does not exist"
        self.file_path = file_path
        self.seq_len = seq_len

        # Load the file
        array = np.load(file_path, mmap_mode="r", allow_pickle=False)
        assert (
            array.ndim == 1
        ), f"Expected 1-D numpy array, but got {array.ndim}-D array of shape {array.shape} in {file_path}"

        # Flatten and truncate
        self.num_sequences = array.size // self.seq_len
        array = array[: self.num_sequences * self.seq_len]

        self.array = np.reshape(array, (self.num_sequences, self.seq_len), copy=False)

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Must convert type to be compatible with PyTorch embedding layer
        return torch.from_numpy(self.array[idx].astype(np.long))
