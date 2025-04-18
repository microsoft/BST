import os
import random
import torch
import lightning as L
from collections.abc import Sequence
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast
from trl.trainer.utils import ConstantLengthDataset


def apply_fim(tokenized_input, fim_token_id, eos_token_id, goal_range, add_eos=False):
    # take token sequence, randomly prepend part of the tail of the
    # sequence (goal) to the remainder (start) andput a <|fim|> special
    # token in the middle between goal and start

    # note that ConstantLengthDataset adds the <eos>, not the tokenizer.
    # we add an option to add an eos to the end of the sequence for
    # easier inference

    # get goal length
    if isinstance(goal_range, Sequence):
        goal_length = random.randint(
            goal_range[0], min(goal_range[1], len(tokenized_input) - 2)
        )
    else:
        goal_length = goal_range

    # [a, b, c, d, e, f, g] --> [e, f, g, <|fim|>, a, b, c, d]
    ret = (
        tokenized_input[-goal_length:] + [fim_token_id] + tokenized_input[:-goal_length]
    )
    if add_eos:  # for easier inference. Do not use for training.
        ret = ret + [eos_token_id]

    return ret


class ConstantLengthDatasetFIM(ConstantLengthDataset):
    """
    Variant of ConstantLengthDataset that creates Fill-In-Middle (FIM) sequences
    """

    def __init__(
        self,
        tokenizer,
        dataset,
        dataset_text_field=None,
        formatting_func=None,
        infinite=False,
        seq_length=1024,
        num_of_sequences=1024,
        chars_per_token=3.6,
        eos_token_id=0,
        shuffle=True,
        append_concat_token=True,
        add_special_tokens=True,
        goal_range=(25, 50),
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset=dataset,
            dataset_text_field=dataset_text_field,
            formatting_func=formatting_func,
            infinite=infinite,
            seq_length=seq_length,
            num_of_sequences=num_of_sequences,
            chars_per_token=chars_per_token,
            eos_token_id=eos_token_id,
            shuffle=shuffle,
            append_concat_token=append_concat_token,
            add_special_tokens=add_special_tokens,
        )
        self.goal_range = goal_range

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    buffer.append(self.formatting_func(next(iterator)))
                    buffer_len += len(buffer[-1])
                    buffer_len += 1  # for the extra <|fim|> token
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                    else:
                        more_examples = False
                        break
            if self.shuffle:
                random.shuffle(buffer)
            if self.pretokenized:
                tokenized_inputs = buffer
            else:
                tokenized_inputs = self.tokenizer(
                    buffer, add_special_tokens=self.add_special_tokens, truncation=False
                )["input_ids"]

            # do FIM processing
            fim_tokenized_inputs = []
            for tokenized_input in tokenized_inputs:
                fim_tokenized_inputs.append(
                    apply_fim(
                        tokenized_input,
                        fim_token_id=self.tokenizer.convert_tokens_to_ids("<|fim|>"),
                        eos_token_id=self.tokenizer.eos_token_id,
                        goal_range=self.goal_range,
                        add_eos=False,
                    )
                )

            all_token_ids = []
            for tokenized_input in fim_tokenized_inputs:
                if self.append_concat_token:
                    tokenized_input = tokenized_input + [self.concat_token_id]
                all_token_ids.extend(tokenized_input)
            examples = []
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i : i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    examples.append(input_ids)
            if self.shuffle:
                # Shuffle again, otherwise split examples occur in consecutive tensors.
                random.shuffle(examples)
            for example in examples:
                self.current_size += 1
                yield {
                    "input_ids": torch.LongTensor(example),
                    "labels": torch.LongTensor(example),
                }


class TinyStoriesDataModule:
    """
    PyTorch Lightning style DataModule for TinyStories dataset
    """

    def __init__(self, fabric: L.Fabric, config, data_path: str = "data/tinystories"):
        self.fabric = fabric
        self.data_path = data_path
        self.batch_size = config.data.device_batch_size
        self.num_workers = config.data.num_workers

        assert self.num_workers >= 0, "num_workers must be >= 0"
        assert (
            self.num_workers <= 1
        ), "num_workers must be <= 1 because ConstantLengthDataset does not support parallel data loading"

        # Load tokenizer
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=os.path.join(data_path, "tokenizer.json"),
            eos_token="<|eos|>",
            pad_token="<|pad|>",
            additional_special_tokens=(
                ["<|fim|>"]
                if (not config.use_bst and config.model.gpt_mode == "fim")
                else None
            ),
        )

        # Load data
        with self.fabric.rank_zero_first(local=True):
            train_dataset, val_dataset = load_dataset(
                "cyrilzhang/TinyStories2-ascii",
                split=["train", "validation"],
                download_mode="reuse_cache_if_exists",
            )

        # Partition data between devices
        # Because we later create an iterable dataset, we first need to manually shard the data here
        train_dataset = train_dataset.shard(
            num_shards=self.fabric.world_size, index=self.fabric.global_rank
        )
        val_dataset = val_dataset.shard(
            num_shards=self.fabric.world_size, index=self.fabric.global_rank
        )

        # Shuffle data
        train_dataset.shuffle(seed=0)
        val_dataset.shuffle(seed=0)

        # Create sequence packing dataset
        PackedDatasetClass = (
            ConstantLengthDatasetFIM
            if (not config.use_bst and config.model.gpt_mode == "fim")
            else ConstantLengthDataset
        )

        train_dataset.set_format(type="torch", columns=["text"])
        train_dataset_packed = PackedDatasetClass(
            tokenizer=self.tokenizer,
            dataset=train_dataset,
            dataset_text_field="text",
            infinite=False,
            seq_length=config.model.block_size,
        )
        self.train_dataset_packed = train_dataset_packed

        val_dataset.set_format(type="torch", columns=["text"])
        val_dataset_packed = PackedDatasetClass(
            tokenizer=self.tokenizer,
            dataset=val_dataset,
            dataset_text_field="text",
            infinite=False,
            seq_length=config.model.block_size,
        )
        self.val_dataset_packed = val_dataset_packed
        self.val_dataset_unpacked = val_dataset

    def train_dataloader(self):
        dataloader = torch.utils.data.DataLoader(
            self.train_dataset_packed,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        return self.fabric.setup_dataloaders(dataloader)

    def val_dataloader(self):
        dataloader = torch.utils.data.DataLoader(
            self.val_dataset_packed,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        return self.fabric.setup_dataloaders(dataloader)

    def eval_dataloader(self):
        dataloader = torch.utils.data.DataLoader(
            self.val_dataset_unpacked,
            batch_size=1,
            drop_last=False,
        )
        return self.fabric.setup_dataloaders(dataloader)

    def update_config(self, config):
        pass

    def get_tokenizer(self):
        return self.tokenizer

    def prepare_batch(self, batch):
        return batch["input_ids"]
