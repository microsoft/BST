"""
Tinystories dataset (for toy experimentation)
https://github.com/manantomar/tiny-tt/blob/master/Untitled.ipynb
Downloads and tokenizes the data and saves data shards to disk.
"""

import os
import multiprocessing as mp
import numpy as np
import tiktoken
from functools import partial
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset  # pip install datasets
from tqdm import tqdm  # pip install tqdm


OUTPUT_DIR = "tinystories2-ascii"
SHARD_SIZE = int(1e8)  # 100M tokens per shard, total of 100 shards
USE_GPT2_TOKENIZER = False  # use gpt2 tokenizer or tinystories tokenizer


def tokenize(doc, tokenizer, eos_token_id: int) -> np.ndarray:
    # tokenizes a single document and returns a numpy array of uint16 tokens
    tokens = [eos_token_id]  # the special <|endoftext|> token delimits all documents
    tokens.extend(tokenizer.encode(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (
        tokens_np < 2**16
    ).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16


def write_datafile(tokens_np: np.ndarray, split: str, index: int, output_dir: str):
    filename = os.path.join(output_dir, f"tinystories2_{split}_{index:06d}")
    print(f"Saving file {filename}")
    np.save(filename, tokens_np)


def tokenize_dataset(dataset, tokenize_fn, split: str, output_dir: str):
    print(f"Tokenizing {split} dataset to {output_dir}")

    # create output directory
    os.makedirs(output_dir, exist_ok=True)

    # tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
    nprocs = max(1, os.cpu_count() // 2)
    with mp.Pool(nprocs) as pool:
        shard_index = 0
        # preallocate buffer to hold current shard
        all_tokens_np = np.empty((SHARD_SIZE,), dtype=np.uint16)
        token_count = 0
        progress_bar = None

        for tokens in pool.imap(tokenize_fn, dataset, chunksize=256):
            # is there enough space in the current shard for the new tokens?
            if token_count + len(tokens) < SHARD_SIZE:
                # simply append tokens to current shard
                all_tokens_np[token_count : token_count + len(tokens)] = tokens
                token_count += len(tokens)

                # update progress bar
                if progress_bar is None:
                    progress_bar = tqdm(
                        total=SHARD_SIZE, unit="tokens", desc=f"Shard {shard_index}"
                    )
                progress_bar.update(len(tokens))

            else:
                # split the document into whatever fits in this shard; the remainder goes to next one
                remainder = SHARD_SIZE - token_count
                progress_bar.update(remainder)
                all_tokens_np[token_count : token_count + remainder] = tokens[
                    :remainder
                ]

                if progress_bar is not None:
                    progress_bar.close()
                progress_bar = None

                # write the current shard
                write_datafile(all_tokens_np, split, shard_index, output_dir)
                shard_index += 1

                # populate the next shard with the leftovers of the current doc
                all_tokens_np[0 : len(tokens) - remainder] = tokens[remainder:]
                token_count = len(tokens) - remainder

        # write any remaining tokens as the last shard
        if token_count != 0:
            write_datafile(all_tokens_np[:token_count], split, shard_index, output_dir)

        if progress_bar is not None:
            progress_bar.close()


def main():
    # download the dataset
    train_dataset, val_dataset = load_dataset(
        "cyrilzhang/TinyStories2-ascii",
        split=["train", "validation"],
        download_mode="reuse_cache_if_exists",
    )

    # init the tokenizer
    if USE_GPT2_TOKENIZER:
        tokenizer = tiktoken.get_encoding("gpt2")
        eos_token_id = tokenizer._special_tokens["<|endoftext|>"]  # end of text token
    else:
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file="tokenizer.json", eos_token="<|eos|>", pad_token="<|pad|>"
        )
        eos_token_id = tokenizer.eos_token_id

    # tokenize the dataset
    tokenize_fn = partial(tokenize, tokenizer=tokenizer, eos_token_id=eos_token_id)
    full_output_dir = os.path.join(os.path.dirname(__file__), OUTPUT_DIR)
    tokenize_dataset(train_dataset, tokenize_fn, "train", full_output_dir)
    tokenize_dataset(val_dataset, tokenize_fn, "val", full_output_dir)


if __name__ == "__main__":
    main()
