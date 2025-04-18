import os
import glob
import numpy as np
from tqdm import tqdm


DATA_ROOT = "../../blob/lm_datasets"
DATA_DIRS = [
    "code_contest",
    "stackoverflow-with-meta-data-filtered",
    "textbook",
    "the-stack-dedup-python-filtered",
]

OUTPUT_PATH = "../../blob/lm_datasets/phi_data_split"
OUTPUT_DTYPE = np.uint16

CONTEXT_LENGTH = 2048
VAL_RATIO = 0.01


def get_output_filenames(file_path: str):
    # Remove root directory
    if file_path.startswith(DATA_ROOT):
        file_path = file_path[len(DATA_ROOT) :]
        if file_path.startswith("/"):
            file_path = file_path[1:]

    # Remove the file name
    file_path = os.path.dirname(file_path)

    # Replace "/" with "_" to create a filename
    file_name = file_path.replace("/", "_")

    # Create the output file names
    file_out_train = os.path.join(OUTPUT_PATH, f"{file_name}_train.npy")
    file_out_val = os.path.join(OUTPUT_PATH, f"{file_name}_val.npy")

    return file_out_train, file_out_val


def split_file(
    file_in: str, file_out_train: str, file_out_val: str, random_seed: int = 0
):
    np.random.seed(random_seed)

    # Load input data
    print(f"Loading file {file_in}")
    data_in = np.load(file_in, mmap_mode="r", allow_pickle=False)
    print(f"Loaded numpy array with shape {data_in.shape} and dtype {data_in.dtype}")
    assert (
        data_in.ndim == 1
    ), f"Expected 1-D numpy array, but got {data_in.ndim}-D array of shape {data_in.shape} in {file_in}"

    # Compute number of sequences in each split
    n_seq = data_in.size // CONTEXT_LENGTH
    n_val_seq = int(VAL_RATIO * n_seq)
    n_train_seq = n_seq - n_val_seq
    print(f"Context length: {CONTEXT_LENGTH}")
    print(f"Total sequences: {n_seq}")
    print(f"Training sequences: {n_train_seq}")
    print(f"Validation sequences: {n_val_seq}")

    # Create 1D output arrays
    file_out_train_temp = f"{file_out_train}.tmp"
    file_out_val_temp = f"{file_out_val}.tmp"
    data_out_train = np.memmap(
        file_out_train_temp,
        dtype=OUTPUT_DTYPE,
        mode="w+",
        shape=(n_train_seq * CONTEXT_LENGTH,),
    )
    data_out_val = np.memmap(
        file_out_val_temp,
        dtype=OUTPUT_DTYPE,
        mode="w+",
        shape=(n_val_seq * CONTEXT_LENGTH,),
    )

    # Generate indices of sequences for validation split
    val_indices = np.random.choice(np.arange(n_seq), n_val_seq, replace=False)
    val_indices = set(val_indices)

    # Assign each sequence to its split
    train_count, val_count = 0, 0
    for i in tqdm(np.arange(n_seq)):
        in_start = i * CONTEXT_LENGTH
        in_end = (i + 1) * CONTEXT_LENGTH

        if i in val_indices:
            assert val_count < n_val_seq, f"Validation count exceeded {n_val_seq}"
            out_start = val_count * CONTEXT_LENGTH
            out_end = (val_count + 1) * CONTEXT_LENGTH
            data_out_val[out_start:out_end] = data_in[in_start:in_end]
            val_count += 1
        else:
            assert train_count < n_train_seq, f"Training count exceeded {n_train_seq}"
            out_start = train_count * CONTEXT_LENGTH
            out_end = (train_count + 1) * CONTEXT_LENGTH
            data_out_train[out_start:out_end] = data_in[in_start:in_end]
            train_count += 1

    # Save the temporary output arrays to final filenames
    print(f"Saving train data to {file_out_train}")
    np.save(file_out_train, data_out_train)
    print(f"Saving val data to {file_out_val}")
    np.save(file_out_val, data_out_val)

    # Clean up temporary files
    del data_out_train
    del data_out_val
    if os.path.exists(file_out_train_temp):
        os.remove(file_out_train_temp)
    if os.path.exists(file_out_val_temp):
        os.remove(file_out_val_temp)


def main():
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    for data_dir in DATA_DIRS:
        data_path = os.path.join(DATA_ROOT, data_dir)
        dataset_files = glob.glob(os.path.join(data_path, "**/*.npy"), recursive=True)
        print(f"Looking in subdirectory {data_dir}")
        print(f"Found dataset files: {dataset_files}")

        for file_path in dataset_files:
            file_out_train, file_out_val = get_output_filenames(file_path)
            split_file(
                file_path,
                file_out_train,
                file_out_val,
            )


if __name__ == "__main__":
    main()
