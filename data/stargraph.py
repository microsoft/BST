import numpy as np
import torch
import lightning as L
from sklearn.model_selection import train_test_split


class Tokenizer:
    def __init__(self, maxNodes):
        self.maxNodes = maxNodes
        self.encoder = {str(i): i for i in range(maxNodes)}
        self.encoder["|"] = maxNodes
        self.encoder["="] = maxNodes + 1
        self.encoder["/"] = maxNodes + 2
        self.encoder["$"] = maxNodes + 3  # Padding token

        self.decoder = {i: str(i) for i in range(maxNodes)}
        self.decoder[maxNodes] = "|"
        self.decoder[maxNodes + 1] = "="
        self.decoder[maxNodes + 2] = "/"
        self.decoder[maxNodes + 3] = "$"
        self.decoder[maxNodes + 4] = ""

        self.numbers = set("0123456789")

        # stargraph has no eos token, padded sequence masking
        # doesn't need to happen, but this property is needed
        # for compatibility with the bst training code
        self.eos_token_id = maxNodes + 5

    def encode(self, data):
        out = []
        i = 0
        while i < len(data):
            if data[i] == ",":
                i += 1
                continue
            s = ""
            while i < len(data) and data[i] in self.numbers:
                s += data[i]
                i += 1
            if s:
                out.append(self.encoder[s])
            else:
                out.append(self.encoder[data[i]])
                i += 1

        return out

    def tokenize(self, prefix):
        prefix_tokens = self.encode(prefix)
        prefix_tokens.append(self.maxNodes + 4)

        seq = np.array(prefix_tokens)

        return seq, len(seq)


class StarGraphDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        line = self.data[idx].strip()
        seq, _ = self.tokenizer.tokenize(line)
        x = torch.tensor(seq, dtype=torch.long)
        return x


class StarGraphDataModule:
    """
    PyTorch Lightning style DataModule for StarGraph dataset
    """

    def __init__(self, fabric: L.Fabric, config):
        self.fabric = fabric
        self.batch_size = config.data.device_batch_size

        # Create tokenizer
        maxNodes = config.data.stargraph_max_nodes
        self.tokenizer = Tokenizer(maxNodes)

        # Load data
        with open(config.data.stargraph_data_path, "r") as f:
            data = f.readlines()

        # Split the data into training and validation sets
        train_data, val_data = train_test_split(data, test_size=0.1, random_state=42)
        self.train_dataset = StarGraphDataset(train_data, self.tokenizer)
        self.val_dataset = StarGraphDataset(val_data, self.tokenizer)

        graph_description_len, total_len = self._measure_index(data)
        print(
            f"[Stargraph] Prefix Length: {graph_description_len}, Sequence Length: {total_len}"
        )

        self.vocab_size = maxNodes + 5 + 1  # Total vocabulary size
        self.graph_description_len = graph_description_len
        self.total_len = total_len

    def _measure_index(self, data):
        line = data[0]
        line = line.strip()  # Remove any trailing whitespace
        prefix = line.split("=")[0] + "="
        _, prefix_len = self.tokenizer.tokenize(prefix)  # Tokenize the prefix
        _, seq_len = self.tokenizer.tokenize(line)  # Tokenize the entire line

        # one for beginning special token 54 at end of sequence, one so it appears on index of equals sign.
        graph_description_len = prefix_len - 2

        return graph_description_len, seq_len

    def update_config(self, config):
        config.model.vocab_size = self.vocab_size
        config.model.context_length = self.graph_description_len
        config.model.block_size = self.total_len

    def get_tokenizer(self):
        return self.tokenizer

    def train_dataloader(self):
        dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=self.collate_fn,
        )
        # This will automatically partition data between devices
        return self.fabric.setup_dataloaders(dataloader, use_distributed_sampler=True)

    def val_dataloader(self):
        dataloader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            collate_fn=self.collate_fn,
        )
        # This will automatically partition data between devices
        return self.fabric.setup_dataloaders(dataloader, use_distributed_sampler=True)

    @staticmethod
    def collate_fn(batch):
        x_batch = torch.stack([item for item in batch])
        return x_batch
