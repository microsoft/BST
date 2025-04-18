"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

Example usage:
python train.py --config path_to_your_config.yaml your.overrides=here

Or run with Lightning Fabric:
fabric run --strategy ddp --devices 4 --precision bf16-mixed train.py --config path_to_your_config.yaml your.overrides=here
"""

import os
import time
import torch
import wandb
import yaml
import wandb
import lightning as L
from argparse import ArgumentParser
from omegaconf import OmegaConf, DictConfig
from lightning.fabric.strategies.model_parallel import ModelParallelStrategy
from lightning.fabric.loggers import CSVLogger
from wandb.integration.lightning.fabric import WandbLogger

from core_train import Trainer, initialize_model, shard_model
from data.tinystories import TinyStoriesDataModule
from data.stargraph import StarGraphDataModule
from data.pretokenized import PretokenizedDataModule

from inference import Inference

DATAMODULES = {
    "tinystories": TinyStoriesDataModule,
    "stargraph": StarGraphDataModule,
    "pretokenized": PretokenizedDataModule,
}


def do_train(
    config: DictConfig, hide_progress_bar: bool = False, use_sharding: bool = False
):
    # Initialize logging
    experiment_name = "run" + str(time.time())
    config.trainer.experiment_name = experiment_name
    loggers = []
    if config.trainer.log_to_file:
        loggers.append(
            CSVLogger(
                root_dir=config.trainer.out_dir,
                name=experiment_name,
                flush_logs_every_n_steps=1,
            )
        )
    if config.trainer.log_to_wandb:
        loggers.append(
            WandbLogger(
                project=config.trainer.wandb_project,
                name=experiment_name,
            )
        )

    # Initialize Fabric
    if use_sharding:
        # Sharding must be defined before creating Fabric
        strategy = ModelParallelStrategy(
            parallelize_fn=shard_model, save_distributed_checkpoint=False
        )
        fabric = L.Fabric(loggers=loggers, strategy=strategy)
        fabric.print("Training with sharded model")
    else:
        # GPUs, parallelism, and precision are set by the "fabric run" command
        fabric = L.Fabric(loggers=loggers)

    # Calculate per device batch size
    assert (
        config.data.effective_batch_size % fabric.world_size == 0
    ), f"effective_batch_size {config.data.effective_batch_size} must be divisible by DDP world size {fabric.world_size}"
    config.data.device_batch_size = (
        config.data.effective_batch_size // fabric.world_size
    )

    # Calculate micro batch size
    assert (
        config.data.device_batch_size % config.data.gradient_accum_steps == 0
    ), f"device_batch_size {config.data.device_batch_size} must be divisible by gradient_accum_steps {config.data.gradient_accum_steps}"
    config.data.micro_batch_size = (
        config.data.device_batch_size // config.data.gradient_accum_steps
    )

    # Print config
    fabric.print(yaml.dump(OmegaConf.to_container(config)))

    tokens_per_update = config.data.effective_batch_size * config.model.block_size
    fabric.print(f"World size: {fabric.world_size}")
    fabric.print(f"Effective batch size: {config.data.effective_batch_size}")
    fabric.print(f"Per-GPU device batch size: {config.data.device_batch_size}")
    fabric.print(
        f"Batch size per gradient accumulation step: {config.data.micro_batch_size}"
    )
    fabric.print(f"Batch size to accumulate pairs: {config.data.pair_batch_size}")
    fabric.print(f"Tokens per gradient update: {tokens_per_update}")

    # Initialize PyTorch settings
    seed_offset = fabric.global_rank
    fabric.seed_everything(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    torch._dynamo.config.cache_size_limit = 16  # allow more recompiles

    # Load dataset
    assert (
        config.data.dataset in DATAMODULES
    ), f"Dataset {config.data.dataset} not supported"
    DataModuleClass = DATAMODULES[config.data.dataset]
    datamodule = DataModuleClass(fabric, config)
    datamodule.update_config(config)
    tokenizer = datamodule.get_tokenizer()

    # Create output directory
    experiment_out_dir = os.path.join(config.trainer.out_dir, experiment_name)
    if fabric.global_rank == 0:
        os.makedirs(experiment_out_dir, exist_ok=True)
    fabric.print(f"Output directory: {experiment_out_dir}")

    # dump config to out_dir
    if fabric.global_rank == 0:
        OmegaConf.save(
            config, os.path.join(experiment_out_dir, "materialized_config.yaml")
        )

    if config.use_bst and config.trainer.print_samples:
        inference = Inference(fabric, config, model, tokenizer)
        generate_samples_func = inference.generate_samples
    else:
        generate_samples_func = None

    # wandb login
    if (
        config.trainer.log_to_wandb
        and fabric.global_rank == 0
        and "WANDB_API_KEY" in os.environ
    ):
        wandb.login(
            key=os.environ.get("WANDB_API_KEY"),
            host=os.environ.get("WANDB_HOST"),
            timeout=0,
        )

    # Create model and trainer
    model = initialize_model(fabric, config, tokenizer, initialize_optimizer=True)
    trainer = Trainer(
        fabric=fabric,
        config=config,
        model=model,
        show_progress_bar=(fabric.local_rank == 0 and not hide_progress_bar),
        generate_samples_func=generate_samples_func,
    )

    # Start training
    trainer.train(datamodule)


if __name__ == "__main__":
    parser = ArgumentParser(
        prog="BST_Trainer",
        description="Belief State Transformer (BST) trainer",
    )
    parser.add_argument("-c", "--config", required=True, help="Path to config file")
    parser.add_argument("--no_pbar", action="store_true", help="Disable progress bar")
    parser.add_argument("--shard", action="store_true", help="Shard the model")
    args, conf_cli = parser.parse_known_args()

    default = OmegaConf.load("defaults.yaml")
    overrides = OmegaConf.load(args.config)
    cli = OmegaConf.from_dotlist(conf_cli)

    config = OmegaConf.merge(default, overrides, cli)
    do_train(config, hide_progress_bar=args.no_pbar, use_sharding=args.shard)
