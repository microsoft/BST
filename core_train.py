"""contains core training/inference logic used for multiple datasets/tasks"""

import gc
import inspect
import math
import os
import torch
import wandb
import lightning as L
from datetime import datetime
from lightning.fabric.strategies.model_parallel import ModelParallelStrategy
from torch.distributed.fsdp import fully_shard
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Any, Callable, Dict, Optional, Tuple

from model_base import ModelBase
from model_bst import BST, BSTConfig
from model_gpt import GPT, GPTConfig, Block


def initialize_model(
    fabric: L.Fabric,
    config,
    tokenizer,
    initialize_optimizer=True,
    checkpoint_path: Optional[str] = None,
):
    fabric.print(f"Initializing model with config: {config}")

    if config.use_bst:
        ModelClass = BST
        ModelConfigClass = BSTConfig
        single_gap_modes = ["next_token", "eos"]
        assert (
            config.model.bst_single_gap_prediction_mode in single_gap_modes
        ), f"BST single gap mode must be one of {single_gap_modes}"
    else:
        ModelClass = GPT
        ModelConfigClass = GPTConfig
        gpt_modes = ["next_token", "fim"]
        assert (
            config.model.gpt_mode in gpt_modes
        ), f"GPT mode must be one of {gpt_modes}"

    # start with model_args from command line
    model_args = dict(
        n_layer=config.model.n_layer,
        n_head=config.model.n_head,
        n_embd=config.model.n_embd,
        block_size=config.model.block_size,
        bias=config.model.bias,
        vocab_size=config.model.vocab_size,
        dropout=config.model.dropout,
        eos_token_id=tokenizer.eos_token_id,
        use_fused=config.trainer.use_fused_kernels,
    )

    if config.use_bst:
        # BST-specific parameters
        model_args = {
            **model_args,
            "context_length": config.model.context_length,
            "bst_pair_minimum_gap": config.model.bst_pair_minimum_gap,
            "bst_pair_maximum_gap": config.model.bst_pair_maximum_gap,
            "bst_pair_subsample_rate": config.model.bst_pair_subsample_rate,
            "bst_single_gap_prediction_mode": config.model.bst_single_gap_prediction_mode,
        }
    else:
        # GPT-specific parameters
        model_args = {
            **model_args,
            "goal_range": config.data.goal_range,
            "fim_token_id": (
                -1
                if config.model.gpt_mode != "fim"
                else tokenizer.convert_tokens_to_ids("<|fim|>")
            ),
            "is_fim_mode": config.model.gpt_mode == "fim",
        }
    model_config = ModelConfigClass(**model_args)

    # Load a checkpoint file
    if checkpoint_path:
        assert os.path.isfile(
            checkpoint_path
        ), f"Checkpoint file {checkpoint_path} does not exist"

        # Create an empty model because we will load the weights from checkpoint
        with fabric.init_module(empty_init=True):
            model = ModelClass(model_config)

    # Resume from a previous training run
    elif config.trainer.init_from == "resume":
        recovery_ckpt_pointer = os.path.join(config.trainer.out_dir, "recovery_ckpt")
        latest_ckpt_pointer = os.path.join(config.trainer.out_dir, "latest_ckpt")

        # Use the recovery checkpoint if it exists
        if os.path.isfile(recovery_ckpt_pointer):
            with open(recovery_ckpt_pointer, "r") as f:
                checkpoint_path = f.read().strip()
            assert os.path.isfile(
                checkpoint_path
            ), f"Checkpoint file {checkpoint_path} does not exist"
            fabric.print(f"Resuming from recovery checkpoint {checkpoint_path}")

        # Otherwise, use the latest validation checkpoint if it exists
        elif os.path.isfile(latest_ckpt_pointer):
            with open(latest_ckpt_pointer, "r") as f:
                checkpoint_path = f.read().strip()
            assert os.path.isfile(
                checkpoint_path
            ), f"Checkpoint file {checkpoint_path} does not exist"
            fabric.print(
                f"Resuming from previous validation checkpoint {checkpoint_path}"
            )

        # If no checkpoint file is found, initialize a new model
        else:
            fabric.print(f"Could not find checkpoint file {recovery_ckpt_pointer}")
            fabric.print(f"Could not find checkpoint file {latest_ckpt_pointer}")
            checkpoint_path = None

        # Empty init if we have found a checkpoint
        with fabric.init_module(empty_init=(checkpoint_path is not None)):
            model = ModelClass(model_config)

    # Initialize a new model from scratch
    elif config.trainer.init_from == "scratch":
        with fabric.init_module():
            model = ModelClass(model_config)

    # Initialize from OpenAI GPT-2 weights
    elif config.trainer.init_from.startswith("gpt2"):  # currently broken @dayan
        fabric.print(f"Initializing from OpenAI GPT-2 weights: {config.init_from}")
        assert not config.use_bst, "BST not supported with GPT-2 weights"
        override_args = dict(dropout=config.model.dropout)
        model = ModelClass.from_pretrained(config.trainer.init_from, override_args)
        # read off the created config params, so we can store them into checkpoint correctly
        for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
            model_args[k] = getattr(model.config, k)

    else:
        raise ValueError(
            f"Invalid init_from value: {config.trainer.init_from}. Must be 'resume', 'scratch', or 'gpt2'."
        )

    # crop down the model block size if desired, using model surgery
    # so that the checkpoint will have the right value
    if config.model.block_size < model.config.block_size:
        assert not config.use_bst, "cropping block size not supported for BST"
        model.crop_block_size(config.model.block_size)
        model_args["block_size"] = config.model.block_size

    fabric.print(
        f"Number of parameters (including embedding): {model.get_num_params(non_embedding=False):,}"
    )
    fabric.print(
        f"Number of parameters (excluding embedding): {model.get_num_params(non_embedding=True):,}"
    )

    # Compile must occur before fabric.setup()
    if config.trainer.compile:
        fabric.print("Compiling model")
        model.compile()

    # FSDP requires model to be setup before initializing the optimizer
    if isinstance(fabric.strategy, ModelParallelStrategy):
        model.setup_fabric(fabric)

    # Initialize optimizer
    if initialize_optimizer:
        is_device_cuda = fabric.device.type == "cuda"
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        # This is native PyTorch fused optimizer, not use_fused_kernels in config
        use_fused = fused_available and is_device_cuda

        model.configure_optimizers(
            weight_decay=config.optimizer.weight_decay,
            learning_rate=config.optimizer.learning_rate,
            betas=(config.optimizer.beta1, config.optimizer.beta2),
            use_fused=use_fused,
        )

    # Setup model and optimizer
    # In the FSDP case, model is already setup and this sets up only the optimizer
    model.setup_fabric(fabric)

    # Load checkpoint file
    if checkpoint_path:
        model.load_checkpoint(checkpoint_path)
        fabric.print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        fabric.print("Initialized a new model from scratch")

    return model


def shard_model(
    module: torch.nn.Module, device_mesh: torch.distributed.device_mesh.DeviceMesh
):
    """
    Function to define the sharding strategy for the model.

    Given a 2D device mesh of (nodes, gpus per node), fully_shard will:
        - Replicate across nodes (data parallel)
        - Shard across GPUs within a node (FSDP)
    """

    # Function to shard individual transformer layers recursively
    # This lets us only gather full weights one layer at a time
    def _shard_recursive(module: torch.nn.Module):
        for submodule in module.children():
            if isinstance(submodule, Block):
                submodule = fully_shard(
                    submodule,
                    mesh=device_mesh,
                    reshard_after_forward=True,
                )
            else:
                _shard_recursive(submodule)

    # Shard the submodules
    _shard_recursive(module)

    # Shard the top level module
    fully_shard(module, mesh=device_mesh, reshard_after_forward=False)

    return module


class Trainer:
    """Trainer class for training/validating BST and GPT models"""

    def __init__(
        self,
        fabric: L.Fabric,
        config,
        model: ModelBase,
        show_progress_bar: bool = True,
        generate_samples_func: Optional[Callable] = None,
    ):
        # Check that model has optimizer
        assert (
            hasattr(model, "optimizer") and model.optimizer is not None
        ), "Model must be initialized with optimizer"

        # Initialization state
        self.fabric = fabric
        self.config = config
        self.model = model
        self.show_progress_bar = show_progress_bar
        self.generate_samples_func = generate_samples_func

        # Training loop state
        self.train_dataloader: DataLoader = None
        self.val_dataloader: DataLoader = None
        self.tokenizer = None
        self.prepare_batch_func: Optional[Callable] = None
        self.epoch: int = 0
        self.step: int = self.model.training_steps
        self.last_train_loss: Optional[float] = None
        self.best_val_loss: Optional[float] = None

        # Checkpointing state
        self.latest_checkpoint_path = None
        self.best_checkpoint_path = None
        self.recovery_checkpoint_path = None
        self.checkpoints_to_always_keep = set()

    def train(self, datamodule):
        """
        Call this to start training.
        """
        # Initialize dataloaders
        self.train_dataloader = datamodule.train_dataloader()
        self.val_dataloader = datamodule.val_dataloader()
        self.tokenizer = datamodule.get_tokenizer()

        # Check if datamodule has a prepare_batch function
        if hasattr(datamodule, "prepare_batch"):
            self.prepare_batch_func = datamodule.prepare_batch
            self.fabric.print(
                f"Using prepare batch function {type(datamodule).__name__}.prepare_batch()"
            )
        else:
            self.prepare_batch_func = None

        # Do training or validation
        if self.config.trainer.val_only:
            self.fabric.print("Running validation only, not training model")
            validation_logs = self._validation_loop()
            self.fabric.print(validation_logs)
        else:
            self.fabric.print(f"Starting training from step={self.step}")
            self._train_loop()

            self.fabric.print("Training complete, running final validation")
            val_logs = self._validation_loop()

            # Save final model checkpoint as wandb artifact
            if self.config.trainer.log_to_wandb and self.fabric.global_rank == 0:
                # Initialize wandb artifact
                artifact = wandb.Artifact(
                    name=self.config.trainer.experiment_name,
                    type="model",
                    metadata={
                        "config": self.config,
                        "validation_logs": val_logs,
                        "last_train_loss": self.last_train_loss,
                        "best_val_loss": self.best_val_loss,
                    },
                )
                # Get the latest checkpoint filename
                latest_ckpt_pointer = os.path.join(
                    self.config.trainer.out_dir, "latest_ckpt"
                )
                if os.path.isfile(latest_ckpt_pointer):
                    # Upload the checkpoint
                    with open(latest_ckpt_pointer, "r") as f:
                        checkpoint_path = f.read().strip()
                    assert os.path.isfile(
                        checkpoint_path
                    ), f"Checkpoint file {checkpoint_path} does not exist"
                    self.fabric.print(f"Uploading {checkpoint_path} to wandb artifact")
                    artifact.add_file(
                        checkpoint_path,
                        name=os.path.basename(checkpoint_path),
                    )
                    wandb.run.log_artifact(artifact)
                else:
                    # Checkpoint file not found
                    self.fabric.print(
                        f"Warning: {latest_ckpt_pointer} file not found, not saving wandb artifact"
                    )

    def _train_loop(self):
        """
        Main training loop.
        Iterate over the train dataloader for the configured number of batches.

        For each batch:
            - If we have reached the validation interval, run validation
            - Split batch into micro batches and accumulate gradients
            - Run optimizer step
            - Do logging
        """
        # Set model to train mode
        self.model.train()

        # Sum of train loss over log_interval batches
        running_train_loss = torch.zeros(1, device=self.fabric.device)

        # Initialize progress bar
        if self.show_progress_bar:
            pbar = tqdm(total=self.config.trainer.val_interval, leave=False)

        # If resuming from a previous checkpoint, fast-forward data
        if self.step > 0:
            do_fast_forward = True
            fast_forward_steps = 0
        else:
            do_fast_forward = False

        while True:
            # Do one epoch over the entire training dataloader
            for batch in self.train_dataloader:
                # Fast forward data
                if do_fast_forward:
                    # Increment fast forward counter until we reach the current step
                    fast_forward_steps += 1
                    if fast_forward_steps >= self.step:
                        self.fabric.print(
                            f"Fast forwarded data to step {fast_forward_steps}"
                        )
                        do_fast_forward = False
                    # Skip this batch
                    continue

                # Garbage collect to free up memory
                if (
                    self.config.trainer.garbage_collect > 0
                    and self.step % self.config.trainer.garbage_collect == 0
                ):
                    gc.collect()

                # Validation
                validation_logs = {}
                if (
                    self.step % self.config.trainer.val_interval == 0 and self.step > 0
                ) or self.config.trainer.val_only:
                    # Close the previous training progress bar
                    if self.show_progress_bar:
                        pbar.close()

                    validation_logs = self._validation_loop()
                    self.model.train()

                    if self.config.trainer.garbage_collect > 0:
                        gc.collect()

                    # Initialize a new progress bar
                    if self.show_progress_bar:
                        pbar = tqdm(total=self.config.trainer.val_interval, leave=False)

                # Prepare batch
                if self.prepare_batch_func is not None:
                    batch = self.prepare_batch_func(batch)

                # Accumulate gradients over gradient_accum_steps
                for accum_step in range(self.config.data.gradient_accum_steps):
                    start_idx = accum_step * self.config.data.micro_batch_size
                    end_idx = (accum_step + 1) * self.config.data.micro_batch_size
                    sub_batch = batch[start_idx:end_idx]

                    # Only sync gradients for the last step
                    no_sync = accum_step < self.config.data.gradient_accum_steps - 1
                    loss = self.model.compute_loss(
                        sub_batch,
                        pair_batch_size=self.config.data.pair_batch_size,
                        backpropagate=True,
                        loss_div=self.config.data.gradient_accum_steps,
                        no_sync=no_sync,
                    )

                    # loss is already divided by gradient_accum_steps
                    running_train_loss += loss

                # Optimizer step
                # This should increment model.training_steps
                lr = self._get_lr()
                grad_clip = self.config.optimizer.grad_clip
                grad_norm = self.model.optimizer_step(
                    learning_rate=lr, grad_clip=grad_clip
                )

                # Logging
                if self.step % self.config.trainer.log_interval == 0 and self.step > 0:
                    # Train loss has been accumulated over log_interval batches
                    self.last_train_loss = (
                        running_train_loss.item() / self.config.trainer.log_interval
                    )
                    running_train_loss.zero_()

                    custom_metrics = self.model.get_custom_metrics()
                    log_dict = {
                        "step": self.step,
                        "epoch": self.epoch,
                        "lr": lr,
                        "train_loss": self.last_train_loss,
                        **validation_logs,
                        **custom_metrics,
                    }
                    if grad_norm is not None:
                        log_dict["grad_norm"] = grad_norm

                    self.fabric.log_dict(log_dict)

                # Update progress bar
                progress_str = f"{datetime.now()} Step: {self.step} Epoch: {self.epoch} Train loss: {self.last_train_loss or 0.0:.4f}"
                if self.show_progress_bar:
                    pbar.set_description(progress_str)
                    pbar.update(1)
                else:
                    self.fabric.print(progress_str)

                # Increment batch count
                self.step += 1
                assert (
                    self.step == self.model.training_steps
                ), "Bug in training loop: trainer step count does not match model step count"
                if self.step > self.config.trainer.train_batches:
                    # End of training
                    return

                # Save recovery checkpoint
                if (
                    self.config.trainer.save_recovery_checkpoint > 0
                    and self.step % self.config.trainer.save_recovery_checkpoint == 0
                ):
                    self._save_recovery_checkpoint()

            # End of one full iteration over train dataloader
            self.epoch += 1

    @torch.inference_mode()
    def _validation_loop(self) -> Dict[str, Any]:
        """
        Main validation loop.
        This is called every val_interval steps during training.

        For validation, we do the following:
            - Compute validation loss
            - For BST, compute next/prev token prediction loss
            - Save checkpoint
            - If generate_samples_func is provided, generate samples
        """
        # Set model to eval mode
        self.model.eval()

        if self.config.trainer.val_batches > 0:
            # Divide total validation batches by world size
            n_val_batches = self.config.trainer.val_batches // self.fabric.world_size
        else:
            # If validation batches are not specified, validate over entire dataloader
            n_val_batches = None

        validation_logs = {}
        batch_count = 0
        total_loss = torch.zeros(1, device=self.fabric.device)

        disable_pbar = (
            True if not self.show_progress_bar else None
        )  # None means automatic

        # Main validation loop
        for batch in tqdm(
            self.val_dataloader,
            desc="Validation",
            total=n_val_batches or len(self.val_dataloader),
            leave=False,
            disable=disable_pbar,
        ):
            if self.prepare_batch_func is not None:
                batch = self.prepare_batch_func(batch)

            # Accumulate loss over gradient_accum_steps
            for accum_step in range(self.config.data.gradient_accum_steps):
                start_idx = accum_step * self.config.data.micro_batch_size
                end_idx = (accum_step + 1) * self.config.data.micro_batch_size
                sub_batch = batch[start_idx:end_idx]

                loss = self.model.compute_loss(
                    sub_batch,
                    pair_batch_size=self.config.data.pair_batch_size,
                    backpropagate=False,
                    loss_div=self.config.data.gradient_accum_steps,
                )
                # loss is already divided by gradient_accum_steps
                total_loss += loss

            # End loop if limit_batches is reached
            batch_count += 1
            if n_val_batches is not None and batch_count >= n_val_batches:
                break

        # Sync average loss across all GPUs by uniform mean of per-device averages
        # This assumes that each GPU has roughly the same number of batches
        device_avg_loss = total_loss / batch_count
        global_avg_loss = self.fabric.all_reduce(device_avg_loss, reduce_op="mean")
        val_loss = global_avg_loss.item()

        # Add validation loss to log
        validation_logs["val_loss"] = val_loss

        # For BST, also compute next/prev token prediction loss
        if isinstance(self.model, BST):
            val_loss_next_token, val_loss_prev_token = self._bst_next_prev_token_loss()
            validation_logs["val_loss_next_token"] = val_loss_next_token
            validation_logs["val_loss_prev_token"] = val_loss_prev_token

        self.fabric.print(
            f"Step: {self.step} Epoch: {self.epoch} Train loss: {self.last_train_loss} Validation loss: {val_loss}"
        )

        # Update best validation loss
        is_new_best = (
            True if self.best_val_loss is None else (val_loss < self.best_val_loss)
        )
        if is_new_best:
            self.best_val_loss = val_loss

        # Save checkpoint
        if not self.config.trainer.val_only and (
            self.config.trainer.save_last_checkpoint
            or (self.config.trainer.save_best_checkpoint and is_new_best)
            or (
                self.config.trainer.keep_checkpoint_steps
                and self.step in self.config.trainer.keep_checkpoint_steps
            )
        ):
            filename = f"ckpt_iter_{self.step}_loss_{val_loss}.pt"
            new_ckpt_path = self._save_checkpoint(filename)

            # Delete outdated checkpoints
            if not self.config.trainer.always_save_checkpoint:
                old_checkpoints = set()
                if (
                    self.config.trainer.save_last_checkpoint
                    and self.latest_checkpoint_path is not None
                    and (
                        self.latest_checkpoint_path != self.best_checkpoint_path
                        or is_new_best
                    )
                ):
                    old_checkpoints.add(self.latest_checkpoint_path)
                if (
                    self.config.trainer.save_best_checkpoint
                    and is_new_best
                    and self.best_checkpoint_path is not None
                ):
                    old_checkpoints.add(self.best_checkpoint_path)
                if self.fabric.global_rank == 0:
                    for ckpt in old_checkpoints:
                        if ckpt not in self.checkpoints_to_always_keep:
                            os.remove(ckpt)

            # Update latest and best checkpoint paths
            if self.config.trainer.save_last_checkpoint:
                self.latest_checkpoint_path = new_ckpt_path
            if self.config.trainer.save_best_checkpoint and is_new_best:
                self.best_checkpoint_path = new_ckpt_path
            if (
                self.config.trainer.keep_checkpoint_steps
                and self.step in self.config.trainer.keep_checkpoint_steps
            ):
                self.checkpoints_to_always_keep.add(new_ckpt_path)

        # Generate samples
        if self.generate_samples_func is not None:
            # generate_samples should be a generator function that yields one sample at a time, therefore use next(..) to avoid lazy evaluation
            self.fabric.print(
                next(
                    self.generate_samples_func(batch, self.config.trainer.sampling_mode)
                )
            )

        return validation_logs

    @torch.inference_mode()
    def _bst_next_prev_token_loss(self) -> Tuple[float, float]:
        """
        Compute next/previous token prediction loss for BST.
        The next token loss computed here is equivalent to the GPT validation loss.
        This lets us have a fair comparison to GPT.
        """
        assert isinstance(self.model, BST), "This function is only valid for BST"

        if self.config.trainer.val_batches > 0:
            # Divide total validation batches by world size
            n_val_batches = self.config.trainer.val_batches // self.fabric.world_size
        else:
            # If validation batches are not specified, validate over entire dataloader
            n_val_batches = None

        batch_count = 0
        total_next_loss = torch.zeros(1, device=self.fabric.device)
        total_prev_loss = torch.zeros(1, device=self.fabric.device)

        disable_pbar = (
            True if not self.show_progress_bar else None
        )  # None means automatic

        for batch in tqdm(
            self.val_dataloader,
            desc="BST next/prev token loss",
            total=n_val_batches or len(self.val_dataloader),
            leave=False,
            disable=disable_pbar,
        ):
            if self.prepare_batch_func is not None:
                batch = self.prepare_batch_func(batch)

            # Accumulate loss over gradient_accum_steps
            for accum_step in range(self.config.data.gradient_accum_steps):
                start_idx = accum_step * self.config.data.micro_batch_size
                end_idx = (accum_step + 1) * self.config.data.micro_batch_size
                sub_batch = batch[start_idx:end_idx]

                # Create tensor of -1 with length equal to sub_batch size
                neg_one = torch.full(
                    (sub_batch.size(0),),
                    fill_value=-1,
                    device=sub_batch.device,
                    dtype=torch.long,
                )

                # Compute next/prev token prediction loss
                # We do not have prefix/suffix as prompt, so set indices to -1
                next_token_losses, prev_token_losses = self.model.evaluation_loss(
                    sub_batch,
                    prefix_end_index=neg_one,
                    suffix_start_index=neg_one,
                )

                # Calculate mean loss over sequences in sub-batch
                # Divide loss by gradient_accum_steps so we sum to the batch mean
                next_token_losses = (
                    next_token_losses.mean() / self.config.data.gradient_accum_steps
                )
                prev_token_losses = (
                    prev_token_losses.mean() / self.config.data.gradient_accum_steps
                )
                total_next_loss += next_token_losses
                total_prev_loss += prev_token_losses

            # End loop if limit_batches is reached
            batch_count += 1
            if n_val_batches is not None and batch_count >= n_val_batches:
                break

        # Sync average loss across all GPUs
        device_avg_next_loss = total_next_loss / batch_count
        device_avg_prev_loss = total_prev_loss / batch_count
        global_avg_next_loss, global_avg_prev_loss = self.fabric.all_reduce(
            (device_avg_next_loss, device_avg_prev_loss), reduce_op="mean"
        )

        return global_avg_next_loss.item(), global_avg_prev_loss.item()

    def _save_checkpoint(self, filename: str) -> str:
        """
        Save the model checkpoint to the given file name.
        Then save the checkpoint file path to the latest_ckpt file.
        """
        # Save the checkpoint file itself
        ckpt_dir = os.path.join(
            self.config.trainer.out_dir,
            self.config.trainer.experiment_name,
        )
        ckpt_path = os.path.join(ckpt_dir, filename)
        self.model.save_checkpoint(ckpt_path)

        # Save the file path to the latest checkpoint
        if self.fabric.global_rank == 0:
            with open(
                os.path.join(self.config.trainer.out_dir, "latest_ckpt"),
                "w",
            ) as f:
                f.write(ckpt_path)

        return ckpt_path

    def _save_recovery_checkpoint(self):
        """
        Save a checkpoint for recovery from crashes.
        """
        # Save the checkpoint file
        ckpt_dir = os.path.join(
            self.config.trainer.out_dir,
            self.config.trainer.experiment_name,
        )
        ckpt_path = os.path.join(ckpt_dir, f"recovery_ckpt_iter_{self.step}.pt")
        self.model.save_checkpoint(ckpt_path)

        # Save the most recent file path to the recovery checkpoint pointer file
        if self.fabric.global_rank == 0:
            with open(
                os.path.join(self.config.trainer.out_dir, "recovery_ckpt"),
                "w",
            ) as f:
                f.write(ckpt_path)

        # Delete the old recovery checkpoint file if it exists
        if self.recovery_checkpoint_path is not None:
            if self.fabric.global_rank == 0:
                os.remove(self.recovery_checkpoint_path)

        # Update the recovery checkpoint path
        self.recovery_checkpoint_path = ckpt_path

    def _get_lr(self) -> float:
        """
        Cosine learning rate decay with warmup
        """
        iter_num = self.step
        warmup_iters = self.config.lr_scheduler.warmup_iters
        decay_iters = self.config.lr_scheduler.lr_decay_iters
        max_lr = self.config.optimizer.learning_rate
        min_lr = self.config.lr_scheduler.min_lr

        # If decay_lr is False, always use max_lr
        if not self.config.lr_scheduler.decay_lr:
            return max_lr

        # 1) linear warmup for warmup_iters steps
        if iter_num < warmup_iters:
            return max_lr * iter_num / warmup_iters

        # 2) if it > lr_decay_iters, return min learning rate
        if iter_num > decay_iters:
            return min_lr

        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (iter_num - warmup_iters) / (decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return min_lr + coeff * (max_lr - min_lr)
