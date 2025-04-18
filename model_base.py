import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from liger_kernel.transformers.fused_linear_cross_entropy import (
    LigerFusedLinearCrossEntropyLoss,
)
from typing import Any, Dict, Optional, Tuple


class ModelBase:
    """Shared base class for BST and GPT models"""

    def __init__(self):
        # To be filled in by setup_fabric()
        self.fabric: L.Fabric = None

        # To be filled in by configure_optimizers()
        self.optimizer: torch.optim.Optimizer = None

        # Total number of training steps
        self.training_steps: int = 0

    def _assert_fabric_is_setup(self, setup: bool = True):
        """Checks that setup_fabric() has been called"""
        if setup:
            assert (
                self.fabric is not None
            ), "Fabric must be set up before calling this function"
        else:
            assert (
                self.fabric is None
            ), "This function must be called before setup_fabric()"

    def setup_fabric(self, fabric: L.Fabric):
        """
        Sets up the model with the provided Fabric object.
        This should set self.fabric = fabric and call fabric.setup_module()
        and fabric.setup_optimizers() on the applicable model and optimizer.
        """
        raise NotImplementedError

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: Tuple[float, float],
        use_fused: Optional[bool] = None,
    ):
        """Create optimizer for training the model"""
        raise NotImplementedError

    def compile(self):
        """Compiles the model for faster training and inference"""
        raise NotImplementedError

    def train(self):
        """Set the model to training mode"""
        raise NotImplementedError

    def eval(self):
        """Set the model to evaluation mode"""
        raise NotImplementedError

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Returns the number of parameters in the model"""
        raise NotImplementedError

    def compute_loss(
        self,
        batch: torch.Tensor,  # Shape is (batch_size, seq_len)
        backpropagate: bool,  # Run backward pass if true, otherwise only compute loss
        no_sync: bool = False,  # If True, don't sync gradients across multiple GPUs
        loss_div: int = 1,  # Loss will be divided by this number
        **kwargs,  # Extra arguments ignored for compatibility between models
    ) -> torch.Tensor:
        """Computes the loss for a batch of data"""
        raise NotImplementedError

    def optimizer_step(
        self,
        learning_rate: Optional[float] = None,  # If given, use this learning rate
        grad_clip: Optional[float] = None,  # If given, clip gradient norm
    ) -> Optional[torch.Tensor]:
        """
        Performs a single optimization step.
        If gradient clipping is enabled, return the gradient norm before clipping.
        """
        raise NotImplementedError

    def _get_checkpoint_state(self) -> Dict[str, Any]:
        """Get the state of the model for checkpoint save/load"""
        raise NotImplementedError

    def save_checkpoint(self, file_path: str):
        """
        Save the model checkpoint to the given file path.
        This includes the encoder, text head, optimizer, and training steps.
        """
        self._assert_fabric_is_setup()
        self.fabric.print(f"Saving checkpoint to {file_path}")
        state = self._get_checkpoint_state()
        if "training_steps" not in state:
            # Make sure to save training steps
            state["training_steps"] = self.training_steps
        self.fabric.save(file_path, state)

    def load_checkpoint(self, file_path: str, strict: bool = True):
        """
        Load the model checkpoint from the given file path.
        If model does not have optimizer, load only the model weights.
        """
        self._assert_fabric_is_setup()
        self.fabric.print(f"Loading checkpoint from {file_path}")
        state = self._get_checkpoint_state()
        if self.optimizer is None:
            # Optimizer has not been initialized, so don't load it
            # No optimizer needed for inference
            self.fabric.print(
                "Optimizer not configured, loading model for inference only"
            )
            state.pop("optimizer", None)
        # fabric.load() will in-place modify all objects in the state
        self.fabric.load(file_path, state, strict=strict)
        # Update training steps manually because it is an int
        self.training_steps = state["training_steps"]

    def get_custom_metrics(self) -> Dict[str, Any]:
        """
        Returns a dictionary of custom metrics for the model.
        This is used for logging and monitoring during training.
        """
        return {}


class DocumentRelativePositions:
    """
    Shared code between BST and GPT models for handling packed sequences.
    This is used for computing positions of each token relative to each document.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def create_position_indices(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Create document-relative position indices for a batch of token sequences.
        Position indices restart for each document, and a sequence can have multiple packed documents.

        In other words, the indices are absolute within a document but relative to each
        document across the entire sequence. EOS tokens always have position index 0.

        For example
            Sequence: [A, B, C, EOS, D, E, F, G, H, EOS, I, J]
            Result:   [1, 2, 3,  0,  1, 2, 3, 4, 5,  0,  1, 2]
        """
        assert self.config is not None, "Expected super-class to set self.config"
        assert self.config.eos_token_id is not None, "self.config.eos_token_id is None"

        batch_size, seq_len = batch.shape
        device = batch.device

        # Find positions of EOS tokens
        # Sequence: [A, B, C, EOS, D, E, F, G, H, EOS, I, J]
        # EOS:      [0, 0, 0,  1,  0, 0, 0, 0, 0,  1,  0, 0]
        eos_positions = batch == self.config.eos_token_id

        # Create indices relative to entire sequence
        # Sequence:    [A, B, C, EOS, D, E, F, G, H, EOS, I, J]
        # seq_indices: [0, 1, 2,  3,  4, 5, 6, 7, 8,  9, 10, 11]
        seq_indices = (
            torch.arange(seq_len, device=device)
            .unsqueeze(0)
            .expand(batch_size, seq_len)
        )

        # Compute offset of each document relative to the sequence
        # Start with tensor filled with -1
        # Replace EOS positions with their index, and leave all other positions as -1
        # Sequence:    [ A,  B,  C, EOS,  D,  E,  F,  G,  H, EOS, I,  J]
        # doc_offsets: [-1, -1, -1,  3,  -1, -1, -1, -1, -1,  9, -1, -1]
        doc_offsets = torch.full(
            (batch_size, seq_len), fill_value=-1, device=device, dtype=seq_indices.dtype
        )
        doc_offsets[eos_positions] = seq_indices[eos_positions]

        # Take cumulative maximum to get the offset of each document
        # Sequence:    [ A,  B,  C, EOS, D, E, F, G, H, EOS, I, J]
        # doc_offsets: [-1, -1, -1,  3,  3, 3, 3, 3, 3,  9,  9, 9]
        doc_offsets = torch.cummax(doc_offsets, dim=1).values

        # To get document indices, subtract the offset from the sequence indices
        # If first token is not EOS, the offset is -1, so the document indices correctly starts at 1
        # Sequence:    [ A,  B,  C, EOS, D, E, F, G, H, EOS, I, J]
        # seq_indices: [ 0,  1,  2,  3,  4, 5, 6, 7, 8,  9, 10, 11]
        # doc_offsets: [-1, -1, -1,  3,  3, 3, 3, 3, 3,  9,  9, 9]
        # Result:      [ 1,  2,  3,  0,  1, 2, 3, 4, 5,  0,  1, 2]
        doc_indices = seq_indices - doc_offsets

        return doc_indices


class FusedCrossEntropyLoss:
    """
    Wrapper that combines the last linear layer with cross entropy loss.
    This enables the use of fused Liger kernel to avoid storing the logits tensor,
    which saves memory when the vocabulary size is large.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fused_loss_fn = LigerFusedLinearCrossEntropyLoss(
            ignore_index=-100,
            reduction="mean",
        )

    def cross_entropy_loss(
        self,
        input: torch.Tensor,
        last_layer: nn.Linear,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute result of passing input through last_layer
        and then applying cross-entropy loss with targets.

        This function should only be called in forward()
        """
        assert self.config is not None, "Expected super-class to set self.config"

        input_flat = input.reshape(-1, input.size(-1))
        targets_flat = targets.reshape(-1)
        assert input_flat.size(0) == targets_flat.size(
            0
        ), f"Flattened input and target shapes do not match: {input_flat.shape} vs {targets_flat.shape}"

        if self.config.use_fused:
            return self._fused_cross_entropy_loss(input_flat, last_layer, targets_flat)
        else:
            return self._standard_cross_entropy_loss(
                input_flat, last_layer, targets_flat
            )

    # Fused kernel causes problems with torch.compile
    @torch.compiler.disable(recursive=True)
    def _fused_cross_entropy_loss(
        self,
        input_flat: torch.Tensor,
        last_layer: nn.Linear,
        targets_flat: torch.Tensor,
    ) -> torch.Tensor:
        # Make sure tensors are contiguous
        input_flat = input_flat.contiguous()
        targets_flat = targets_flat.contiguous()
        # Manually type cast last_layer weights to match input dtype
        # This is needed to avoid issues with mixed precision
        weight = last_layer.weight.to(dtype=input_flat.dtype)
        bias = (
            last_layer.bias.to(dtype=input_flat.dtype)
            if last_layer.bias is not None
            else None
        )
        return self._fused_loss_fn(weight, input_flat, targets_flat, bias=bias)

    def _standard_cross_entropy_loss(
        self,
        input_flat: torch.Tensor,
        last_layer: nn.Linear,
        targets_flat: torch.Tensor,
    ) -> torch.Tensor:
        logits = last_layer(input_flat)
        return F.cross_entropy(logits, targets_flat)


class RotaryPositionEmbedding:
    def __init__(self, max_seq_len: int, head_dim: int, base_freq: float = 10000):
        assert head_dim % 2 == 0, "Dimension of attention head must be even"
        # round max_seq_len up to next multiple of 256
        self.max_seq_len = (max_seq_len + 255) // 256 * 256
        self.head_dim = head_dim
        self.base_freq = base_freq
        self.cos_lookup = None
        self.sin_lookup = None

    @torch.no_grad()
    def __call__(self, pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the cosine and sine components of the rotary position embedding.
        Input is a tensor of shape (batch_size, seq_len) containing position indices
        Returns cos and sin tensors of shape (batch_size, seq_len, head_dim)
        """
        assert (
            pos.max() < self.max_seq_len
        ), f"Position index {pos.max()} exceeds max_seq_len {self.max_seq_len}"

        if self.cos_lookup is None or self.sin_lookup is None:
            # Precompute cos and sin values if not already done
            self._precompute_cos_sin(pos.device)

        if self.cos_lookup.device != pos.device:
            # Move precomputed values to the same device as pos
            self.cos_lookup = self.cos_lookup.to(pos.device)
            self.sin_lookup = self.sin_lookup.to(pos.device)

        return self.cos_lookup[pos], self.sin_lookup[pos]

    def _precompute_cos_sin(self, device: torch.device):
        # Always use full precision for precomputation
        with torch.autocast(device_type=device.type, enabled=False):
            # Create frequencies of shape (head_dim/2)
            dims = torch.arange(
                0, self.head_dim, 2, dtype=torch.int64, device=device
            ).float()
            freqs = 1.0 / (self.base_freq ** (dims / self.head_dim))
            # Create position indices of shape (max_seq_len)
            positions = torch.arange(
                0, self.max_seq_len, dtype=torch.int64, device=device
            ).float()
            # Compute angles of shape (max_seq_len, head_dim/2)
            angles = torch.outer(positions, freqs)
            # Repeat angles to get shape (max_seq_len, head_dim)
            angles = torch.cat((angles, angles), dim=-1)
            # Compute cosine and sine lookup tables
            self.cos_lookup = angles.cos()
            self.sin_lookup = angles.sin()

    @staticmethod
    def apply(
        q: torch.Tensor,
        k: torch.Tensor,
        rope_cos_sin: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary position embedding to query and key tensors.

        q and k have shape (batch_size, num_heads, seq_len, head_dim)
        cos and sin have shape (batch_size, seq_len, head_dim)

        If all sequences in the batch have the same positions,
        shape of cos and sin can also be (1, seq_len, head_dim)
        """
        cos, sin = rope_cos_sin
        assert cos is not None and sin is not None
        assert cos.shape == sin.shape

        # Unsqueeze cos and sin to match the dimensions of q and k
        # Shape becomes (batch_size, 1, seq_len, head_dim)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        # Apply rotations
        q_rot = q * cos + RotaryPositionEmbedding._rotate_half(q) * sin
        k_rot = k * cos + RotaryPositionEmbedding._rotate_half(k) * sin

        return q_rot, k_rot

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """
        Rotates half the hidden dims of the input.
        This is non-interleaved Llama / Huggingface style rotation:
            [1, 2, 3, 4, 5, 6, 7, 8] -> [-5, -6, -7, -8, 1, 2, 3, 4]
        https://github.com/huggingface/transformers/blob/v4.50.0/src/transformers/models/llama/modeling_llama.py#L151
        """
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. If bias=False, use RMSNorm."""

    def __init__(
        self,
        ndim: int,
        bias: bool,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.eps = eps

        if bias:
            self.weight = nn.Parameter(torch.ones(ndim))
            self.bias = nn.Parameter(torch.zeros(ndim))
        else:
            self.weight = nn.Parameter(torch.ones(ndim))
            self.register_parameter("bias", None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.bias is not None:
            return F.layer_norm(
                input, self.weight.shape, self.weight, self.bias, self.eps
            )
        else:
            # just use RMSNorm
            return F.rms_norm(input, self.weight.shape, self.weight, self.eps)


class SwiGLU(nn.Module):
    """Linear layer with SwiGLU activation function"""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__()
        self.gate_up = nn.Linear(input_size, 2 * output_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split the linear layer output into two chunks
        gate_up = self.gate_up(x)
        gate, up = gate_up.chunk(2, dim=-1)
        return F.silu(gate) * up
