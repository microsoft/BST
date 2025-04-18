"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from dataclasses import dataclass
from lightning.fabric import is_wrapped
from torch.distributed.fsdp import FSDPModule
from typing import Any, Dict, Optional, Tuple

from model_base import (
    ModelBase,
    DocumentRelativePositions,
    FusedCrossEntropyLoss,
    LayerNorm,
    RotaryPositionEmbedding,
    SwiGLU,
)


@dataclass
class GPTConfig:
    block_size: int = 1024
    # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    bias: bool = False
    goal_range: Tuple[int] = (25, 75)
    eos_token_id: int = -1
    fim_token_id: int = -1
    is_fim_mode: bool = False
    # True: use fused Liger kernels. False: use regular PyTorch functions.
    use_fused: bool = False


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = (
            nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        )
        self.resid_dropout = (
            nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        )
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print(
                "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0"
            )
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        # input should have shape (batch_size, seq_len, n_embd)
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # convert to shape (B, n_head, T, d_head)
        d_head = C // self.n_head
        k = k.view(B, T, self.n_head, d_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, d_head).transpose(1, 2)

        # apply rotary position embedding to q and k
        if rope is not None:
            q, k = RotaryPositionEmbedding.apply(q, k, rope)

        # causal self-attention; Self-attend: (B, Nh, T, Dh) x (B, Nh, Dh, T) -> (B, Nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            if mask is not None:
                mask = mask.view(B, 1, T, T)
            y = torch.nn.functional.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0,
                is_causal=(mask is None),
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            if mask is not None:
                att.masked_fill_(mask.logical_not(), float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).reshape(
            B, T, C
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        hidden_dim = 8 * config.n_embd / 3
        hidden_dim = 128 * round(hidden_dim / 128)
        self.up = SwiGLU(config.n_embd, hidden_dim, bias=config.bias)
        self.down = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.dropout = (
            nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.down(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(
            config.n_embd,
            bias=config.bias,
        )
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(
            config.n_embd,
            bias=config.bias,
        )
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), mask=mask, rope=rope)
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(DocumentRelativePositions, FusedCrossEntropyLoss, nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # shared for forward and backward encoders
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)

        # add 1 extra position embedding for implicit EOS at start
        n_pos_embd = config.block_size + 1
        self.rotary_embedding = RotaryPositionEmbedding(
            max_seq_len=n_pos_embd,
            head_dim=config.n_embd // config.n_head,
        )

        self.transformer = nn.ModuleDict(
            dict(
                blocks=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                norm=LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # init all weights
        self.apply(self._init_weights)

    def get_num_params(self, non_embedding=True) -> int:
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the token embeddings get subtracted.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.token_embedding.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @torch.no_grad()
    def create_attention_mask(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Create causal attention mask for a sequence of packed documents.
        Mask will avoid attending to tokens in different documents.
        """
        # batch has shape (batch_size, seq_len)

        # find positions of eos tokens
        # shape is (batch_size, seq_len)
        eos_positions = batch == self.config.eos_token_id

        # create indices of packed documents in the token sequence
        # each token can attend up to and including previous EOS, but not next EOS
        # shape is (batch_size, seq_len)
        document_id = torch.cumsum(eos_positions, dim=1)

        # create mask of tokens within the same document
        # shape is (batch_size, seq_len, seq_len)
        document_mask = document_id.unsqueeze(1) == document_id.unsqueeze(2)

        # lower triangular causal mask
        mask = torch.tril(document_mask)

        return mask

    def forward(
        self,
        batch: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len = batch.size()
        assert (
            seq_len <= self.config.block_size
        ), f"Cannot forward sequence of length {seq_len}, block size is only {self.config.block_size}"

        # token positions relative to each document
        pos = self.create_position_indices(batch)
        rope = self.rotary_embedding(pos)

        if mask is None:
            # causal attention mask for packed sequence
            mask = self.create_attention_mask(batch)

        # token embeddings of shape (b, t, n_embd)
        x = self.token_embedding(batch)

        for block in self.transformer.blocks:
            x = block(x, mask=mask, rope=rope)
        x = self.transformer.norm(x)

        # If no targets given, return logits
        if targets is None:
            logits = self.lm_head(x)
            return logits

        # If targets are given, compute loss
        else:
            return self.cross_entropy_loss(
                input=x,
                last_layer=self.lm_head,
                targets=targets,
            )

    def crop_block_size(self, block_size: int):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        for block in self.transformer.blocks:
            if hasattr(block.attn, "bias"):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]


class GPT(ModelBase):
    def __init__(
        self,
        config: GPTConfig,
    ):
        super().__init__()
        if config.is_fim_mode:
            assert config.fim_token_id > 0, "fim_token_id must be set"
            assert (
                config.fim_token_id < config.vocab_size
            ), f"fim_token_id={config.fim_token_id} exceeds vocab size={config.vocab_size}. Did you forget to increment vocab_size to include the FIM token?"

        self.config = config
        self.model = Transformer(config)

    def setup_fabric(self, fabric: L.Fabric):
        """
        Setup Lightning Fabric for distributed training
        This wraps the models and optimizer with a FabricModule
        """
        self.fabric = fabric

        if not is_wrapped(self.model):
            self.model = fabric.setup_module(self.model)
            # Print model architecture
            self.fabric.print(self.model)
            self.fabric.print(f"Total number of parameters: {self.get_num_params():,}")

        if self.optimizer is not None and not is_wrapped(self.optimizer):
            self.optimizer = fabric.setup_optimizers(self.optimizer)

    def compute_loss(
        self,
        batch: torch.Tensor,  # Shape is (batch_size, seq_len)
        backpropagate: bool,  # Run backward pass if true, otherwise only compute loss
        no_sync: bool = False,  # If True, don't sync gradients across multiple GPUs
        loss_div: int = 1,  # Loss will be divided by this number
        **kwargs,  # Extra arguments ignored for compatibility with BST
    ) -> torch.Tensor:
        """
        Compute loss on a given batch of data. Optionally run backward pass.
        For gradient accumulation, call this function multiple times, iterating over each sub-batch.
        Loss will be divided by loss_div before backpropagation.
        Returns detached loss as a tensor.
        """
        self._assert_fabric_is_setup()

        # Predict next token for all tokens before the last token
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        if self.config.is_fim_mode:
            # Create FIM loss mask
            fim_mask = self._get_fim_loss_mask_fast(targets)
            # Set masked targets to ignore_index=-100
            targets = targets.masked_fill(fim_mask.logical_not(), -100)

        # If using FSDP, we ignore no_sync and always sync gradients
        # This allows us to re-shard each layer of the model after backward pass
        is_fsdp = isinstance(self.model.module, FSDPModule)
        with self.fabric.no_backward_sync(self.model, no_sync and not is_fsdp):
            loss = self.model(inputs, targets=targets)
            loss = loss / loss_div

            # Backward pass
            if backpropagate:
                self.fabric.backward(loss)

        return loss.detach()

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: Tuple[float, float],
        use_fused: Optional[bool] = None,
    ):
        """
        Create optimizer for training the model
        """
        # start with all of the candidate parameters
        # filter out those that do not require grad
        param_dict = {
            pn: p for pn, p in self.model.named_parameters() if p.requires_grad
        }

        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        # Create AdamW optimizer and use the fused version if it is available
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, eps=1e-8, fused=use_fused
        )

        self.optimizer = optimizer

    def optimizer_step(
        self,
        learning_rate: Optional[float] = None,  # If given, use this learning rate
        grad_clip: Optional[float] = None,  # If given, clip gradient norm
    ) -> Optional[torch.Tensor]:
        """
        Performs a single optimization step.
        If gradient clipping is enabled, return the gradient norm before clipping.
        """
        self._assert_fabric_is_setup()
        assert (
            self.optimizer is not None
        ), "Optimizer must be set up before calling this function"

        # Update learning rate
        if learning_rate is not None:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = learning_rate

        # Clip gradient
        grad_norm = None
        if grad_clip is not None:
            grad_norm = self.fabric.clip_gradients(
                self.model, self.optimizer, max_norm=grad_clip
            )

        # Step optimizer
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.training_steps += 1

        # Return gradient norm
        return grad_norm

    def compile(self):
        """
        Compiles the model using torch.compile()
        """
        # Must compile before setup_fabric()
        self._assert_fabric_is_setup(setup=False)
        self.model.compile()
        self._get_fim_loss_mask_fast = torch.compile(self._get_fim_loss_mask_fast)

    def train(self):
        """
        Set the model to training mode
        """
        self.model.train()

    def eval(self):
        """
        Set the model to evaluation mode
        """
        self.model.eval()

    def get_num_params(self, non_embedding: bool = True) -> int:
        return self.model.get_num_params(non_embedding)

    def _get_checkpoint_state(self) -> Dict[str, Any]:
        """
        Get the state of the model for checkpoint save/load
        """
        return {
            "model": self.model,
            "optimizer": self.optimizer,
            "training_steps": self.training_steps,
        }

    @classmethod
    def from_pretrained(cls, model_type: str, override_args: Optional[dict] = None):
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        override_args = override_args or {}  # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == "dropout" for k in override_args)
        from transformers import GPT2LMHeadModel

        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args["vocab_size"] = 50257  # always 50257 for GPT model checkpoints
        config_args["block_size"] = 1024  # always 1024 for GPT model checkpoints
        config_args["bias"] = True  # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if "dropout" in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args["dropout"] = override_args["dropout"]
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        gpt = GPT(config)
        sd = gpt.model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [
            k for k in sd_keys if not k.endswith(".attn.bias")
        ]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")
        ]  # ignore these, just a buffer
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.bias")
        ]  # same, just the mask (buffer)
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(
            sd_keys
        ), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.inference_mode():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.inference_mode():
                    sd[k].copy_(sd_hf[k])

        return gpt

    def crop_block_size(self, block_size: int):
        self.model.crop_block_size(block_size)

    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float) -> float:
        """estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS"""
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.inference_mode()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = (
                idx
                if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )
            # forward the model to get the logits for the index in the sequence
            logits = self.model(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    # Slow version is not actually used, but here for reference and debugging
    def _get_fim_loss_mask_slow(self, targets: torch.Tensor) -> torch.Tensor:
        """
        Create a loss mask to exclude the FIM suffix at the beginning of each document.
        Only compute loss for tokens after FIM and up to EOS.

        targets: tensor of shape (batch_size, seq_len)
        for targets in the form [..., [... <eos>, e, f, g, <|fim|>, a, b, c, d, <eos>, ...], ...]
        of shape (batch_size, seq_len)
        we want a mask      [..., [... 1,     0, 0, 0, 0,       1, 1, 1, 1, 1,     ...]]
        for selecting the tokens which we want to compute loss on
        """

        eos_id = self.config.eos_token_id
        fim_id = self.config.fim_token_id

        bz, seq_len = targets.shape
        mask = torch.zeros_like(targets, dtype=torch.bool)

        for i in range(bz):
            sequence = targets[i]

            start = None
            fim_found = False
            eos_found = False

            for j in range(seq_len):
                if sequence[j] == fim_id:
                    fim_found = True
                    start = j

                elif sequence[j] == eos_id:
                    if start:
                        mask[i, start + 1 : j + 1] = 1
                    else:
                        mask[i, : j + 1] = 1
                    start = j

                    if not eos_found:
                        # always mask out the first EOS
                        mask[i, j] = 0
                    eos_found = True

            if start is not None:
                mask[i, start + 1 :] = 1

            # edge case: if neither fim nor eos are in the sequence,
            # set the entire mask true (we default to next token prediction)
            if not fim_found and not eos_found:
                mask[i, :] = True

        return mask

    def _get_fim_loss_mask_fast(self, targets: torch.Tensor) -> torch.Tensor:
        """
        A vectorized version of get_fim_token_mask_slow

        This version doesn't support the edge cases due to not being
        able to implement them using only vectorized ops, but rather
        just treats each sequence as a slice of an infinitely long
        sequence of tokens with alternating <fim> and <eos>, so:
        - makes the assumption that eos and fim are always alternating
        - If there's only a single eos token, still treats the tokens
        after the eos as needing to be masked
        """

        eos_id = self.config.eos_token_id
        fim_id = self.config.fim_token_id

        eos_positions = targets == eos_id
        fim_positions = targets == fim_id

        eos_cumsum = torch.cumsum(eos_positions, dim=-1)
        fim_cumsum = torch.cumsum(fim_positions, dim=-1)
        eos_reverse_cumsum = (
            eos_positions
            + torch.sum(eos_positions, dim=-1, keepdims=True)
            - torch.cumsum(eos_positions, dim=-1)
        )
        fim_reverse_cumsum = (
            fim_positions
            + torch.sum(fim_positions, dim=-1, keepdims=True)
            - torch.cumsum(fim_positions, dim=-1)
        )

        fwd_diff = eos_cumsum - fim_cumsum
        rv_diff = eos_reverse_cumsum - fim_reverse_cumsum
        mask = (fwd_diff <= 0).logical_and(rv_diff >= 0)

        mask[fim_positions] = False
        mask[eos_positions] = True

        # always mask out the first EOS
        first_eos = eos_positions.logical_and(eos_cumsum == 1)
        mask[first_eos] = False

        return mask

    @torch.inference_mode()
    def evaluation_loss(
        self,
        batch: torch.Tensor,  # Shape is (batch_size, seq_len)
        prompt_len: torch.Tensor,  # Shape is (batch_size)
    ) -> torch.Tensor:
        """
        Compute next token prediction loss on the given batch of sequences.
        Prompt tokens are excluded from loss computation.
        """
        batch_size, seq_len = batch.size()

        # Predict next token for all tokens before the last token
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        # Mask out prompt tokens before last prompt token
        # Last prompt token predicts first output token, so we don't mask it
        target_indices = torch.arange(seq_len - 1, device=targets.device)
        last_prompt_index = (prompt_len - 1).unsqueeze(1)
        prompt_mask = target_indices.expand(batch_size, -1) < last_prompt_index
        targets = targets.masked_fill(prompt_mask, -100)

        # Forward pass
        logits = self.model(inputs)

        # Compute loss
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            reduction="none",
        ).view_as(targets)

        # Compute mean loss per sequence in batch
        num_target_tokens = (targets != -100).sum(dim=1)
        return loss.sum(dim=1) / num_target_tokens
