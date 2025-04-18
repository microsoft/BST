import argparse
import json
import os
import torch
import torch.nn.functional as F
import lightning as L
import yaml
from dataclasses import dataclass, field
from datetime import datetime
from omegaconf import OmegaConf
from queue import PriorityQueue
from tqdm import trange
from typing import List, Optional, Tuple, Union

from core_train import initialize_model
from data.tinystories import TinyStoriesDataModule
from model_bst import BST
from model_gpt import GPT


@dataclass
class Generation:
    # The inputs to generation
    prompt: str
    goal: str

    # Name of the sampling method used
    sampling_method: str

    # The top generated output and its score
    generated: str
    score: Optional[float] = None

    # Some sampling methods have a greedy generation
    greedy: Optional[str] = None
    greedy_score: Optional[float] = None

    # Some sampling methods return multiple outputs [(text, score)]
    top_n: List[Tuple[str, float]] = field(default_factory=list)

    # Original text from dataset used to construct prompt and goal
    original: Optional[str] = None

    def to_json_str(self, indent=None) -> str:
        return json.dumps(self, default=lambda o: o.__dict__, indent=indent)


class PartialGeneration:
    """
    Class to represent a partially generated sequence.
    This helps us sort candidate sequences by sum or mean log probability.
    """

    def __init__(
        self,
        sequence: torch.Tensor,  # shape (1, seq_len)
        generated_len: int,  # number of tokens generated, not including prompt
        sum_log_prob: float,  # sum of log probabilities of generated tokens
    ):
        assert sequence.ndim == 2, "Expected sequence to have shape (batch, seq_len)"
        assert sequence.size(0) == 1, "Expected sequence to have batch size of 1"
        self.sequence = sequence
        self.generated_len = generated_len
        self.sum_log_prob = sum_log_prob
        self.avg_log_prob = sum_log_prob / generated_len if generated_len > 0 else 0.0

    def __len__(self) -> int:
        # Length of the entire partial sequence, not just the generated part
        return self.sequence.size(1)


class CandidatesQueue(PriorityQueue):
    """
    Priority queue to store candidates for beam search or goal-conditioned planning.
    Depending on use_sum_log_prob, the queue will sort by sum or mean log probability.
    """

    def __init__(self, use_sum_log_prob: bool, maxsize: int = 0):
        super().__init__(maxsize)
        self._use_sum_log_prob = use_sum_log_prob
        self._count = 0

    def put(self, state: PartialGeneration):
        # Use negative of log prob for max heap
        # Add count for tie-breaking
        prob = state.sum_log_prob if self._use_sum_log_prob else state.avg_log_prob
        super().put((-prob, self._count, state))
        self._count += 1

    def get(self) -> PartialGeneration:
        # Get the sequence with the highest probability
        _, _, state = super().get()
        return state


class Inference:
    """
    Main class to perform inference with a trained model.
    """

    def __init__(
        self,
        fabric: L.Fabric,
        config,
        model: Union[BST, GPT],
        tokenizer,
        top_k: int = None,  # limit sampling to top K next tokens
        add_eos_token: bool = True,  # add EOS token to start of prompt and end of goal
        use_relative_probability: bool = False,  # use relative probability for goal_cond_plan
    ):
        self.fabric = fabric
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.top_k = top_k if top_k > 0 else None
        self.add_eos_token = add_eos_token
        self.use_relative_probability = use_relative_probability

        # Set model to eval mode
        self.model.eval()

    @torch.inference_mode()
    def generate_samples(
        self,
        batch: torch.Tensor,  # shape (batch_size, seq_len)
        sampling_mode: str,  # name of the sampling method
        prefix_len: int = 64,
        suffix_len: int = 64,
        rollout_count: int = 200,  # used for beam and goal_cond_plan
    ):
        """
        Main function to generate samples from the model.
        Since generation functions require batch size 1, we iterate over each sequence in the batch.
        prefix_len and suffix_len are used to split the sequence into prompt and goal.
        rollout_count is the number of rollouts to generate for beam search and goal-conditioned planning.
        """
        batch_size, seq_len = batch.shape
        batch = batch.to(self.fabric.device)
        eos_tensor = torch.tensor(
            [self.tokenizer.eos_token_id], device=self.fabric.device, dtype=batch.dtype
        )

        # Truncate seq_len to model context size
        seq_len = min(seq_len, self.model.config.block_size)
        rollout_len = seq_len - prefix_len - suffix_len
        if self.add_eos_token:
            rollout_len = rollout_len - 2  # prefix and suffix EOS

        # If sequence is too short for given prefix and suffix length, recalculate them
        if rollout_len < 1:
            # Set prefix and suffix lengths to 1/4 of the sequence length
            unit_len = seq_len // 4
            prefix_len = unit_len
            suffix_len = unit_len
            rollout_len = seq_len - prefix_len - suffix_len
            if self.add_eos_token:
                rollout_len = rollout_len - 2  # prefix and suffix EOS

        # Iterate over each sequence in the batch
        for i in range(batch_size):
            # Get prompt and goal for this sequence, without batch dimension
            # Shape is (prefix_len) and (suffix_len)
            prompt = batch[i, :prefix_len]
            goal = batch[i, -suffix_len:]

            # Sampling methods for GPT model
            if isinstance(model, GPT):
                if config.model.gpt_mode == "fim":
                    # For FIM mode, create new prompt with suffix and FIM token
                    fim_tensor = torch.tensor(
                        [model.config.fim_token_id],
                        device=self.fabric.device,
                        dtype=batch.dtype,
                    )
                    if self.add_eos_token:
                        prompt = torch.cat([eos_tensor, goal, fim_tensor, prompt])
                    else:
                        prompt = torch.cat([goal, fim_tensor, prompt])
                    goal = eos_tensor
                else:
                    # For next-token GPT, goal is ignored
                    if self.add_eos_token:
                        prompt = torch.cat([eos_tensor, prompt])
                    goal = eos_tensor

                # Call the desired sampling method
                if sampling_mode == "AR":
                    yield self.sample_autoregressive_gpt(prompt, goal, rollout_len)
                else:
                    raise ValueError(
                        f"GPT sampling mode {sampling_mode} not valid. Check trainer.sampling_mode in your config file."
                    )

            # Sampling methods for BST model
            elif isinstance(model, BST):
                # Add EOS to prompt and goal
                if self.add_eos_token:
                    prompt = torch.cat([eos_tensor, prompt])
                    goal = torch.cat([goal, eos_tensor])

                # Call the desired sampling method
                if sampling_mode == "AR":
                    yield self.sample_autoregressive(prompt, goal, rollout_len)
                elif sampling_mode == "beam":
                    yield self.sample_beamsearch(
                        prompt, goal, rollout_len, rollout_count
                    )
                elif sampling_mode == "goal_cond_plan":
                    yield self.sample_goal_cond_plan(
                        prompt, goal, rollout_len, rollout_count
                    )
                elif sampling_mode == "reverse_AR":
                    yield self.sample_reverse_autoregressive(prompt, goal, rollout_len)
                elif sampling_mode == "reverse_beam":
                    yield self.sample_reverse_beamsearch(
                        prompt, goal, rollout_len, rollout_count
                    )
                elif sampling_mode == "reverse_goal_cond_plan":
                    yield self.sample_reverse_goal_cond_plan(
                        prompt, goal, rollout_len, rollout_count
                    )
                else:
                    raise ValueError(
                        f"BST sampling mode {sampling_mode} not valid. Check trainer.sampling_mode in your config file."
                    )
            else:
                raise ValueError(
                    f"Model not valid. {type(model)} is not supported. Must be either BST or GPT."
                )

    def _untokenize(self, tokens: torch.Tensor) -> str:
        # Convert tokens to string using tokenizer
        return self.tokenizer.decode(
            tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).replace("##", "")

    @torch.inference_mode()
    def sample_autoregressive_gpt(
        self,
        prompt: torch.Tensor,  # shape (prompt_len)
        goal: torch.Tensor,  # shape (goal_len)
        rollout_len: int,
    ) -> Generation:
        assert prompt.ndim == 1
        assert goal.ndim == 1
        prompt_len = prompt.size(0)
        generated_len = 0

        # Add batch dimension, shape becomes (1, prompt_len)
        sequence = prompt.unsqueeze(0)

        while generated_len < rollout_len:
            # Shape of logits is (1, 1, vocab_size)
            next_token_logits = self.model.model(sequence)[0:1, -1:, :]
            next_token_probs = F.softmax(next_token_logits, dim=-1)
            # Shape of next_token is (1, 1)
            next_token = torch.argmax(next_token_probs, dim=-1, keepdim=False)
            # Shape of generated is (1, generated_len + 1)
            sequence = torch.cat((sequence, next_token), dim=1)
            generated_len += 1

            # End of sequence
            if next_token == self.tokenizer.eos_token_id:
                break

        return Generation(
            prompt=self._untokenize(prompt),
            goal=self._untokenize(goal),
            generated=self._untokenize(sequence[0, prompt_len:]),
            sampling_method="autoregressive_gpt",
        )

    @torch.inference_mode()
    def sample_autoregressive(
        self,
        prompt: torch.Tensor,  # shape (prompt_len)
        goal: torch.Tensor,  # shape (goal_len)
        rollout_len: int,
        use_goal: bool = True,
    ) -> Generation:
        assert prompt.ndim == 1
        assert goal.ndim == 1
        prompt_len = prompt.size(0)
        generated_len = 0

        # Add batch dimension, shape becomes (1, prompt_len)
        sequence = prompt.unsqueeze(0)

        # Compute backward embedding of goal
        # Output has shape (1, goal_len, emb_dim)
        _, bwd_emb = self.model.encoder(
            goal.unsqueeze(0), compute_forward=False, compute_backward=True
        )

        # Select the first or last token of the embedding depending on use_goal
        # Shape is (1, emb_dim)
        if use_goal:
            backward = bwd_emb[0, 0:1, :]
        else:
            backward = bwd_emb[0, -1:, :]

        while generated_len < rollout_len:
            # Compute forward embedding for currently generated tokens
            fwd_emb, _ = self.model.encoder(
                sequence, compute_forward=True, compute_backward=False
            )
            forward = fwd_emb[0, -1:, :]
            # Text head inputs have shape (1, emb_dim)
            # Logits has shape (1, 2, vocab_size)
            logits = self.model.text_head(forward, backward)
            next_token_logits = logits[:, 0, :]  # (1, vocab_size)
            next_token_probs = F.softmax(next_token_logits, dim=-1)
            # Shape of next_token is (1, 1)
            next_token = torch.argmax(next_token_probs, dim=-1, keepdim=True)
            # Shape of generated is (1, generated_len + 1)
            sequence = torch.cat((sequence, next_token), dim=1)
            generated_len += 1

            # End of sequence
            if next_token == self.tokenizer.eos_token_id:
                break

        return Generation(
            prompt=self._untokenize(prompt),
            goal=self._untokenize(goal),
            generated=self._untokenize(sequence[0, prompt_len:]),
            sampling_method="autoregressive",
        )

    @torch.inference_mode()
    def sample_reverse_autoregressive(
        self,
        prompt: torch.Tensor,  # shape (prompt_len)
        goal: torch.Tensor,  # shape (goal_len)
        rollout_len: int,
        use_prompt: bool = True,
    ) -> Generation:
        assert prompt.ndim == 1
        assert goal.ndim == 1
        goal_len = goal.size(0)
        generated_len = 0

        # Add batch dimension, shape becomes (1, goal_len)
        sequence = goal.unsqueeze(0)

        # Compute forward embedding of prompt
        # Output has shape (1, prompt_len, emb_dim)
        fwd_emb, _ = self.model.encoder(
            prompt.unsqueeze(0), compute_forward=True, compute_backward=False
        )

        # Select the first or last token of the embedding depending on use_prompt
        # Shape is (1, emb_dim)
        if use_prompt:
            forward = fwd_emb[0, -1:, :]
        else:
            forward = fwd_emb[0, 0:1, :]

        while generated_len < rollout_len:
            # Compute backward embedding for currently generated tokens
            _, bwd_emb = self.model.encoder(
                sequence, compute_forward=False, compute_backward=True
            )
            backward = bwd_emb[0, 0:1, :]
            # Text head inputs have shape (1, emb_dim)
            # Logits has shape (1, 2, vocab_size)
            logits = self.model.text_head(forward, backward)
            prev_token_logits = logits[:, 1, :]  # (1, vocab_size)
            prev_token_probs = F.softmax(prev_token_logits, dim=-1)
            # Shape of prev_token is (1, 1)
            prev_token = torch.argmax(prev_token_probs, dim=-1, keepdim=True)
            # Shape of generated is (1, generated_len + 1)
            sequence = torch.cat((prev_token, sequence), dim=1)
            generated_len += 1

            # End of sequence
            if prev_token == self.tokenizer.eos_token_id:
                break

        return Generation(
            prompt=self._untokenize(prompt),
            goal=self._untokenize(goal),
            generated=self._untokenize(sequence[0, :-goal_len]),
            sampling_method="reverse_autoregressive",
        )

    @torch.inference_mode()
    def sample_beamsearch(
        self,
        prompt: torch.Tensor,  # shape (prompt_len)
        goal: torch.Tensor,  # shape (goal_len)
        rollout_len: int,
        rollout_count: int = 2,
        use_goal: bool = True,
    ) -> Generation:
        assert prompt.ndim == 1
        assert goal.ndim == 1
        prompt_len = prompt.size(0)

        # Add batch dimension
        prompt = prompt.unsqueeze(0)
        goal = goal.unsqueeze(0)

        # Create priority queue for candidates
        # Beam search always uses mean log probability
        candidates = CandidatesQueue(use_sum_log_prob=False)
        candidates.put(
            PartialGeneration(
                sequence=prompt,
                generated_len=0,
                sum_log_prob=0,
            )
        )

        # Compute backward embedding of goal
        # Output has shape (batch, goal_len, emb_dim)
        _, bwd_emb = self.model.encoder(
            goal, compute_forward=False, compute_backward=True
        )

        # Select the first or last token of the embedding depending on use_goal
        # Shape is (batch, emb_dim)
        if use_goal:
            backward = bwd_emb[:, 0, :]
        else:
            backward = bwd_emb[:, -1, :]

        # Generate rollout_count sequences
        completed = []
        for _ in range(rollout_count):
            if not candidates.empty():
                partial_gen = candidates.get()
            else:
                # No more candidates in queue
                break

            completed_gen = self._generate_rollout_bst_forward(
                partial_gen=partial_gen,
                max_new_tokens=rollout_len - (len(partial_gen) - prompt_len),
                backward=backward,
                use_relative_probability=False,  # beam search always uses absolute
                queue=candidates,
            )
            completed.append((completed_gen.sequence, completed_gen.avg_log_prob))

        return self._create_generation_output(
            prompt=self._untokenize(prompt[0]),
            goal=self._untokenize(goal[0]),
            sampling_mode="beamsearch",
            prompt_len=prompt_len,
            goal_len=0,  # goal is not in the generated output
            seq_score=completed,
        )

    @torch.inference_mode()
    def sample_reverse_beamsearch(
        self,
        prompt: torch.Tensor,  # shape (prompt_len)
        goal: torch.Tensor,  # shape (goal_len)
        rollout_len: int,
        rollout_count: int = 2,
        use_prompt: bool = True,
    ) -> Generation:
        assert prompt.ndim == 1
        assert goal.ndim == 1
        goal_len = goal.size(0)

        # Add batch dimension
        prompt = prompt.unsqueeze(0)
        goal = goal.unsqueeze(0)

        # Create priority queue for candidates
        # Beam search always uses mean log probability
        candidates = CandidatesQueue(use_sum_log_prob=False)
        candidates.put(
            PartialGeneration(
                sequence=goal,
                generated_len=0,
                sum_log_prob=0,
            )
        )

        # Compute forward embedding of prompt
        # Output has shape (batch, prompt_len, emb_dim)
        fwd_emb, _ = self.model.encoder(
            prompt, compute_forward=True, compute_backward=False
        )

        # Select the first or last token of the embedding depending on use_prompt
        # Shape is (batch, emb_dim)
        if use_prompt:
            forward = fwd_emb[:, -1, :]
        else:
            forward = fwd_emb[:, 0, :]

        # Generate rollout_count sequences
        completed = []
        for _ in range(rollout_count):
            if not candidates.empty():
                partial_gen = candidates.get()
            else:
                # No more candidates in queue
                break

            completed_gen = self._generate_rollout_bst_reverse(
                partial_gen=partial_gen,
                max_new_tokens=rollout_len - (len(partial_gen) - goal_len),
                forward=forward,
                use_relative_probability=False,  # beam search always uses absolute
                queue=candidates,
            )
            completed.append((completed_gen.sequence, completed_gen.avg_log_prob))

        return self._create_generation_output(
            prompt=self._untokenize(prompt[0]),
            goal=self._untokenize(goal[0]),
            sampling_mode="reverse_beamsearch",
            prompt_len=0,  # prompt is not in the generated output
            goal_len=goal_len,
            seq_score=completed,
        )

    @torch.inference_mode()
    def sample_goal_cond_plan(
        self,
        prompt: torch.Tensor,  # shape (prompt_len)
        goal: torch.Tensor,  # shape (goal_len)
        rollout_len: int,
        rollout_count: int = 200,
        use_goal: bool = True,
    ) -> Generation:
        assert prompt.ndim == 1
        assert goal.ndim == 1
        prompt_len = prompt.size(0)
        goal_len = goal.size(0)

        # Add batch dimension
        prompt = prompt.unsqueeze(0)
        goal = goal.unsqueeze(0)

        # Create priority queue for candidates
        # Use sum for relative probability and mean for absolute probability
        candidates = CandidatesQueue(use_sum_log_prob=self.use_relative_probability)
        candidates.put(
            PartialGeneration(
                sequence=prompt,
                generated_len=0,
                sum_log_prob=0,
            )
        )

        # Compute backward embedding of goal
        # Output has shape (batch, goal_len, emb_dim)
        _, bwd_emb = self.model.encoder(
            goal, compute_forward=False, compute_backward=True
        )

        # Select the first or last token of the embedding depending on use_goal
        # Shape is (batch, emb_dim)
        if use_goal:
            backward = bwd_emb[:, 0, :]
        else:
            backward = bwd_emb[:, -1, :]

        # Generate sequences
        completed = []
        for _ in range(rollout_count):
            if not candidates.empty():
                partial_gen = candidates.get()
            else:
                # No more candidates in queue
                break

            completed_gen = self._generate_rollout_bst_forward(
                partial_gen=partial_gen,
                max_new_tokens=rollout_len - (len(partial_gen) - prompt_len),
                backward=backward,
                use_relative_probability=self.use_relative_probability,
                queue=candidates,
            )
            completed.append(completed_gen.sequence)

        # Score the completed sequences
        scores = []
        for idx in range(len(completed)):
            sequence = completed[idx]
            prompt_infill_len = sequence.size(1)

            # Add back the goal tokens
            sequence = torch.cat([sequence, goal], dim=1)
            full_seq_len = sequence.size(1)
            completed[idx] = sequence

            # Compute forward and backward embeddings
            fwd_emb, bwd_emb = self.model.encoder(sequence)

            # Get embeddings for (t, t+2) pairs with middle token in the goal region
            target_indices = torch.arange(
                prompt_infill_len, full_seq_len - 1, device=sequence.device
            )
            forward = fwd_emb[0, target_indices - 1]
            backward = bwd_emb[0, target_indices + 1]

            # Get logits for next and previous token predictions
            next_prev_logits = self.model.text_head(forward, backward)

            # If goal-conditioned, use next-head. Otherwise, use prev-head.
            if use_goal:
                head_logits = next_prev_logits[:, 0]
            else:
                head_logits = next_prev_logits[:, 1]
            head_logprobs = F.log_softmax(head_logits, dim=-1)

            # Target tokens to predict are in the middle of the (t, t+2) gap
            targets = sequence[0, target_indices]
            target_logprobs = head_logprobs[
                torch.arange(head_logprobs.size(0)), targets
            ]
            score = target_logprobs.mean().item()
            scores.append(score)

        return self._create_generation_output(
            prompt=self._untokenize(prompt[0]),
            goal=self._untokenize(goal[0]),
            sampling_mode="goal_cond_plan",
            prompt_len=prompt_len,
            goal_len=goal_len,
            seq_score=list(zip(completed, scores)),
        )

    @torch.inference_mode()
    def sample_reverse_goal_cond_plan(
        self,
        prompt: torch.Tensor,  # shape (prompt_len)
        goal: torch.Tensor,  # shape (goal_len)
        rollout_len: int,
        rollout_count: int = 200,
        use_prompt: bool = True,
    ) -> Generation:
        assert prompt.ndim == 1
        assert goal.ndim == 1
        prompt_len = prompt.size(0)
        goal_len = goal.size(0)

        # Add batch dimension
        prompt = prompt.unsqueeze(0)
        goal = goal.unsqueeze(0)

        # Create priority queue for candidates
        # Use sum for relative probability and mean for absolute probability
        candidates = CandidatesQueue(use_sum_log_prob=self.use_relative_probability)
        candidates.put(
            PartialGeneration(
                sequence=goal,
                generated_len=0,
                sum_log_prob=0,
            )
        )

        # Compute forward embedding of prompt
        # Output has shape (batch, prompt_len, emb_dim)
        fwd_emb, _ = self.model.encoder(
            goal, compute_forward=True, compute_backward=False
        )

        # Select the first or last token of the embedding depending on use_prompt
        # Shape is (batch, emb_dim)
        if use_prompt:
            forward = fwd_emb[:, -1, :]
        else:
            forward = fwd_emb[:, 0, :]

        # Generate sequences
        completed = []
        for _ in range(rollout_count):
            if not candidates.empty():
                partial_gen = candidates.get()
            else:
                # No more candidates in queue
                break

            completed_gen = self._generate_rollout_bst_reverse(
                partial_gen=partial_gen,
                max_new_tokens=rollout_len - (len(partial_gen) - goal_len),
                forward=forward,
                use_relative_probability=self.use_relative_probability,
                queue=candidates,
            )
            completed.append(completed_gen.sequence)

        # Score the completed sequences
        scores = []
        for idx in range(len(completed)):
            sequence = completed[idx]

            # Add back the prompt tokens
            sequence = torch.cat([prompt, sequence], dim=1)
            completed[idx] = sequence

            # Compute forward and backward embeddings
            fwd_emb, bwd_emb = self.model.encoder(sequence)

            # Get embeddings for (t, t+2) pairs with middle token in the prompt region
            target_indices = torch.arange(1, prompt_len, device=sequence.device)
            forward = fwd_emb[0, target_indices - 1]
            backward = bwd_emb[0, target_indices + 1]

            # Get logits for next and previous token predictions
            next_prev_logits = self.model.text_head(forward, backward)

            # If goal-conditioned, use prev-head. Otherwise, use next-head.
            if use_prompt:
                head_logits = next_prev_logits[:, 1]
            else:
                head_logits = next_prev_logits[:, 0]
            head_logprobs = F.log_softmax(head_logits, dim=-1)

            # Target tokens to predict are in the middle of the (t, t+2) gap
            targets = sequence[0, target_indices]
            target_logprobs = head_logprobs[
                torch.arange(head_logprobs.size(0)), targets
            ]
            score = target_logprobs.mean().item()
            scores.append(score)

        return self._create_generation_output(
            prompt=self._untokenize(prompt[0]),
            goal=self._untokenize(goal[0]),
            sampling_mode="reverse_goal_cond_plan",
            prompt_len=prompt_len,
            goal_len=goal_len,
            seq_score=list(zip(completed, scores)),
        )

    def _generate_rollout_bst_forward(
        self,
        partial_gen: PartialGeneration,
        max_new_tokens: int,  # Number of additional tokens to generate
        backward: torch.Tensor,  # Backward encoder's embedding for the goal
        use_relative_probability: bool,  # Score with relative or absolute probability
        queue: Optional[CandidatesQueue] = None,  # Queue to add new candidates
    ) -> PartialGeneration:
        """
        Generate a sequence using the BST model with forward sampling.
        Shared for beam search and goal-conditioned planning.
        """
        # Get current state of partial generation
        sequence = partial_gen.sequence
        generated_len = partial_gen.generated_len
        sum_log_prob = partial_gen.sum_log_prob

        assert isinstance(self.model, BST), "BST model required for this function"
        assert sequence.ndim == 2, "Expected sequence to have shape (batch, seq_len)"
        assert sequence.size(0) == 1, "Expected sequence to have batch size of 1"
        assert backward.ndim == 2, "Expected backward to have shape (batch, emb_dim)"
        assert backward.size(0) == 1, "Expected backward to have batch size of 1"

        for _ in range(max_new_tokens):
            # Compute forward embedding for currently generated tokens
            fwd_emb, _ = self.model.encoder(
                sequence, compute_forward=True, compute_backward=False
            )
            forward = fwd_emb[:, -1, :]

            # Text head inputs have shape (1, emb_dim)
            # Logits has shape (1, 2, vocab_size)
            logits = self.model.text_head(forward, backward)
            vocab_size = logits.size(-1)
            next_token_logits = logits[:, 0, :]  # (1, vocab_size)
            next_token_logprobs = F.log_softmax(next_token_logits, dim=-1)

            # Get top token and its probability
            # Shape is (1)
            best_token_logprob, best_token = torch.max(next_token_logprobs, dim=-1)

            # Generate sequences with alternative tokens and add to queue
            if queue is not None:
                # Get reward for each token in vocabulary
                # Shape is (1, vocab_size)
                if use_relative_probability:
                    all_next_rewards = next_token_logprobs - best_token_logprob
                else:
                    all_next_rewards = next_token_logprobs

                # Get list of candidate tokens
                if self.top_k is not None:
                    next_tokens = (
                        torch.topk(next_token_logprobs.squeeze(0), k=self.top_k)
                        .indices.cpu()
                        .tolist()
                    )
                else:
                    next_tokens = range(vocab_size)

                # Iterate over all candidate tokens
                best_token_cpu = best_token.item()
                for new_token in next_tokens:
                    # Skip best token and EOS token
                    if (
                        new_token == best_token_cpu
                        or new_token == self.tokenizer.eos_token_id
                    ):
                        continue

                    # Append new token to sequence
                    new_token_tensor = torch.tensor(
                        [new_token], dtype=sequence.dtype, device=sequence.device
                    ).unsqueeze(0)
                    new_sequence = torch.cat([sequence, new_token_tensor], dim=1)
                    new_reward = all_next_rewards[0, new_token].item()

                    # Add to queue
                    queue.put(
                        PartialGeneration(
                            sequence=new_sequence,
                            generated_len=generated_len + 1,
                            sum_log_prob=sum_log_prob + new_reward,
                        )
                    )

            # If the best token is EOS, stop generation
            # Don't append EOS to the output sequence
            if best_token == self.tokenizer.eos_token_id:
                break

            # Add best token to sequence
            sequence = torch.cat([sequence, best_token.unsqueeze(0)], dim=-1)
            generated_len += 1
            # Update reward with the best token probability
            # Only needed for absolute probability, since relative logprob of best token is 0
            if not use_relative_probability:
                sum_log_prob = sum_log_prob + best_token_logprob.item()

        # Wrap completed sequence in PartialGeneration object
        return PartialGeneration(
            sequence=sequence,
            generated_len=generated_len,
            sum_log_prob=sum_log_prob,
        )

    def _generate_rollout_bst_reverse(
        self,
        partial_gen: PartialGeneration,
        max_new_tokens: int,  # Number of additional tokens to generate
        forward: torch.Tensor,  # Forward encoder's embedding for the prompt
        use_relative_probability: bool,  # Score with relative or absolute probability
        queue: Optional[CandidatesQueue] = None,  # Queue to add new candidates
    ) -> PartialGeneration:
        """
        Generate a sequence using the BST model with reverse sampling.
        Shared for beam search and goal-conditioned planning.
        """
        # Get current state of partial generation
        sequence = partial_gen.sequence
        generated_len = partial_gen.generated_len
        sum_log_prob = partial_gen.sum_log_prob

        assert isinstance(self.model, BST), "BST model required for this function"
        assert sequence.ndim == 2, "Expected sequence to have shape (batch, seq_len)"
        assert sequence.size(0) == 1, "Expected sequence to have batch size of 1"
        assert forward.ndim == 2, "Expected backward to have shape (batch, emb_dim)"
        assert forward.size(0) == 1, "Expected backward to have batch size of 1"

        for _ in range(max_new_tokens):
            # Compute forward embedding for currently generated tokens
            _, bwd_emb = self.model.encoder(
                sequence, compute_forward=False, compute_backward=True
            )
            backward = bwd_emb[:, 0, :]

            # Text head inputs have shape (1, emb_dim)
            # Logits has shape (1, 2, vocab_size)
            logits = self.model.text_head(forward, backward)
            vocab_size = logits.size(-1)
            prev_token_logits = logits[:, 1, :]  # (1, vocab_size)
            prev_token_logprobs = F.log_softmax(prev_token_logits, dim=-1)

            # Get top token and its probability
            # Shape is (1)
            best_token_logprob, best_token = torch.max(prev_token_logprobs, dim=-1)

            # Generate sequences with alternative tokens and add to queue
            if queue is not None:
                # Get reward for each token in vocabulary
                # Shape is (1, vocab_size)
                if use_relative_probability:
                    all_prev_rewards = prev_token_logprobs - best_token_logprob
                else:
                    all_prev_rewards = prev_token_logprobs

                # Get list of candidate tokens
                if self.top_k is not None:
                    prev_tokens = (
                        torch.topk(prev_token_logprobs.squeeze(0), k=self.top_k)
                        .indices.cpu()
                        .tolist()
                    )
                else:
                    prev_tokens = range(vocab_size)

                # Iterate over all candidate tokens
                best_token_cpu = best_token.item()
                for new_token in prev_tokens:
                    # Skip best token and EOS token
                    if (
                        new_token == best_token_cpu
                        or new_token == self.tokenizer.eos_token_id
                    ):
                        continue

                    # Append new token to sequence
                    new_token_tensor = torch.tensor(
                        [new_token], dtype=sequence.dtype, device=sequence.device
                    ).unsqueeze(0)
                    new_sequence = torch.cat([new_token_tensor, sequence], dim=1)
                    new_reward = all_prev_rewards[0, new_token].item()

                    # Add to queue
                    queue.put(
                        PartialGeneration(
                            sequence=new_sequence,
                            generated_len=generated_len + 1,
                            sum_log_prob=sum_log_prob + new_reward,
                        )
                    )

            # If the best token is EOS, stop generation
            # Don't append EOS to the output sequence
            if best_token == self.tokenizer.eos_token_id:
                break

            # Add best token to sequence
            sequence = torch.cat([best_token.unsqueeze(0), sequence], dim=-1)
            generated_len += 1
            # Update reward with the best token probability
            # Only needed for absolute probability, since relative logprob of best token is 0
            if not use_relative_probability:
                sum_log_prob = sum_log_prob + best_token_logprob.item()

        # Wrap completed sequence in PartialGeneration object
        return PartialGeneration(
            sequence=sequence,
            generated_len=generated_len,
            sum_log_prob=sum_log_prob,
        )

    def _remove_prompt_and_goal(
        self,
        sequence: torch.Tensor,  # shape (batch, seq_len)
        prompt_len: int,
        goal_len: int,
    ) -> torch.Tensor:
        assert sequence.ndim == 2, "Expected sequence to have shape (batch, seq_len)"
        assert sequence.size(0) == 1, "Expected sequence to have batch size of 1"
        # Remove prompt and goal from the sequence
        if prompt_len > 0:
            sequence = sequence[:, prompt_len:]
        if goal_len > 0:
            sequence = sequence[:, :-goal_len]
        return sequence[0]  # Remove batch dimension

    def _create_generation_output(
        self,
        prompt: str,
        goal: str,
        sampling_mode: str,
        prompt_len: int,
        goal_len: int,
        seq_score: List[Tuple[torch.Tensor, float]],
    ) -> Generation:
        """
        Helper function to create a Generation object from the generated sequences and their scores.
        The input seq_score list is assumed to be unsorted, and seq_score[0] is the greedy sequence.
        """
        # Input sequences have not yet been sorted
        greedy_seq = self._remove_prompt_and_goal(seq_score[0][0], prompt_len, goal_len)
        greedy_score = seq_score[0][1]

        # Sort sequences by score
        seq_score = sorted(seq_score, reverse=True, key=lambda x: x[1])
        best_seq = self._remove_prompt_and_goal(seq_score[0][0], prompt_len, goal_len)
        best_score = seq_score[0][1]

        # Create generation object
        generation = Generation(
            prompt=prompt,
            goal=goal,
            sampling_method=sampling_mode,
            generated=self._untokenize(best_seq),
            score=best_score,
            greedy=self._untokenize(greedy_seq),
            greedy_score=greedy_score,
        )
        for seq, score in seq_score:
            generation.top_n.append(
                (
                    self._untokenize(
                        self._remove_prompt_and_goal(seq, prompt_len, goal_len)
                    ),
                    score,
                )
            )
        return generation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load a model checkpoint")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        required=False,
        help="Number of samples to generate for inference",
    )
    parser.add_argument(
        "--checkpoint_path", type=str, required=True, help="Path to the checkpoint file"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="",
        required=False,
        help="Path to write output - if missing only stout will be used",
    )
    parser.add_argument(
        "--sampling_mode",
        type=str,
        required=True,
        choices=(
            "AR",
            "beam",
            "goal_cond_plan",
            "reverse_AR",
            "reverse_beam",
            "reverse_goal_cond_plan",
        ),
    )
    parser.add_argument(
        "--rollout_count",
        type=int,
        default=10,
        help="Number of rollouts for beam search and goal conditioned sampling",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of top-k tokens to sample from the model",
    )
    parser.add_argument(
        "--relative_prob",
        action="store_true",
        default=False,
        help="Use relative probability for goal_cond_plan algorithms",
    )

    parser.add_argument("-c", "--config", required=True)
    args, conf_cli = parser.parse_known_args()

    default = OmegaConf.load("defaults.yaml")
    overrides = OmegaConf.load(args.config)
    cli = OmegaConf.from_dotlist(conf_cli)
    config = OmegaConf.merge(default, overrides, cli)

    # Initialize fabric
    fabric = L.Fabric()

    # Calculate per device batch size
    assert (
        config.data.effective_batch_size % fabric.world_size == 0
    ), f"effective_batch_size {config.data.effective_batch_size} must be divisible by fabric.world_size {fabric.world_size}"
    config.data.device_batch_size = (
        config.data.effective_batch_size // fabric.world_size
    )

    print(yaml.dump(OmegaConf.to_container(config)))

    # Load data
    datamodule = TinyStoriesDataModule(fabric, config)
    datamodule.update_config(config)

    eval_dataloader = datamodule.eval_dataloader()
    tokenizer = datamodule.get_tokenizer()

    # Load model from checkpoint
    checkpoint_path = args.checkpoint_path
    assert os.path.isfile(
        checkpoint_path
    ), f"Checkpoint file {checkpoint_path} not found"
    model = initialize_model(
        fabric,
        config,
        tokenizer,
        initialize_optimizer=False,
        checkpoint_path=checkpoint_path,
    )

    # Create inference object
    inference = Inference(
        fabric,
        config,
        model,
        tokenizer,
        top_k=args.top_k,
        add_eos_token=True,
        use_relative_probability=args.relative_prob,
    )

    output_file = None
    if args.output_path:
        isfile = os.path.isfile(args.output_path)
        output_file = open(args.output_path, "a")
        if isfile:
            print("output file exists: adding break line and appending")
            output_file.write(
                "--------------------- NEW SESSION --------------------\n"
            )

    samples_generated = 0
    for i, batch in enumerate(eval_dataloader):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"Batch {i} - {now}")

        text = batch["text"]
        batch["input_ids"] = tokenizer(
            text, return_tensors="pt", max_length=256, padding=False, truncation=True
        )["input_ids"]
        batch = datamodule.prepare_batch(batch)

        for generation_result in inference.generate_samples(
            batch, sampling_mode=args.sampling_mode, rollout_count=args.rollout_count
        ):
            generation_result.original = text[0]

            if output_file is not None:
                # Write to file
                generation_str = generation_result.to_json_str()
                output_file.write(generation_str + "\n")
            else:
                # Print to stdout
                print("--------------- STORY ------------------")
                generation_str = generation_result.to_json_str(indent=2)
                print(generation_str)

            samples_generated += 1
            print(f"Samples generated: {samples_generated}")

            # inner loop break
            if samples_generated >= args.num_samples:
                break

        # Break outer loop if we have generated enough samples
        if samples_generated >= args.num_samples:
            break

    if output_file is not None:
        output_file.close()
