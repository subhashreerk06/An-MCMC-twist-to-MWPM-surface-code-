"""Conditional autoregressive Bernoulli model with exact normalized density.

q(z | s) = product_j Bernoulli(z_j; p_j(z[:j], s)). Fixed masks enforce
ascending coordinate order; syndrome bits are unrestricted context. The same
bounded probabilities are used for sampling and log_prob, without a partition
function or a continuous relaxation. Full support follows from eps > 0 and
p_j in [eps, 1-eps]. Probabilities and log probabilities use float64 to keep
these bounds representable, even for saturated network logits.

Inputs are binary vectors or batches. log_prob broadcasts one syndrome over
a batch of z vectors. Training can minimize -model.log_prob(z, s).mean().
Syndrome validity is handled externally by the affine coordinate mapping;
this module only models unconstrained binary coordinates and syndrome context.
"""

import math
from operator import index
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def _dimension(value, name: str, minimum: int = 0) -> int:
    try:
        if isinstance(value, (bool, np.bool_)) or index(value) < minimum:
            raise ValueError
        return int(index(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer >= {minimum}.") from exc


def _fingerprints(value: dict) -> dict:
    required = ("edge_ordering_fingerprint", "h_nullspace_fingerprint")
    if not isinstance(value, dict) or any(key not in value for key in required):
        raise ValueError(f"Fingerprints must include {required}.")
    result = {key: value[key] for key in required}
    if any(
        not isinstance(v, str) or len(v) != 64 or any(c not in "0123456789abcdef" for c in v)
        for v in result.values()
    ):
        raise ValueError("Fingerprints must be lowercase SHA-256 hex strings.")
    return result


class ConditionalAutoregressiveBernoulli(nn.Module):
    """Masked nonlinear autoregression plus direct prefix/context connections.

    sample(s, rng) accepts a NumPy Generator or a torch.Generator (on the
    model's device). It returns uint8 tensors shaped (*s.shape[:-1], z_dim).
    forward(z, s) returns conditional bit probabilities of that same shape;
    log_prob(z, s) sums their Bernoulli log probabilities over the last axis.
    Single inputs return a single vector/probability scalar; batches remain
    batches. There is no dropout or batch-dependent normalization.
    """

    def __init__(self, syndrome_dim: int, z_dim: int, hidden_dim: int = 128, eps: float = 1e-6):
        super().__init__()
        self.syndrome_dim = _dimension(syndrome_dim, "syndrome_dim")
        self.z_dim = _dimension(z_dim, "z_dim")
        self.hidden_dim = _dimension(hidden_dim, "hidden_dim", 1)
        self.eps = float(eps)
        if not math.isfinite(self.eps) or not torch.finfo(torch.float64).eps <= self.eps < 0.5:
            raise ValueError("eps must be >= float64 machine epsilon and < 0.5.")

        self.prefix_hidden = nn.Linear(self.z_dim, self.hidden_dim, bias=False)
        self.context_hidden = nn.Linear(self.syndrome_dim, self.hidden_dim)
        self.hidden_output = nn.Linear(self.hidden_dim, self.z_dim)
        self.prefix_output = nn.Linear(self.z_dim, self.z_dim, bias=False)
        self.context_output = nn.Linear(self.syndrome_dim, self.z_dim, bias=False)
        bits = torch.arange(self.z_dim)
        degrees = torch.arange(self.hidden_dim) % max(1, self.z_dim)
        # Hidden degree d sees only z indices < d; output j sees degrees <= j.
        # Masks are architecture-derived, not trainable or checkpoint-supplied.
        self.register_buffer("input_mask", bits[None, :] < degrees[:, None], persistent=False)
        self.register_buffer("output_mask", degrees[None, :] <= bits[:, None], persistent=False)
        self.register_buffer("prefix_mask", bits[None, :] < bits[:, None], persistent=False)

    def _binary(self, value, width: int, name: str) -> torch.Tensor:
        reference = self.context_hidden.weight
        value = torch.as_tensor(value, device=reference.device)
        if value.ndim not in (1, 2) or value.shape[-1] != width:
            raise ValueError(f"{name} must have shape ({width},) or (batch, {width}).")
        if value.is_complex() or not bool(torch.all((value == 0) | (value == 1))):
            raise ValueError(f"{name} must contain only binary entries.")
        return value.to(dtype=reference.dtype)

    def _probabilities(self, z: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        hidden = torch.tanh(
            F.linear(z, self.prefix_hidden.weight * self.input_mask) + self.context_hidden(s)
        )
        logits = (
            F.linear(hidden, self.hidden_output.weight * self.output_mask, self.hidden_output.bias)
            + F.linear(z, self.prefix_output.weight * self.prefix_mask)
            + self.context_output(s)
        )
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("Model logits must be finite.")
        return torch.sigmoid(logits.to(torch.float64)).clamp(self.eps, 1.0 - self.eps)

    def forward(self, z, s) -> torch.Tensor:
        z = self._binary(z, self.z_dim, "z")
        s = self._binary(s, self.syndrome_dim, "s")
        try:
            batch = torch.broadcast_shapes(z.shape[:-1], s.shape[:-1])
        except RuntimeError as exc:
            raise ValueError("z and s batch sizes must match or broadcast.") from exc
        return self._probabilities(z.expand(*batch, self.z_dim), s.expand(*batch, self.syndrome_dim))

    def log_prob(self, z, s) -> torch.Tensor:
        """Return exact log q(z | s), using the bounded model probabilities."""
        probabilities = self(z, s)
        bits = self._binary(z, self.z_dim, "z").bool()
        return torch.where(bits, probabilities.log(), torch.log1p(-probabilities)).sum(dim=-1)

    @torch.no_grad()
    def sample(self, s, rng: np.random.Generator | torch.Generator) -> torch.Tensor:
        """Draw in deterministic z-bit order, consuming only the supplied RNG."""
        if not isinstance(rng, (np.random.Generator, torch.Generator)):
            raise TypeError("rng must be a NumPy Generator or torch.Generator.")
        syndrome = self._binary(s, self.syndrome_dim, "s")
        z = syndrome.new_zeros((*syndrome.shape[:-1], self.z_dim))
        for j in range(self.z_dim):
            probability = self._probabilities(z, syndrome)[..., j]
            if isinstance(rng, np.random.Generator):
                uniform = torch.as_tensor(rng.random(tuple(probability.shape)), device=z.device)
            else:
                uniform = torch.rand(
                    probability.shape, generator=rng, device=z.device, dtype=torch.float64,
                )
            z[..., j] = (uniform < probability).to(z.dtype)
        return z.to(torch.uint8)

    def save_checkpoint(self, path: str | Path, *, fingerprints: dict) -> None:
        """Save architecture, parameters, dtype, and dataset graph/basis identity.

        fingerprints may be the complete flow_dataset metadata dictionary;
        the edge ordering and combined H/nullspace hashes are retained.
        """
        fingerprints = _fingerprints(fingerprints)
        dtype = self.context_hidden.weight.dtype
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("Checkpoints support float32 or float64 model parameters.")
        torch.save({
            "format_version": 1,
            "architecture": {
                "name": "conditional_autoregressive_bernoulli",
                "syndrome_dim": self.syndrome_dim, "z_dim": self.z_dim,
                "hidden_dim": self.hidden_dim, "eps": self.eps,
                "bit_order": "ascending_free_column", "dtype": str(dtype).split(".")[-1],
            },
            "fingerprints": fingerprints, "state_dict": self.state_dict(),
        }, path)

    @classmethod
    def load_checkpoint(cls, path: str | Path, *, expected_fingerprints: dict, device="cpu"):
        """Load parameters only after verifying the expected graph/basis identity."""
        expected = _fingerprints(expected_fingerprints)
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        if checkpoint.get("format_version") != 1:
            raise ValueError("Unsupported checkpoint format.")
        if _fingerprints(checkpoint["fingerprints"]) != expected:
            raise ValueError("Checkpoint graph/basis fingerprints do not match.")
        architecture = checkpoint["architecture"].copy()
        if architecture.pop("name") != "conditional_autoregressive_bernoulli":
            raise ValueError("Unsupported checkpoint architecture.")
        if architecture.pop("bit_order") != "ascending_free_column":
            raise ValueError("Unsupported coordinate ordering.")
        dtype = architecture.pop("dtype")
        if dtype not in ("float32", "float64"):
            raise ValueError("Unsupported parameter dtype.")
        model = cls(**architecture).to(device=device, dtype=getattr(torch, dtype))
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval()
        return model
