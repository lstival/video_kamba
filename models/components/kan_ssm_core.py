"""KAN-Modulated State Space Model Core.

This module implements the intricate fusion of KAN (Kolmogorov-Arnold Networks)
with Mamba's SSM (State Space Model) by using learned B-spline polynomial
coefficients to dynamically modulate the B and C matrices.

Key Innovation:
    Instead of just replacing MLPs with KAN layers (shallow integration),
    we use KAN's learned polynomial coefficients to guide HOW inputs affect
    the hidden state (B matrix) and HOW states produce outputs (C matrix).

Mathematical Formulation:
    Standard SSM:
        h_t = Ā·h_{t-1} + B̄·u_t
        y_t = C·h_t

    KAN-Modulated SSM:
        h_t = Ā·h_{t-1} + (B̄ ⊙ α(u_t))·u_t
        y_t = (C ⊙ β(h_t))·h_t

    Where:
        α(u_t) = σ(Σ c_i^(B) · N_{i,k}(u_t))  # Learnable B-modulation
        β(h_t) = σ(Σ c_i^(C) · N_{i,k}(h_t))  # Learnable C-modulation
        N_{i,k}(·) are Cox-de Boor B-spline basis functions

References:
    - Gu & Dao (2023): Mamba: Linear-Time Sequence Modeling with Selective SSMs
    - Liu et al. (2024): KAN: Kolmogorov-Arnold Networks
    - Cox-de Boor recursion for B-spline evaluation

Author: KANGA Project
"""

from __future__ import annotations

import math
from typing import Optional, Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float

from .fast_kan_layer import FastKANLayer

try:
    import mamba_ssm
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    HAS_MAMBA_KERNELS = True
    print("Mamba SSM kernels found! 🎉")
except ImportError:
    HAS_MAMBA_KERNELS = False
    print("Mamba SSM kernels NOT found. Using slow Python fallback.")



class KANModulator(nn.Module):
    """Learnable B-spline based modulator for SSM matrices.
    
    This module computes input-dependent modulation factors using
    B-spline basis functions with learned coefficients.
    
    Args:
        input_dim: Dimension of the input features.
        output_dim: Dimension of the modulation output.
        grid_size: Number of grid intervals for B-splines (default: 5).
        spline_order: Order of B-spline basis (default: 3, cubic).
        grid_range: Range of the grid (default: [-1, 1]).
        activation: Activation to ensure positive modulation ('softplus', 'sigmoid', 'exp').
    
    Shape:
        - Input: ``[B, D_in]``
        - Output: ``[B, D_out]``
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        grid_size: int = 5,
        spline_order: int = 3,
        grid_range: Tuple[float, float] = (-1.0, 1.0),
        activation: Literal["softplus", "sigmoid", "exp"] = "softplus",
    ) -> None:
        super().__init__()
        
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.n_bases = grid_size + spline_order
        self.activation_type = activation
        
        # Compute grid points with boundary extensions
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            torch.arange(-spline_order, grid_size + spline_order + 1, dtype=torch.float32) * h
            + grid_range[0]
        )
        self.register_buffer("grid", grid)
        
        # Learnable spline coefficients for modulation
        # Shape: [output_dim, input_dim, n_bases]
        # These coefficients define the learned polynomial relationship
        self.coefficients = nn.Parameter(
            torch.empty(output_dim, input_dim, self.n_bases)
        )
        
        # Bias term for modulation
        self.bias = nn.Parameter(torch.zeros(output_dim))
        
        # Scale parameter for numerical stability
        self.log_scale = nn.Parameter(torch.zeros(output_dim))
        
        self._init_weights()
        
    def _init_weights(self) -> None:
        """Initialize coefficients to produce near-identity modulation."""
        # Initialize to EXACT zero so initial modulation depends only on bias
        nn.init.zeros_(self.coefficients)
        
        # Initialize bias such that activation(bias) ≈ 1.0
        # For softplus: softplus(x) + 0.1 = 1.0 => x ≈ 0.378
        if self.activation_type == "softplus":
             nn.init.constant_(self.bias, 0.38)
        elif self.activation_type == "sigmoid":
             # sigmoid(x) + 0.5 = 1.0 => sigmoid(x) = 0.5 => x = 0
             nn.init.zeros_(self.bias)
        elif self.activation_type == "exp":
             # exp(x) = 1.0 => x = 0
             nn.init.zeros_(self.bias)
        else:
             nn.init.zeros_(self.bias)
             
        nn.init.zeros_(self.log_scale)
        
    def _b_splines(
        self, x: Float[torch.Tensor, "B D_in"]
    ) -> Float[torch.Tensor, "B D_in n_bases"]:
        """Compute B-spline basis values using Cox-de Boor recursion.
        
        Args:
            x: Input tensor of shape [batch, input_dim].
            
        Returns:
            B-spline basis values of shape [batch, input_dim, n_bases].
        """
        grid = self.grid  # [grid_size + 2 * spline_order + 1]
        x = x.unsqueeze(-1)  # [B, D_in, 1]
        
        # Order 0: characteristic functions
        bases = ((x >= grid[:-1]) & (x < grid[1:])).float()
        
        # Build up to desired order via Cox-de Boor recursion
        for k in range(1, self.spline_order + 1):
            left_num = x - grid[: -(k + 1)]
            left_den = grid[k:-1] - grid[: -(k + 1)]
            right_num = grid[k + 1:] - x
            right_den = grid[k + 1:] - grid[1:(-k)]
            
            # Handle division by zero gracefully
            left_term = torch.where(
                left_den.abs() > 1e-8,
                left_num / left_den * bases[..., :-1],
                torch.zeros_like(bases[..., :-1])
            )
            right_term = torch.where(
                right_den.abs() > 1e-8,
                right_num / right_den * bases[..., 1:],
                torch.zeros_like(bases[..., 1:])
            )
            bases = left_term + right_term
        
        assert bases.size(-1) == self.n_bases, (
            f"Expected {self.n_bases} bases, got {bases.size(-1)}"
        )
        return bases
    
    def forward(
        self, x: Float[torch.Tensor, "B D_in"]
    ) -> Float[torch.Tensor, "B D_out"]:
        """Compute input-dependent modulation factors.
        
        Args:
            x: Input tensor of shape [batch, input_dim].
            
        Returns:
            Modulation factors of shape [batch, output_dim].
            Values are positive (suitable for scaling SSM matrices).
        """
        batch_size = x.size(0)
        
        # Normalize input to grid range for stable spline evaluation
        x_norm = torch.tanh(x)  # Compress to [-1, 1]
        
        # Compute B-spline basis activations
        spline_bases = self._b_splines(x_norm)  # [B, D_in, n_bases]
        
        # Contract with learned coefficients
        # coefficients: [D_out, D_in, n_bases]
        # spline_bases: [B, D_in, n_bases]
        # Result: [B, D_out]
        modulation = torch.einsum(
            'oin,bin->bo',
            self.coefficients,
            spline_bases
        )
        
        # Apply scale and bias
        scale = self.log_scale.exp()
        modulation = modulation * scale + self.bias
        
        # Ensure positive output for matrix scaling
        if self.activation_type == "softplus":
            # Softplus with offset to ensure modulation ≥ 0.1
            modulation = F.softplus(modulation) + 0.1
        elif self.activation_type == "sigmoid":
            # Sigmoid scaled to [0.5, 1.5] for mild modulation
            modulation = torch.sigmoid(modulation) + 0.5
        elif self.activation_type == "exp":
            # Exponential (can be unstable, use with care)
            modulation = torch.exp(modulation.clamp(-5, 5))
        
        return modulation
    
    def regularization_loss(
        self, l1_weight: float = 1.0, entropy_weight: float = 0.1
    ) -> torch.Tensor:
        """Compute regularization loss on coefficients.
        
        Encourages sparsity in the learned polynomial representation.
        
        The entropy term uses negative entropy: -H(p) = Σ p·log(p)
        This is ≤ 0 for any distribution, with max at uniform (log(1/n)).
        Minimizing this encourages sparse (non-uniform) coefficients.
        """
        # L1 sparsity on coefficients
        l1_loss = self.coefficients.abs().mean()
        
        # Negative entropy regularization (encourages sparsity)
        # H(p) = -Σ p·log(p) is always positive
        # We use Σ p·log(p) which is always negative, minimizing it = maximizing entropy
        # Actually we want to MINIMIZE entropy to encourage sparsity, so we use:
        # entropy_loss = -(-Σ p·log(p)) = Σ p·log(p) ≤ 0
        coeff_probs = F.softmax(self.coefficients.abs().view(-1), dim=0)
        entropy_loss = (coeff_probs * (coeff_probs + 1e-10).log()).sum()  # ≤ 0
        
        return l1_weight * l1_loss + entropy_weight * entropy_loss
    
    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, "
            f"output_dim={self.output_dim}, "
            f"grid_size={self.grid_size}, "
            f"spline_order={self.spline_order}, "
            f"activation={self.activation_type}"
        )


class FastKANModulator(nn.Module):
    """Fast KAN-based Modulator using Gaussian RBFs.
    
    Faster variant of KANModulator.
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        grid_size: int = 8,
        activation: Literal["softplus", "sigmoid", "exp"] = "softplus",
    ) -> None:
        super().__init__()
        self.kan = FastKANLayer(input_dim, output_dim, grid_size=grid_size)
        self.activation_type = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # FastKANLayer already includes LayerNorm and RBF+Linear
        out = self.kan(x)
        
        # Apply activation for modulation constraints
        if self.activation_type == "softplus":
            return F.softplus(out) + 0.1
        elif self.activation_type == "sigmoid":
            return torch.sigmoid(out) + 0.5
        elif self.activation_type == "exp":
            return torch.exp(out.clamp(-5, 5))
        return out

    def regularization_loss(self) -> torch.Tensor:
        return self.kan.regularization_loss()


class MLPModulator(nn.Module):
    """Two-layer MLP modulator — drop-in baseline for FastKANModulator.

    Used in ablation experiments to provide a dense, entangled gating
    vector as a contrast to the sparse KAN modulation.

    The hidden dimension ``input_dim`` gives the same width as the input,
    ensuring a fair comparison without inflating the parameter count.

    Args:
        input_dim: Dimension of the input features.
        output_dim: Dimension of the modulation output.
        grid_size: Accepted for API parity with FastKANModulator; not used.
        activation: Activation to ensure positive modulation
            (``'softplus'``, ``'sigmoid'``, or ``'exp'``).

    Shape:
        - Input:  ``[B, input_dim]``
        - Output: ``[B, output_dim]``

    Example::

        >>> mod = MLPModulator(768, 768)
        >>> out = mod(torch.randn(4, 768))
        >>> out.shape
        torch.Size([4, 768])
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        grid_size: int = 8,          # noqa: ARG002  — API parity only
        activation: Literal["softplus", "sigmoid", "exp"] = "softplus",
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")

        hidden_dim = input_dim  # symmetric 2-layer MLP
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )
        self.activation_type = activation
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialise output linear so activation(0) ≈ 1.0 at start of training."""
        # Last linear in self.net is index 3
        nn.init.zeros_(self.net[3].weight)
        if self.activation_type == "softplus":
            # softplus(x) + 0.1 = 1.0  =>  x ≈ 0.378
            nn.init.constant_(self.net[3].bias, 0.38)
        else:
            nn.init.zeros_(self.net[3].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute dense modulation factors.

        Args:
            x: Input tensor ``[B, input_dim]``.

        Returns:
            Modulation factors ``[B, output_dim]``, positive-valued.
        """
        out = self.net(x)
        if self.activation_type == "softplus":
            return F.softplus(out) + 0.1
        elif self.activation_type == "sigmoid":
            return torch.sigmoid(out) + 0.5
        elif self.activation_type == "exp":
            return torch.exp(out.clamp(-5, 5))
        return out

    def regularization_loss(self) -> torch.Tensor:
        """L1 regularisation on weight matrices."""
        return sum(p.abs().mean() for p in self.parameters() if p.ndim >= 2)

    def extra_repr(self) -> str:
        hidden = self.net[1].out_features
        return (
            f"input_dim={self.net[1].in_features}, "
            f"hidden_dim={hidden}, "
            f"output_dim={self.net[3].out_features}, "
            f"activation={self.activation_type}"
        )


class IntricateKANSSMCore(nn.Module):
    """Intricate KAN-SSM Core with learned B and C modulation.
    
    This implements the deep integration of KAN into Mamba's SSM,
    where learned polynomial coefficients modulate how inputs affect
    state (B matrix) and how states produce outputs (C matrix).
    
    Args:
        inner_dim: Dimension of the SSM input/output.
        state_dim: Dimension of the hidden state.
        grid_size: Grid size for KAN modulators.
        modulate_B: Whether to use KAN modulation on B matrix.
        modulate_C: Whether to use KAN modulation on C matrix.
        modulation_mode: How to apply modulation ('element', 'factor', 'mixture').
            - 'element': Element-wise scaling of entire B/C
            - 'factor': Per-feature scaling factors
            - 'mixture': Mixture of expert B/C matrices
    
    The modulation modes enable different levels of expressiveness:
        - 'element': Simple but efficient, single scalar per sample
        - 'factor': Per-feature importance weighting
        - 'mixture': K different B/C templates, soft-selected per sample
    """
    
    def __init__(
        self,
        inner_dim: int,
        state_dim: int,
        *,
        grid_size: int = 5,
        modulate_B: bool = True,
        modulate_C: bool = True,
        modulation_mode: Literal["element", "factor", "mixture"] = "factor",
        n_mixtures: int = 4,  # Only used if modulation_mode == "mixture"
        use_fast_kan: bool = False,
        modulator_type: str = "kan",
        use_mamba_kernels: bool = True,
        learnable_A: bool = False,  # Default to False for stability
        identity_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        
        self.inner_dim = inner_dim
        self.state_dim = state_dim
        self.modulate_B = modulate_B
        self.modulate_C = modulate_C
        self.modulation_mode = modulation_mode
        self.n_mixtures = n_mixtures
        self.use_fast_kan = use_fast_kan
        self.modulator_type = modulator_type
        self.use_mamba_kernels = use_mamba_kernels and HAS_MAMBA_KERNELS
        
        # Initialize HiPPO-LegS matrices for long-range dependencies
        A_init, B_init, C_init = self._hippo_legS_init(state_dim)
        scale = 1.0 / math.sqrt(inner_dim)
        
        if modulation_mode == "mixture":
            # Multiple B and C prototypes for mixture-of-experts
            if learnable_A:
                self.A = nn.Parameter(A_init)
            else:
                self.register_buffer("A", A_init)
                
            self.B_stack = nn.Parameter(
                B_init.unsqueeze(1).repeat(1, inner_dim).unsqueeze(0).repeat(n_mixtures, 1, 1) * scale +
                torch.randn(n_mixtures, state_dim, inner_dim) * 0.01
            )
            self.C_stack = nn.Parameter(
                C_init.unsqueeze(0).repeat(inner_dim, 1).unsqueeze(0).repeat(n_mixtures, 1, 1) * scale +
                torch.randn(n_mixtures, inner_dim, state_dim) * 0.01
            )
        else:
            # Standard single B and C
            if learnable_A:
                self.A = nn.Parameter(A_init)
            else:
                self.register_buffer("A", A_init)
                
            self.B = nn.Parameter(B_init.unsqueeze(1).repeat(1, inner_dim) * scale)
            self.C = nn.Parameter(C_init.unsqueeze(0).repeat(inner_dim, 1) * scale)
        
        # KAN Modulators for B matrix (input-to-state)
        # Select modulator class based on modulator_type flag.
        # 'mlp'  → dense two-layer MLP (ablation baseline)
        # 'kan'  → FastKAN (RBF) or B-spline KAN depending on use_fast_kan
        if modulator_type == "mlp":
            ModulatorClass = MLPModulator
        else:
            ModulatorClass = FastKANModulator if use_fast_kan else KANModulator
        
        if modulate_B:
            # We allow an optional identity_dim to be concatenated for modulation
            mod_in_dim = inner_dim + (identity_dim if identity_dim is not None else 0)
            if modulation_mode == "element":
                # Single scalar modulation per sample
                self.B_modulator = ModulatorClass(
                    mod_in_dim, 1, grid_size=grid_size, activation="softplus"
                )
            elif modulation_mode == "factor":
                # Per-feature modulation factors
                self.B_modulator = ModulatorClass(
                    mod_in_dim, inner_dim, grid_size=grid_size, activation="softplus"
                )
            elif modulation_mode == "mixture":
                # Mixture weights for B prototypes
                self.B_modulator = ModulatorClass(
                    mod_in_dim, n_mixtures, grid_size=grid_size, activation="softplus"
                )
        
        # KAN Modulators for C matrix (state-to-output)
        if modulate_C:
            mod_in_dim = inner_dim + (identity_dim if identity_dim is not None else 0)
            if modulation_mode == "element":
                self.C_modulator = ModulatorClass(
                    mod_in_dim, 1, grid_size=grid_size, activation="softplus"
                )
            elif modulation_mode == "factor":
                # Per-output modulation: [B, D] -> scale each row of C
                self.C_modulator = ModulatorClass(
                    mod_in_dim, inner_dim, grid_size=grid_size, activation="softplus"
                )
            elif modulation_mode == "mixture":
                # Mixture-of-experts
                self.C_modulator = ModulatorClass(
                    mod_in_dim, n_mixtures, grid_size=grid_size, activation="softplus"
                )
        
        # Identity matrix for discretization
        self.register_buffer("eye_state", torch.eye(state_dim), persistent=False)
        
        # SSM discretization bounds
        self.delta_min = 1e-4
        self.delta_max = 3.0
        
    @staticmethod
    def _hippo_legS_init(
        state_dim: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """HiPPO-LegS initialization for long-range dependencies."""
        A = torch.zeros((state_dim, state_dim), dtype=torch.float32)
        for n in range(state_dim):
            coeff_n = math.sqrt(2 * n + 1)
            for m in range(n + 1):
                coeff_m = math.sqrt(2 * m + 1)
                if n == m:
                    A[n, m] = -(2 * n + 1)
                else:
                    sign = -1.0 if (n - m) % 2 == 0 else 1.0
                    A[n, m] = sign * coeff_n * coeff_m
        
        indices = torch.arange(state_dim, dtype=torch.float32)
        B = torch.sqrt(2.0 * indices + 1.0)
        C = B.clone()
        C[1::2] *= -1.0
        
        return A, B, C
    
    def _get_modulated_B(
        self,
        u_t: Float[torch.Tensor, "B D"],
        identity: Optional[Float[torch.Tensor, "B D_id"]] = None,
    ) -> Float[torch.Tensor, "B N D"]:
        """Get KAN-modulated B matrix for current input.
        
        Args:
            u_t: Current input [batch, inner_dim]
            identity: Optional object identity vector [batch, identity_dim]
            
        Returns:
            Modulated B matrix [batch, state_dim, inner_dim]
        """
        batch = u_t.size(0)
        
        if not self.modulate_B:
            # No modulation, broadcast base B
            return self.B.unsqueeze(0).expand(batch, -1, -1)
        
        # Combine input and identity for modulation
        mod_in = u_t
        if identity is not None:
            mod_in = torch.cat([u_t, identity], dim=-1)

        # Compute modulation from KAN
        mod = self.B_modulator(mod_in)  # [B, ?]
        
        if self.modulation_mode == "element":
            # Scalar modulation: [B, 1] -> scale entire B
            return self.B.unsqueeze(0) * mod.unsqueeze(-1)  # [B, N, D]
            
        elif self.modulation_mode == "factor":
            # Per-feature modulation: [B, D] -> scale each column of B
            return self.B.unsqueeze(0) * mod.unsqueeze(1)  # [B, N, D]
            
        elif self.modulation_mode == "mixture":
            # Mixture-of-experts: [B, K] -> weighted sum of B prototypes
            weights = F.softmax(mod, dim=-1)  # [B, K]
            # B_stack: [K, N, D]
            B_mix = torch.einsum('bk,knd->bnd', weights, self.B_stack)
            return B_mix
    
    def _get_modulated_C(
        self,
        x: Float[torch.Tensor, "B D"],
        identity: Optional[Float[torch.Tensor, "B D_id"]] = None,
    ) -> Float[torch.Tensor, "B D N"]:
        """Get KAN-modulated C matrix based on input (not state).
        
        NOTE: This method now takes input x instead of state h for parallelization.
        
        Args:
            x: Current input [batch, inner_dim]
            identity: Optional object identity vector [batch, identity_dim]
            
        Returns:
            Modulated C matrix [batch, inner_dim, state_dim]
        """
        batch = x.size(0)
        
        if not self.modulate_C:
            # No modulation, broadcast base C
            return self.C.unsqueeze(0).expand(batch, -1, -1)
        
        # Combine input and identity for modulation
        mod_in = x
        if identity is not None:
            mod_in = torch.cat([x, identity], dim=-1)

        # Compute modulation from KAN based on input
        mod = self.C_modulator(mod_in)  # [B, ?]
        
        if self.modulation_mode == "element":
            # Scalar modulation
            return self.C.unsqueeze(0) * mod.unsqueeze(-1)
            
        elif self.modulation_mode == "factor":
            # Per-output modulation: [B, D] -> scale each row of C
            return self.C.unsqueeze(0) * mod.unsqueeze(-1)
            
        elif self.modulation_mode == "mixture":
            # Mixture-of-experts
            weights = F.softmax(mod, dim=-1)  # [B, K]
            # C_stack: [K, D, N]
            C_mix = torch.einsum('bk,kdn->bdn', weights, self.C_stack)
            return C_mix
    
    def _compute_C_modulation_batched(
        self, x: Float[torch.Tensor, "BT D"],
        identity: Optional[Float[torch.Tensor, "BT D_id"]] = None,
    ) -> Float[torch.Tensor, "BT D N"]:
        """Compute C modulation for batched inputs.
        
        Args:
            x: Flattened input [batch*seq_len, inner_dim]
            identity: Flattened identity context [batch*seq_len, identity_dim]
            
        Returns:
            Modulated C matrices [batch*seq_len, inner_dim, state_dim]
        """
        bt = x.size(0)
        
        # Combine input and identity
        mod_in = x
        if identity is not None:
             mod_in = torch.cat([x, identity], dim=-1)

        if not self.modulate_C:
            return self.C.unsqueeze(0).expand(bt, -1, -1)
        
        # Get modulation factors
        mod = self.C_modulator(mod_in)  # [BT, ?]
        
        if self.modulation_mode == "element":
            # Scalar modulation
            return self.C.unsqueeze(0) * mod.unsqueeze(-1)  # [BT, D, N]
            
        elif self.modulation_mode == "factor":
            # Per-feature modulation
            return self.C.unsqueeze(0) * mod.unsqueeze(-1)  # [BT, D, N]
            
        elif self.modulation_mode == "mixture":
            # Mixture of experts
            weights = F.softmax(mod, dim=-1)  # [BT, K]
            return torch.einsum('bk,kdn->bdn', weights, self.C_stack)
    
    def forward(
        self,
        x: Float[torch.Tensor, "B T D"],
        delta: Float[torch.Tensor, "B T 1"],
        initial_state: Optional[Float[torch.Tensor, "B N 1"]] = None,
        return_last_state: bool = False,
        identity: Optional[Float[torch.Tensor, "B D_id"]] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Selective scan with KAN-modulated B and C matrices (fully optimized).
        
        This method implements the core 'Temporal Modeling' step of the Training/Inference processes.
        It uses KAN-based modulation factors to adjust the SSM matrices dynamically.
        
        Args:
            x: Input sequence [batch, seq_len, inner_dim].
            delta: Step sizes [batch, seq_len, 1].
            initial_state: Optional initial hidden state [batch, state_dim, 1].
            return_last_state: Whether to return the final hidden state.
            identity: Optional object identity vector [batch, identity_dim].
            
        Returns:
            Output sequence [batch, seq_len, inner_dim] or 
            tuple (output, final_state).
        """
        batch, seq_len, _ = x.shape
        device = x.device
        dtype = x.dtype
        
        # === OPTIMIZED KERNEL PATH (NOT COMPATIBLE WITH INTENSE B DISCRETIZATION) ===
        # (Keeping the infrastructure but following the Python path for full SSM)

        # === PRECOMPUTE ALL MODULATIONS FOR ALL TIMESTEPS ===
        x_flat = x.reshape(batch * seq_len, -1)  # [B*T, D]
        
        # Broadcast identity over time if provided
        id_flat = None
        if identity is not None:
             # identity is [B, D_id] -> [B, T, D_id] -> [B*T, D_id]
             id_flat = identity.unsqueeze(1).repeat(1, seq_len, 1).reshape(batch * seq_len, -1)

        # B modulations
        if self.modulate_B:
            B_mod_flat = self._compute_B_modulation_batched(x_flat, identity=id_flat)  # [B*T, N, D]
            B_mod_all = B_mod_flat.view(batch, seq_len, self.state_dim, self.inner_dim)
        else:
            B_mod_all = self.B.unsqueeze(0).unsqueeze(0).expand(batch, seq_len, -1, -1)
        
        # C modulations (NOW BASED ON INPUT, NOT STATE!)
        C_mod_flat = self._compute_C_modulation_batched(x_flat, identity=id_flat)  # [B*T, D, N]
        C_mod_all = C_mod_flat.view(batch, seq_len, self.inner_dim, self.state_dim)  # [B, T, D, N]
        
        # === PRECOMPUTE MATRIX EXPONENTIALS ===
        delta_expanded = delta.unsqueeze(-1)  # [B, T, 1, 1]
        A_scaled = delta_expanded * self.A.unsqueeze(0).unsqueeze(0)  # [B, T, N, N]
        
        A_scaled_flat = A_scaled.reshape(batch * seq_len, self.state_dim, self.state_dim)
        # Avoid exploding exponentials
        A_scaled_flat = A_scaled_flat.clamp(min=-50, max=20) 
        A_expm_flat = torch.matrix_exp(A_scaled_flat)
        A_expm_all = A_expm_flat.view(batch, seq_len, self.state_dim, self.state_dim)
        
        # === PRECOMPUTE B DISCRETIZATION ===
        eye = self.eye_state.to(device=device, dtype=dtype)
        
        # We use a robust discretization for the integral: (exp(A*delta) - I) * A^-1
        A_expand = self.A.unsqueeze(0).unsqueeze(0).expand(batch, seq_len, -1, -1)
        A_flat = A_expand.reshape(batch * seq_len, self.state_dim, self.state_dim)
        diff_flat = (A_expm_flat - eye)
        
        try:
            # Try to solve the ZOH integral: A * integral = exp(A*delta) - I
            A_stable = A_flat + 1e-6 * eye.unsqueeze(0)
            integral_flat = torch.linalg.solve(A_stable, diff_flat)
            if torch.isnan(integral_flat).any():
                integral_flat = delta_expanded.view(-1, 1, 1) * (eye.unsqueeze(0) + 0.5 * A_scaled_flat)
        except RuntimeError:
            integral_flat = delta_expanded.view(-1, 1, 1) * (eye.unsqueeze(0) + 0.5 * A_scaled_flat)
            
        integral_all = integral_flat.view(batch, seq_len, self.state_dim, self.state_dim)
        B_disc_all = torch.einsum('btij,btjd->btid', integral_all, B_mod_all)  # [B, T, N, D]
        
        # === SEQUENTIAL STATE UPDATES (fundamental RNN constraint) ===
        states_all = x.new_zeros(batch, seq_len, self.state_dim, 1)  # [B, T, N, 1]
        state = initial_state if initial_state is not None else x.new_zeros(batch, self.state_dim, 1)
        
        for t in range(seq_len):
            u_t = x[:, t, :]  # [B, D]
            A_expm_t = A_expm_all[:, t, :, :]  # [B, N, N]
            B_disc_t = B_disc_all[:, t, :, :]  # [B, N, D]
            
            # State update: h = A_exp @ h + B_disc @ u
            state = torch.bmm(A_expm_t, state)  # [B, N, 1]
            state = state + torch.bmm(B_disc_t, u_t.unsqueeze(-1))  # [B, N, 1]
            states_all[:, t, :, :] = state
        
        # === VECTORIZED OUTPUT COMPUTATION ===
        outputs = torch.einsum('btdn,btn->btd', C_mod_all, states_all.squeeze(-1))  # [B, T, D]
        
        if return_last_state:
            return outputs, state
        return outputs
    
    def _compute_B_modulation_batched(
        self, x: Float[torch.Tensor, "BT D"],
        identity: Optional[Float[torch.Tensor, "BT D_id"]] = None,
    ) -> Float[torch.Tensor, "BT N D"]:
        """Compute B modulation for batched inputs.
        
        Args:
            x: Flattened input [batch*seq_len, inner_dim]
            identity: Flattened identity context [batch*seq_len, identity_dim]
            
        Returns:
            Modulated B matrices [batch*seq_len, state_dim, inner_dim]
        """
        bt = x.size(0)
        
        # Combine input and identity
        mod_in = x
        if identity is not None:
             mod_in = torch.cat([x, identity], dim=-1)

        # Get modulation factors
        mod = self.B_modulator(mod_in)  # [BT, ?]
        
        if self.modulation_mode == "element":
            # Scalar modulation
            return self.B.unsqueeze(0) * mod.unsqueeze(-1)  # [BT, N, D]
            
        elif self.modulation_mode == "factor":
            # Per-feature modulation
            return self.B.unsqueeze(0) * mod.unsqueeze(1)  # [BT, N, D]
            
        elif self.modulation_mode == "mixture":
            # Mixture of experts
            weights = F.softmax(mod, dim=-1)  # [BT, K]
            return torch.einsum('bk,knd->bnd', weights, self.B_stack)
    
    def get_regularization_loss(self) -> torch.Tensor:
        """Get combined regularization loss from KAN modulators."""
        loss = torch.tensor(0.0, device=self.A.device)
        
        if self.modulate_B:
            loss = loss + self.B_modulator.regularization_loss()
        if self.modulate_C:
            loss = loss + self.C_modulator.regularization_loss()
            
        return loss
    
    def extra_repr(self) -> str:
        return (
            f"inner_dim={self.inner_dim}, "
            f"state_dim={self.state_dim}, "
            f"modulate_B={self.modulate_B}, "
            f"modulate_C={self.modulate_C}, "
            f"modulation_mode={self.modulation_mode}, "
            f"modulator_type={self.modulator_type}, "
            f"use_fast_kan={self.use_fast_kan}, "
            f"mamba_kernels={self.use_mamba_kernels}"
        )


if __name__ == "__main__":
    print("Testing IntricateKANSSMCore...")
    
    # Test different modulation modes
    for mode in ["element", "factor", "mixture"]:
        print(f"\n--- Mode: {mode} ---")
        core = IntricateKANSSMCore(
            inner_dim=64,
            state_dim=16,
            modulate_B=True,
            modulate_C=True,
            modulation_mode=mode,
        )
        
        x = torch.randn(4, 32, 64)  # [batch, seq_len, dim]
        delta = torch.ones(4, 32, 1) * 0.1
        
        out = core(x, delta)
        print(f"Input: {x.shape} -> Output: {out.shape}")
        print(f"Parameters: {sum(p.numel() for p in core.parameters()):,}")
        print(f"Regularization loss: {core.get_regularization_loss().item():.6f}")
        
        # Test gradient flow
        loss = out.sum()
        loss.backward()
        print("Gradient flow: OK")
    
    print("\n✅ All tests passed!")
