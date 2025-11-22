"""
Moral Game Generator (MGG) - generates games with payoff archetypes and fairness functions.

This module implements the game generation framework from the paper, including:
- Four payoff archetypes (Dilemma, Assurance, Bargain, Public Goods)
- Three fairness functions (Max-Sum, Equal-Split, Rawls)
- Game sampling with normalized payoffs and fairness values
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Callable, Tuple
import numpy as np


class Archetype(Enum):
    """Payoff archetypes from the paper."""
    DILEMMA = 1      # Prisoner's dilemma style
    ASSURANCE = 2    # Stag hunt / coordination game
    BARGAIN = 3      # Bargaining over resources
    PUBLIC_GOODS = 4 # Public goods contribution


@dataclass
class Game:
    """
    Represents a single sampled game with payoffs and fairness functions.
    
    All payoffs and fairness values are normalized to [0, 1].
    """
    archetype: Archetype
    action_space_i: np.ndarray    # 1D array of actions for player i
    action_space_j: np.ndarray    # 1D array of actions for player j
    R_i: np.ndarray              # Payoff matrix for i, shape (n_ai, n_aj)
    R_j: np.ndarray              # Payoff matrix for j, shape (n_ai, n_aj)
    F: Dict[str, np.ndarray]     # Fairness functions: {"max_sum": ..., "equal_split": ..., "rawls": ...}
    R_max: float                 # Maximum payoff (for reference)


# ============================================================================
# Payoff Archetype Functions
# ============================================================================

def payoff_dilemma(ai: float, aj: float, params: Dict) -> Tuple[float, float]:
    """
    Dilemma archetype (Prisoner's Dilemma style).
    
    Players choose cooperation level in [0, 1].
    Cost c * action, benefit b * partner's action, b > c.
    Defection (low action) dominates but mutual cooperation is better.
    """
    c = params.get('cost', 0.5)
    b = params.get('benefit', 1.0)
    
    R_i = b * aj - c * ai
    R_j = b * ai - c * aj
    
    return R_i, R_j


def payoff_assurance(ai: float, aj: float, params: Dict) -> Tuple[float, float]:
    """
    Assurance archetype (Stag Hunt / Coordination).
    
    High payoffs when both coordinate on high actions (stag),
    safe but lower payoff at low actions (hare).
    """
    # Coordination bonus when both cooperate
    coord_threshold = params.get('threshold', 0.5)
    high_payoff = params.get('high', 1.0)
    low_payoff = params.get('low', 0.3)
    
    # Both coordinate high
    if ai >= coord_threshold and aj >= coord_threshold:
        R_i = high_payoff * ai * aj
        R_j = high_payoff * ai * aj
    # Safe low payoff
    else:
        R_i = low_payoff * (1 - ai) * (1 - aj) + 0.1 * ai * aj
        R_j = low_payoff * (1 - ai) * (1 - aj) + 0.1 * ai * aj
    
    return R_i, R_j


def payoff_bargain(ai: float, aj: float, params: Dict) -> Tuple[float, float]:
    """
    Bargain archetype - splitting a resource.
    
    Actions represent demands. If demands are compatible (sum ≤ 1), 
    each gets their demand; otherwise both get nothing.
    """
    pie_size = params.get('pie_size', 1.0)
    
    if ai + aj <= pie_size:
        R_i = ai
        R_j = aj
    else:
        # Demands incompatible - disagreement outcome
        R_i = 0.0
        R_j = 0.0
    
    return R_i, R_j


def payoff_public_goods(ai: float, aj: float, params: Dict) -> Tuple[float, float]:
    """
    Public Goods archetype.
    
    Each player contributes (action level), gets share of multiplied total.
    Freeriding (low contribution) tempting but mutual contribution best.
    """
    multiplier = params.get('multiplier', 2.0)
    endowment = params.get('endowment', 1.0)
    
    # Contribution to public good
    total_contribution = ai + aj
    public_good = multiplier * total_contribution
    
    # Each gets equal share of public good, plus keeps what they didn't contribute
    R_i = public_good / 2.0 + (endowment - ai)
    R_j = public_good / 2.0 + (endowment - aj)
    
    return R_i, R_j


# ============================================================================
# Game Sampling
# ============================================================================

def sample_payoff_function(archetype: Archetype, rng: np.random.Generator) -> Callable:
    """
    Sample a payoff function for the given archetype.
    
    Returns a closure payoff(ai, aj) -> (R_i, R_j).
    """
    if archetype == Archetype.DILEMMA:
        # Sample cost and benefit parameters
        cost = rng.uniform(0.3, 0.7)
        benefit = rng.uniform(cost + 0.2, 1.5)
        params = {'cost': cost, 'benefit': benefit}
        return lambda ai, aj: payoff_dilemma(ai, aj, params)
    
    elif archetype == Archetype.ASSURANCE:
        threshold = rng.uniform(0.4, 0.6)
        high = rng.uniform(0.8, 1.2)
        low = rng.uniform(0.2, 0.4)
        params = {'threshold': threshold, 'high': high, 'low': low}
        return lambda ai, aj: payoff_assurance(ai, aj, params)
    
    elif archetype == Archetype.BARGAIN:
        pie_size = rng.uniform(0.8, 1.2)
        params = {'pie_size': pie_size}
        return lambda ai, aj: payoff_bargain(ai, aj, params)
    
    elif archetype == Archetype.PUBLIC_GOODS:
        # For freeriding to be tempting: multiplier/n < 1, so m < n
        # With n=2, keep multiplier slightly less than 2
        multiplier = rng.uniform(1.5, 1.9)
        endowment = rng.uniform(0.8, 1.2)
        params = {'multiplier': multiplier, 'endowment': endowment}
        return lambda ai, aj: payoff_public_goods(ai, aj, params)
    
    else:
        raise ValueError(f"Unknown archetype: {archetype}")


def compute_fairness_functions(R_i: np.ndarray, R_j: np.ndarray, R_max: float) -> Dict[str, np.ndarray]:
    """
    Compute the three fairness functions from payoff matrices.
    
    From the paper:
    - F_Max-Sum = (R_i + R_j) / (2 * R_max)  # normalized by max possible sum
    - F_Equal-Split = 1 - |R_i - R_j| / R_max
    - F_Rawls = min(R_i, R_j) / R_max
    
    All values are in [0, 1].
    """
    # Max-sum normalized by the maximum possible sum (2 * R_max)
    F_max_sum = (R_i + R_j) / (2.0 * R_max)
    F_equal_split = 1.0 - np.abs(R_i - R_j) / R_max
    F_rawls = np.minimum(R_i, R_j) / R_max
    
    return {
        "max_sum": F_max_sum,
        "equal_split": F_equal_split,
        "rawls": F_rawls,
    }


class MoralGameGenerator:
    """
    Generates games from the Moral Game Generator (MGG).
    
    Each game has:
    - A payoff archetype
    - Discrete action spaces for both players
    - Payoff matrices R_i, R_j normalized to [0, 1]
    - Fairness functions F_φ for φ ∈ {Max-Sum, Equal-Split, Rawls}
    """
    
    def __init__(self, 
                 archetype_probs: np.ndarray = None,
                 action_grid: np.ndarray = None,
                 rng: np.random.Generator = None,
                 num_actions: int = 21):
        """
        Initialize the MGG.
        
        Args:
            archetype_probs: Probability distribution over archetypes (default: uniform)
            action_grid: Discrete action space (default: linspace(0, 1, num_actions))
            rng: Random number generator
            num_actions: Number of discrete actions (if action_grid not provided)
        """
        self.rng = rng if rng is not None else np.random.default_rng()
        
        # Uniform distribution over archetypes by default
        if archetype_probs is None:
            archetype_probs = np.ones(4) / 4
        self.archetype_probs = archetype_probs
        
        # Discrete action space
        if action_grid is None:
            action_grid = np.linspace(0.0, 1.0, num_actions)
        self.action_grid = action_grid
    
    def sample_game(self) -> Game:
        """
        Sample a game from the MGG.
        
        Returns a Game object with payoffs and fairness functions.
        """
        # 1. Sample archetype
        archetype = self.rng.choice(list(Archetype), p=self.archetype_probs)
        
        # 2. Sample payoff function for this archetype
        payoff_fn = sample_payoff_function(archetype, self.rng)
        
        # 3. Construct payoff matrices over discrete action grid
        A = self.action_grid
        n = len(A)
        R_i = np.zeros((n, n))
        R_j = np.zeros((n, n))
        
        for idx_i, ai in enumerate(A):
            for idx_j, aj in enumerate(A):
                Ri_val, Rj_val = payoff_fn(ai, aj)
                R_i[idx_i, idx_j] = Ri_val
                R_j[idx_i, idx_j] = Rj_val
        
        # 4. Normalize payoffs to [0, 1]
        R_min = min(R_i.min(), R_j.min())
        R_max_raw = max(R_i.max(), R_j.max())
        
        # Avoid division by zero
        if R_max_raw - R_min < 1e-10:
            R_max_raw = R_min + 1.0
        
        R_i = (R_i - R_min) / (R_max_raw - R_min)
        R_j = (R_j - R_min) / (R_max_raw - R_min)
        R_max = max(R_i.max(), R_j.max())
        
        # 5. Compute fairness functions
        F = compute_fairness_functions(R_i, R_j, R_max)
        
        return Game(
            archetype=archetype,
            action_space_i=A.copy(),
            action_space_j=A.copy(),
            R_i=R_i,
            R_j=R_j,
            F=F,
            R_max=R_max,
        )


# ============================================================================
# Moral Prior Sampling
# ============================================================================

def sample_moral_prior(rng: np.random.Generator, num_principles: int = 3) -> np.ndarray:
    """
    Sample a moral prior from Dirichlet(1, 1, 1).
    
    Returns a probability distribution over fairness principles:
    [w_Max-Sum, w_Equal-Split, w_Rawls]
    """
    alpha = np.ones(num_principles)
    return rng.dirichlet(alpha)
