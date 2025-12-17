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
    ai, aj: Actions in [0, 1] where 0=defect, 1=cooperate
    params: {'cost': 0.5, 'benefit': 1.0}
    """
    c = params.get('cost', 0.5)
    b = params.get('benefit', 1.0)
    
    # I pay cost for my cooperation, get benefit from partner's cooperation
    R_i = b * aj - c * ai
    R_j = b * ai - c * aj
    
    return R_i, R_j

# some example situations for this -> 

# # Both cooperate fully
# R_i, R_j = payoff_dilemma(1.0, 1.0, {'cost': 0.5, 'benefit': 1.0})
# # R_i = 1.0 × 1.0 - 0.5 × 1.0 = 0.5
# # R_j = 1.0 × 1.0 - 0.5 × 1.0 = 0.5

# # I defect (0.0), you cooperate (1.0)
# R_i, R_j = payoff_dilemma(0.0, 1.0, {'cost': 0.5, 'benefit': 1.0})
# # R_i = 1.0 × 1.0 - 0.5 × 0.0 = 1.0  ← I do better!
# # R_j = 1.0 × 0.0 - 0.5 × 1.0 = -0.5 ← You do worse!

# # Both defect
# R_i, R_j = payoff_dilemma(0.0, 0.0, {'cost': 0.5, 'benefit': 1.0})
# # R_i = 1.0 × 0.0 - 0.5 × 0.0 = 0.0
# # R_j = 0.0


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

# some example situations for assurance -> 

# params = {'threshold': 0.5, 'high': 1.0, 'low': 0.3}

# # Both coordinate HIGH (both hunt stag together)
# R_i, R_j = payoff_assurance(0.8, 0.8, params)
# # Both ≥ 0.5, so: R_i = 1.0 × 0.8 × 0.8 = 0.64
# # R_j = 1.0 × 0.8 × 0.8 = 0.64
# # BEST OUTCOME: High reward when both commit!

# # Both coordinate LOW (both hunt hare - safe choice)
# R_i, R_j = payoff_assurance(0.2, 0.2, params)
# # Both < 0.5, so: R_i = 0.3 × (1-0.2) × (1-0.2) + 0.1 × 0.2 × 0.2
# #                    = 0.3 × 0.8 × 0.8 + 0.1 × 0.04 = 0.192 + 0.004 = 0.196
# # SAFE OUTCOME: Decent payoff, no risk

# # MISMATCH: I go high (0.8), you go low (0.2) - DISASTER!
# R_i, R_j = payoff_assurance(0.8, 0.2, params)
# # One ≥ threshold but not both, so use low formula:
# # R_i = 0.3 × (1-0.8) × (1-0.2) + 0.1 × 0.8 × 0.2
# #     = 0.3 × 0.2 × 0.8 + 0.1 × 0.16 = 0.048 + 0.016 = 0.064
# # WORST OUTCOME: Coordination failure hurts both!
# 
# # The dilemma: Going high is risky - if partner doesn't match, you do worse
# # than if you both played safe. But if both match high, it's the best outcome.


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

# some example situations for bargain -> 

# params = {'pie_size': 1.0}

# # Compatible demands: Both modest
# R_i, R_j = payoff_bargain(0.4, 0.5, params)
# # Sum = 0.4 + 0.5 = 0.9 ≤ 1.0 ✓
# # R_i = 0.4 (I get what I asked for)
# # R_j = 0.5 (You get what you asked for)
# # Both satisfied! Left 0.1 on the table though.

# # Compatible demands: Both greedy but just fit
# R_i, R_j = payoff_bargain(0.6, 0.4, params)
# # Sum = 0.6 + 0.4 = 1.0 ≤ 1.0 ✓ (exact fit!)
# # R_i = 0.6
# # R_j = 0.4
# # Efficient! No waste, but unequal split.

# # Incompatible demands: Both too greedy
# R_i, R_j = payoff_bargain(0.7, 0.8, params)
# # Sum = 0.7 + 0.8 = 1.5 > 1.0 ✗
# # R_i = 0.0 (Disagreement! Get nothing!)
# # R_j = 0.0 (Both lose!)
# # DISASTER: Greed leads to conflict.

# # Very modest demands: Leave money on table
# R_i, R_j = payoff_bargain(0.2, 0.3, params)
# # Sum = 0.2 + 0.3 = 0.5 ≤ 1.0 ✓
# # R_i = 0.2
# # R_j = 0.3
# # Safe but inefficient - left 0.5 unused!
#
# # The dilemma: 
# # - Too modest → waste resources
# # - Too greedy → risk getting nothing
# # - Must anticipate partner's demand to find sweet spot


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

# some example situations for public goods -> 

# params = {'multiplier': 1.8, 'endowment': 1.0}

# # Both contribute FULLY (1.0 each)
# R_i, R_j = payoff_public_goods(1.0, 1.0, params)
# # Total contribution = 1.0 + 1.0 = 2.0
# # Public good = 1.8 × 2.0 = 3.6
# # R_i = 3.6/2 + (1.0 - 1.0) = 1.8 + 0.0 = 1.8
# # R_j = 3.6/2 + (1.0 - 1.0) = 1.8 + 0.0 = 1.8
# # BEST COLLECTIVE: Both do great when both contribute!

# # I FREERIDE (0.0), you contribute FULLY (1.0)
# R_i, R_j = payoff_public_goods(0.0, 1.0, params)
# # Total contribution = 0.0 + 1.0 = 1.0
# # Public good = 1.8 × 1.0 = 1.8
# # R_i = 1.8/2 + (1.0 - 0.0) = 0.9 + 1.0 = 1.9 ← I do BEST!
# # R_j = 1.8/2 + (1.0 - 1.0) = 0.9 + 0.0 = 0.9 ← You do worse
# # I win by free-riding! I keep my endowment AND get half the public good.

# # You FREERIDE (0.0), I contribute FULLY (1.0)
# R_i, R_j = payoff_public_goods(1.0, 0.0, params)
# # R_i = 0.9 + 0.0 = 0.9 ← I'm the sucker
# # R_j = 0.9 + 1.0 = 1.9 ← You exploit me
# # Symmetric to above - contributor gets exploited.

# # Both FREERIDE (0.0 each)
# R_i, R_j = payoff_public_goods(0.0, 0.0, params)
# # Total contribution = 0.0 + 0.0 = 0.0
# # Public good = 1.8 × 0.0 = 0.0
# # R_i = 0.0/2 + (1.0 - 0.0) = 0.0 + 1.0 = 1.0
# # R_j = 0.0/2 + (1.0 - 0.0) = 0.0 + 1.0 = 1.0
# # We both keep our endowment, but no multiplication benefit.

# # Partial contribution (0.5 each)
# R_i, R_j = payoff_public_goods(0.5, 0.5, params)
# # Total contribution = 0.5 + 0.5 = 1.0
# # Public good = 1.8 × 1.0 = 1.8
# # R_i = 1.8/2 + (1.0 - 0.5) = 0.9 + 0.5 = 1.4
# # R_j = 1.8/2 + (1.0 - 0.5) = 0.9 + 0.5 = 1.4
# # Better than no contribution, worse than full mutual contribution.
#
# # The dilemma:
# # Individual incentive: Contribute 0, let partner contribute
# #   → I get 1.9 (best individual outcome)
# # Collective optimum: Both contribute 1.0
# #   → Each gets 1.8 (best mutual outcome)
# # Mutual defection: Both contribute 0
# #   → Each gets 1.0 (safe but suboptimal)
#
# # This only works as a dilemma if multiplier < n_players
# # With multiplier=1.8 and n=2, freeriding (1.9) > mutual cooperation (1.8)
# # so there's individual incentive to defect!


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
