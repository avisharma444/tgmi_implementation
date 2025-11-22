"""
TGMI Agent Implementation.

This module implements the Trust-Gated Moral Inference agent from the paper:
- Partner state management (beliefs, trust, confidence, reservation utility)
- Moral utility computation (trust-confidence-gated)
- Virtual bargaining for joint action selection
- Trust update (leaky integrator of compliance)
- Belief update (CK-ToM: self-anchoring + fairness evidence)
- Confidence and κ computation
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional
import numpy as np

from mgg import Game
from config import Hyperparameters


@dataclass
class PartnerState:
    """
    State that agent i maintains about partner j.
    
    Attributes:
        B_hat: Belief over partner's moral norms, shape (|F|,) - probability distribution
        tau: Trust τ_i ∈ [0, 1]
        c: Confidence c_i ∈ [0, 1] (inverse of entropy of B_hat)
        kappa: κ_i = τ_i × c_i (trust-confidence gate)
        d: Reservation utility d_i (fairness-only utility from last VB action)
    """
    B_hat: np.ndarray
    tau: float
    c: float
    kappa: float
    d: float


class TGMI_Agent:
    """
    Trust-Gated Moral Inference (TGMI) agent.
    
    Key mechanisms:
    1. Moral utility: Weighted combination of self prior B_i and partner belief B_hat
    2. Virtual bargaining: Nash-product-style joint action selection
    3. Trust evolution: Leaky integrator of fairness compliance
    4. Belief update: CK-ToM with trust-modulated self-anchoring
    5. Confidence: Based on entropy of beliefs
    """
    
    def __init__(self, 
                 agent_id: int, 
                 moral_prior: np.ndarray, 
                 hyperparams: Hyperparameters, 
                 rng: np.random.Generator):
        """
        Initialize TGMI agent.
        
        Args:
            agent_id: Unique identifier for this agent
            moral_prior: Intrinsic moral prior B_i over fairness principles
            hyperparams: Hyperparameters (η, λ_dev, α, β, γ, etc.)
            rng: Random number generator
        """
        self.id = agent_id
        self.B = moral_prior.copy()  # Intrinsic moral prior B_i(φ)
        self.hyper = hyperparams
        self.rng = rng
        
        # Partner states: partner_id -> PartnerState
        self.partners: Dict[int, PartnerState] = {}
        
        # For logging
        self.log: list = []
    
    def _init_partner_state(self, partner_id: int) -> None:
        """
        Initialize state for a new partner.
        
        When i first meets j:
        - Initialize belief B_hat from Dirichlet(1, 1, 1) (uniform prior)
        - Compute initial confidence from entropy
        - Set initial trust to τ0
        - Compute κ = τ × c
        - Initialize reservation utility to 0
        """
        num_phi = len(self.B)
        
        # Uniform prior over partner's moral norms
        B_hat = self.rng.dirichlet(np.ones(num_phi))
        
        # Confidence = 1 - normalized entropy
        H = -np.sum(B_hat * np.log(B_hat + 1e-12))
        c = 1.0 - H / np.log(num_phi)
        
        # Initial trust
        tau = self.hyper.tau0
        
        # Trust-confidence gate
        kappa = tau * c
        
        # Initial reservation utility
        d = 0.0
        
        self.partners[partner_id] = PartnerState(
            B_hat=B_hat,
            tau=tau,
            c=c,
            kappa=kappa,
            d=d
        )
    
    def ensure_partner_initialized(self, partner_id: int) -> None:
        """Ensure partner state exists; initialize if needed."""
        if partner_id not in self.partners:
            self._init_partner_state(partner_id)
    
    def moral_utility(self, partner_id: int, F_vec: np.ndarray) -> float:
        """
        Compute moral utility U_i(a) for a joint action.
        
        From Eq. 2:
        U_i(a) = Σ_φ [(1-κ_i)B_i(φ) + κ_i B̂_i→j(φ)] F_φ(a)
        
        Args:
            partner_id: ID of partner j
            F_vec: Fairness values at joint action, shape (3,)
                   [F_max_sum(a), F_equal_split(a), F_rawls(a)]
        
        Returns:
            Moral utility U_i(a) ∈ [0, 1]
        """
        ps = self.partners[partner_id]
        
        # Trust-confidence-gated combination of self prior and partner belief
        w = (1.0 - ps.kappa) * self.B + ps.kappa * ps.B_hat
        
        # Weighted sum over fairness principles
        return float(np.dot(w, F_vec))
    
    def fairness_utility(self, F_vec: np.ndarray) -> float:
        """
        Compute fairness-only utility U^F_i(a).
        
        U^F_i(a) = Σ_φ B_i(φ) F_φ(a)
        
        Uses only self moral prior (not gated by trust/confidence).
        
        Args:
            F_vec: Fairness values at joint action, shape (3,)
        
        Returns:
            Fairness utility U^F_i(a) ∈ [0, 1]
        """
        return float(np.dot(self.B, F_vec))


def virtual_bargain(game: Game, 
                   agent_i: TGMI_Agent, 
                   agent_j: TGMI_Agent,
                   gamma: float) -> Tuple[int, int, float, float]:
    """
    Virtual bargaining: select joint action via Nash product.
    
    From Eq. 3 and Algorithm 1:
    (a_i^VB, a_j^VB) = argmax_{a_i,a_j} (U_i(a) - d_i)_+^γ (U_j(a) - d_j)_+^(1-γ)
    
    where (x)_+ = max(x, 0).
    
    Args:
        game: Game object with action spaces and fairness functions
        agent_i: First agent
        agent_j: Second agent
        gamma: Bargaining power for agent i (0.5 = symmetric)
    
    Returns:
        (idx_ai_vb, idx_aj_vb, U_i_vb, U_j_vb):
            - idx_ai_vb: Index of agent i's VB action in action_space_i
            - idx_aj_vb: Index of agent j's VB action in action_space_j
            - U_i_vb: Agent i's moral utility at VB joint action
            - U_j_vb: Agent j's moral utility at VB joint action
    """
    A_i = game.action_space_i
    A_j = game.action_space_j
    
    ps_i = agent_i.partners[agent_j.id]
    ps_j = agent_j.partners[agent_i.id]
    
    best_value = -np.inf
    best_result = None
    
    # Exhaustive search over discrete action space
    for idx_i in range(len(A_i)):
        for idx_j in range(len(A_j)):
            # Fairness vector at this joint action
            F_vec = np.array([
                game.F["max_sum"][idx_i, idx_j],
                game.F["equal_split"][idx_i, idx_j],
                game.F["rawls"][idx_i, idx_j],
            ])
            
            # Moral utilities
            U_i = agent_i.moral_utility(agent_j.id, F_vec)
            U_j = agent_j.moral_utility(agent_i.id, F_vec)
            
            # Gains over reservation utilities
            gain_i = max(U_i - ps_i.d, 0.0)
            gain_j = max(U_j - ps_j.d, 0.0)
            
            # Nash product with asymmetric bargaining power
            value = (gain_i ** gamma) * (gain_j ** (1.0 - gamma))
            
            if value > best_value:
                best_value = value
                best_result = (idx_i, idx_j, U_i, U_j)
    
    return best_result


def apply_action_noise(idx_ai: int, idx_aj: int, 
                       game: Game, 
                       epsilon_a: float, 
                       rng: np.random.Generator) -> Tuple[int, int]:
    """
    Apply action noise: with probability ε_a, randomize each action.
    
    Args:
        idx_ai, idx_aj: Intended action indices
        game: Game object
        epsilon_a: Action error probability
        rng: Random number generator
    
    Returns:
        (idx_ai_real, idx_aj_real): Realized action indices after noise
    """
    n_i = len(game.action_space_i)
    n_j = len(game.action_space_j)
    
    # Agent i's action
    if rng.random() < epsilon_a:
        idx_ai_real = rng.integers(0, n_i)
    else:
        idx_ai_real = idx_ai
    
    # Agent j's action
    if rng.random() < epsilon_a:
        idx_aj_real = rng.integers(0, n_j)
    else:
        idx_aj_real = idx_aj
    
    return idx_ai_real, idx_aj_real


def compute_fairness_deviation(agent: TGMI_Agent, 
                               partner_id: int,
                               game: Game, 
                               idx_ai_vb: int, 
                               idx_aj_vb: int) -> Tuple[float, float]:
    """
    Compute fairness deviation d_i caused by partner's action.
    
    From Algorithm 1:
    d_i = max_{a_j'} U^F_i(a_i^VB, a_j') - U^F_i(a_i^VB, a_j^VB)
    
    This measures the fairness shortfall: how much better could i have done
    (in terms of fairness utility) if j had chosen a different action, 
    given i's VB action.
    
    Args:
        agent: Agent i
        partner_id: Partner j's ID
        game: Game object
        idx_ai_vb: Agent i's VB action index
        idx_aj_vb: Agent j's VB action index (realized)
    
    Returns:
        (d_i, UF_i_vb):
            - d_i: Fairness deviation
            - UF_i_vb: Fairness utility at VB joint action
    """
    A_j = game.action_space_j
    
    # Fairness utility under realized joint action
    F_vec_vb = np.array([
        game.F["max_sum"][idx_ai_vb, idx_aj_vb],
        game.F["equal_split"][idx_ai_vb, idx_aj_vb],
        game.F["rawls"][idx_ai_vb, idx_aj_vb],
    ])
    UF_i_vb = agent.fairness_utility(F_vec_vb)
    
    # Best fairness i could have gotten from j, given i's action
    best_UF = -np.inf
    for idx_aj_prime in range(len(A_j)):
        F_vec_prime = np.array([
            game.F["max_sum"][idx_ai_vb, idx_aj_prime],
            game.F["equal_split"][idx_ai_vb, idx_aj_prime],
            game.F["rawls"][idx_ai_vb, idx_aj_prime],
        ])
        UF_candidate = agent.fairness_utility(F_vec_prime)
        if UF_candidate > best_UF:
            best_UF = UF_candidate
    
    # Deviation = shortfall caused by partner's action
    d_i = best_UF - UF_i_vb
    
    return d_i, UF_i_vb


def update_trust(agent: TGMI_Agent, partner_id: int, d_i: float) -> None:
    """
    Update trust based on fairness compliance.
    
    From the text:
    s_i = exp(-λ_dev × d_i)  (compliance signal)
    τ_i ← (1-η)τ_i + η s_i  (leaky integrator)
    
    High deviation → low compliance → trust decreases
    Low deviation → high compliance → trust increases
    
    Args:
        agent: Agent i
        partner_id: Partner j's ID
        d_i: Fairness deviation
    """
    ps = agent.partners[partner_id]
    
    # Compliance signal: exponentially decreases with deviation
    s_i = np.exp(-agent.hyper.lambda_dev * d_i)
    
    # Leaky integrator update
    ps.tau = (1.0 - agent.hyper.eta) * ps.tau + agent.hyper.eta * s_i
    
    # Clamp to [0, 1] for numerical stability
    ps.tau = np.clip(ps.tau, 0.0, 1.0)


def update_belief(agent: TGMI_Agent, 
                 partner_id: int,
                 game: Game, 
                 idx_ai_vb: int, 
                 idx_aj_vb: int) -> None:
    """
    Update belief about partner's moral norms using CK-ToM.
    
    From Eq. 4:
    B̂_i→j^(t+1)(φ) ∝ B̂_i→j^(t)(φ) × 
                      [exp(β F_φ(a^t))]^(1-ατ_i^t) × 
                      [B_i(φ)]^(ατ_i^t)
    
    Interpretation:
    - When trust is LOW (τ ≈ 0): Weight evidence more (1-ατ ≈ 1), ignore self prior (ατ ≈ 0)
    - When trust is HIGH (τ ≈ 1): Weight self prior more (ατ ≈ α), discount evidence
    
    This implements "self-anchoring": high trust → assume partner is like me.
    
    Args:
        agent: Agent i
        partner_id: Partner j's ID
        game: Game object
        idx_ai_vb, idx_aj_vb: Joint action indices
    """
    ps = agent.partners[partner_id]
    
    # Fairness values at joint action
    F_vec = np.array([
        game.F["max_sum"][idx_ai_vb, idx_aj_vb],
        game.F["equal_split"][idx_ai_vb, idx_aj_vb],
        game.F["rawls"][idx_ai_vb, idx_aj_vb],
    ])
    
    old_B_hat = ps.B_hat
    tau = ps.tau
    alpha = agent.hyper.alpha
    beta = agent.hyper.beta
    
    # Exponents for fairness evidence and self-anchoring
    w_evidence = 1.0 - alpha * tau
    w_prior = alpha * tau
    
    # Bayesian-style update with trust-modulated weights
    likelihood = np.exp(beta * F_vec)
    unnorm = old_B_hat * (likelihood ** w_evidence) * (agent.B ** w_prior)
    
    # Normalize to probability distribution
    new_B_hat = unnorm / (unnorm.sum() + 1e-12)
    
    ps.B_hat = new_B_hat


def update_confidence_and_kappa(agent: TGMI_Agent, partner_id: int) -> None:
    """
    Update confidence and trust-confidence gate κ.
    
    From the text:
    c_i = 1 - H(B̂_i→j) / log|F|  (confidence = 1 - normalized entropy)
    κ_i = τ_i × c_i  (trust-confidence gate)
    
    High confidence (low entropy) → beliefs are peaked → κ sensitive to trust
    Low confidence (high entropy) → beliefs are diffuse → κ low regardless of trust
    
    Args:
        agent: Agent i
        partner_id: Partner j's ID
    """
    ps = agent.partners[partner_id]
    B_hat = ps.B_hat
    num_phi = len(B_hat)
    
    # Shannon entropy
    H = -np.sum(B_hat * np.log(B_hat + 1e-12))
    
    # Confidence = 1 - normalized entropy
    ps.c = 1.0 - H / np.log(num_phi)
    
    # Trust-confidence gate
    ps.kappa = ps.tau * ps.c


def update_reservation_utility(agent: TGMI_Agent, 
                               partner_id: int,
                               UF_i_vb: float) -> None:
    """
    Update reservation utility from fairness-only utility at VB action.
    
    From Algorithm 1:
    d_i ← U^F_i(a^VB)
    
    This sets the baseline for future virtual bargaining negotiations.
    
    Args:
        agent: Agent i
        partner_id: Partner j's ID
        UF_i_vb: Fairness utility at VB joint action
    """
    ps = agent.partners[partner_id]
    ps.d = UF_i_vb


def tgmi_round(game: Game, 
               agent_i: TGMI_Agent, 
               agent_j: TGMI_Agent) -> Tuple[float, float, Dict]:
    """
    Execute one round of TGMI interaction between two agents.
    
    This implements Algorithm 1 from the paper:
    1. Virtual bargaining to select joint action
    2. Apply action noise (optional)
    3. Compute payoffs and fairness utilities
    4. Compute fairness deviations
    5. Update trust
    6. Update beliefs (CK-ToM)
    7. Update confidence and κ
    8. Update reservation utilities
    
    Args:
        game: Game object sampled from MGG
        agent_i: First TGMI agent
        agent_j: Second TGMI agent
    
    Returns:
        (R_i, R_j, info_dict):
            - R_i, R_j: Payoffs for both agents
            - info_dict: Dictionary with additional info for logging
    """
    # 1. Ensure partner states exist
    agent_i.ensure_partner_initialized(agent_j.id)
    agent_j.ensure_partner_initialized(agent_i.id)
    
    # 2. Virtual bargaining joint action
    idx_ai_vb, idx_aj_vb, U_i_vb, U_j_vb = virtual_bargain(
        game, agent_i, agent_j, agent_i.hyper.gamma
    )
    
    # 3. Apply action noise (if any)
    idx_ai_real, idx_aj_real = apply_action_noise(
        idx_ai_vb, idx_aj_vb, game, agent_i.hyper.epsilon_a, agent_i.rng
    )
    
    # 4. Get realized payoffs
    R_i = game.R_i[idx_ai_real, idx_aj_real]
    R_j = game.R_j[idx_ai_real, idx_aj_real]
    
    # 5. Compute fairness utilities and deviations
    d_i, UF_i_vb = compute_fairness_deviation(agent_i, agent_j.id, game, idx_ai_vb, idx_aj_vb)
    d_j, UF_j_vb = compute_fairness_deviation(agent_j, agent_i.id, game, idx_aj_vb, idx_ai_vb)
    
    # 6. Update trust
    update_trust(agent_i, agent_j.id, d_i)
    update_trust(agent_j, agent_i.id, d_j)
    
    # 7. Update beliefs (CK-ToM)
    update_belief(agent_i, agent_j.id, game, idx_ai_vb, idx_aj_vb)
    update_belief(agent_j, agent_i.id, game, idx_aj_vb, idx_ai_vb)
    
    # 8. Update confidence and κ
    update_confidence_and_kappa(agent_i, agent_j.id)
    update_confidence_and_kappa(agent_j, agent_i.id)
    
    # 9. Update reservation utilities
    update_reservation_utility(agent_i, agent_j.id, UF_i_vb)
    update_reservation_utility(agent_j, agent_i.id, UF_j_vb)
    
    # Collect info for logging
    info = {
        'idx_ai_vb': idx_ai_vb,
        'idx_aj_vb': idx_aj_vb,
        'idx_ai_real': idx_ai_real,
        'idx_aj_real': idx_aj_real,
        'U_i_vb': U_i_vb,
        'U_j_vb': U_j_vb,
        'UF_i_vb': UF_i_vb,
        'UF_j_vb': UF_j_vb,
        'd_i': d_i,
        'd_j': d_j,
        'archetype': game.archetype.name,
    }
    
    return R_i, R_j, info
