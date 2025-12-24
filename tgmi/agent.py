import numpy as np
from enum import Enum
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
import sys
import os

_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
sys.path.insert(0, _parent_dir)

from mgg.generator import Game, MoralGameGenerator, compute_fairness_functions


class FairnessPrinciple(Enum):
    MAX_SUM = 0      
    EQUAL_SPLIT = 1   
    RAWLS = 2         


FAIRNESS_KEYS = {
    FairnessPrinciple.MAX_SUM: "max_sum",
    FairnessPrinciple.EQUAL_SPLIT: "equal_split",
    FairnessPrinciple.RAWLS: "rawls",
}


def entropy(p: np.ndarray) -> float:
    p = np.asarray(p)
    p = p[p > 0]
    return -np.sum(p * np.log(p))


def normalized(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p)
    s = p.sum()
    if s > 0:
        return p / s
    else:
        return np.ones_like(p) / len(p)


@dataclass
class TGMIConfig:
    theta_0: float = 0.5     # Initial trust τ_0
    phi: float = 0.1         # Trust learning rate η (called phi in code)
    alpha: float = 0.5       # Self-anchoring weight α (DEPRECATED - not used in new update)
    beta: float = 1.0        # Softmax temperature β in likelihood
    xi_dev: float = 3.0      # Sensitivity to fairness deviation λ_dev
    gamma_bargain: float = 0.5  # Bargaining asymmetry γ (for Nash bargaining)
    gamma_mixture: float = 0.1  # Epistemic vigilance γ (for restless mixture)
    n_principles: int = 3
    epsilon: float = 0.0     # Action error (trembling hand) ε_a
    eta: float = 0.0         # Observation error ε_p


class TGMIAgent:
    def __init__(
        self,
        moral_prior: np.ndarray,
        config: TGMIConfig = None,
        agent_id: str = None,
        rng: np.random.Generator = None,
    ):
        self.config = config if config is not None else TGMIConfig()
        self.agent_id = agent_id
        self.rng = rng if rng is not None else np.random.default_rng()
        
        # Intrinsic moral prior B_i (does not change)
        self.B_i = normalized(np.asarray(moral_prior))
        self.n_principles = len(self.B_i)
        
        self._initialize_state()
        
        self.history = {
            'theta': [self.theta],
            'c': [self.c],
            'varpi': [self.varpi],
            'B_hat': [self.B_hat.copy()],
            'd_i': [self.d_i],
            'actions': [],
            'vb_outcomes': [],
        }
    
    def _initialize_state(self):
        K = self.n_principles 
        
        # Line 1: Initialize belief over partner's norms from Dirichlet(1,...,1)
        self.B_hat = self.rng.dirichlet(np.ones(K))
        
        # Line 2: Initialize trust
        self.theta = self.config.theta_0
        
        # Line 3: Initialize confidence from entropy of belief
        # c = 1 - H(B_hat) / log(K)
        max_entropy = np.log(K)
        self.c = 1.0 - entropy(self.B_hat) / max_entropy if max_entropy > 0 else 0.0
        
        # Line 4: Initialize cooperation weight
        self.varpi = self.theta * self.c # ki = ti * ci wali line
        
        # Line 5: Initialize reservation utilities
        self.d_i = 0.0
        self.d_j = 0.0
    
    def reset(self):
        self._initialize_state()
        self.history = {
            'theta': [self.theta],
            'c': [self.c],
            'varpi': [self.varpi],
            'B_hat': [self.B_hat.copy()],
            'd_i': [self.d_i],
            'actions': [],
            'vb_outcomes': [],
        }
    
    def compute_moral_utility(
        self,
        F_table: Dict[str, np.ndarray],
        use_partner_model: bool = False,
    ) -> np.ndarray:
        first_key = list(F_table.keys())[0]
        n_ai, n_aj = F_table[first_key].shape
        
        if use_partner_model: # line 8 ki implementation for U_j_hat
            U = np.zeros((n_aj, n_ai))
            for idx, principle in enumerate(FairnessPrinciple):
                key = FAIRNESS_KEYS[principle]
                if key in F_table:
                    F_omega = F_table[key].T
                    U += self.B_hat[idx] * F_omega
        else:
            U = np.zeros((n_ai, n_aj))
            for idx, principle in enumerate(FairnessPrinciple):
                key = FAIRNESS_KEYS[principle]
                if key in F_table:
                    F_omega = F_table[key]
                    weight = (1 - self.varpi) * self.B_i[idx] + self.varpi * self.B_hat[idx]
                    U += weight * F_omega
        
        return U
    
    def compute_fairness_utility(
        self,
        F_table: Dict[str, np.ndarray],
    ) -> np.ndarray:
        first_key = list(F_table.keys())[0]
        n_ai, n_aj = F_table[first_key].shape
        U_F = np.zeros((n_ai, n_aj))
        
        for idx, principle in enumerate(FairnessPrinciple):
            key = FAIRNESS_KEYS[principle]
            if key in F_table:
                F_omega = F_table[key]
                U_F += self.B_i[idx] * F_omega
        
        return U_F
    
    def virtual_bargain(
        self,
        U_i: np.ndarray,
        U_j_hat: np.ndarray,
    ) -> Tuple[int, int]:
        n_ai, n_aj = U_i.shape
        gamma = self.config.gamma_bargain  # Bargaining asymmetry
        
        best_val = -np.inf
        best_pair = (0, 0)
        
        for a_i in range(n_ai):
            for a_j in range(n_aj):
                surplus_i = max(U_i[a_i, a_j] - self.d_i, 0.0)
                surplus_j = max(U_j_hat[a_j, a_i] - self.d_j, 0.0)
                
                # Nash product: (U_i - d_i)^γ * (U_j - d_j)^(1-γ)
                if surplus_i > 0 and surplus_j > 0:
                    nash_product = (surplus_i ** gamma) * (surplus_j ** (1 - gamma))
                elif surplus_i > 0:
                    # One-sided: partner gets no surplus
                    nash_product = (surplus_i ** gamma) * 1e-10
                elif surplus_j > 0:
                    # One-sided: I get no surplus
                    nash_product = 1e-10 * (surplus_j ** (1 - gamma))
                else:
                    nash_product = 0.0
                
                if nash_product > best_val:
                    best_val = nash_product
                    best_pair = (a_i, a_j)
        
        return best_pair
    
    def select_action(
        self,
        game: Game,
    ) -> Tuple[int, dict]:
        F_table = game.F
        
        U_i = self.compute_moral_utility(F_table, use_partner_model=False)
        U_j_hat = self.compute_moral_utility(F_table, use_partner_model=True)
        a_i_VB, a_j_VB = self.virtual_bargain(U_i, U_j_hat)
        
        info = {
            'a_i_VB': a_i_VB,
            'a_j_VB': a_j_VB,
            'U_i': U_i,
            'U_j_hat': U_j_hat,
        }
        
        return a_i_VB, info
    
    def update(
        self,
        game: Game,
        a_i_VB: int,
        a_j_actual: int,
        U_i: np.ndarray,
        U_j_hat: np.ndarray,
    ):
        F_table = game.F
        
        U_F = self.compute_fairness_utility(F_table)
        # Line 10: Compute fairness deviation
        best_fairness = U_F[a_i_VB, :].max()
        realized_fairness = U_F[a_i_VB, a_j_actual]
        d_dev = best_fairness - realized_fairness
        
        # Line 11: Compliance signal
        s_t = np.exp(-self.config.xi_dev * d_dev)
        
        # Line 12: Trust update
        self.theta = (1 - self.config.phi) * self.theta + self.config.phi * s_t
        
        # Line 13: NEW Likelihood-based belief update
        # Compute likelihood L(a_j^(t) | φ) for each principle φ
        likelihoods = np.zeros(self.n_principles)
        n_aj = len(game.action_space_j)
        
        for idx, principle in enumerate(FairnessPrinciple):
            key = FAIRNESS_KEYS[principle]
            if key in F_table:
                # Compute normalizer Z_φ(a_i^(t)) = sum over a_j of exp(β * F_φ(a_j, a_i^(t)))
                # NOTE: Using REALIZED action a_i_VB (actual action taken by agent i)
                Z_phi = 0.0
                for a_j in range(n_aj):
                    F_phi_aj = F_table[key][a_i_VB, a_j]
                    Z_phi += np.exp(self.config.beta * F_phi_aj)
                
                # Fairness utility for the REALIZED outcome (a_j_actual, a_i_VB)
                # CRITICAL FIX: Use realized action, not VB action
                F_phi_realized = F_table[key][a_i_VB, a_j_actual]
                
                # Trust-gated likelihood:
                # L(a_j^(t) | φ) = τ * [exp(β*F_φ(a_j^(t), a_i^(t))) / Z_φ] + (1-τ) * [1/|A_j|]
                informative_prob = np.exp(self.config.beta * F_phi_realized) / Z_phi if Z_phi > 0 else 0.0
                uniform_prob = 1.0 / n_aj
                likelihoods[idx] = self.theta * informative_prob + (1 - self.theta) * uniform_prob
        
        # Bayesian update: B^Bayes(φ) = B̂(φ) * L(a_j^(t) | φ) / sum
        B_bayes_unnorm = self.B_hat * likelihoods
        B_bayes_sum = B_bayes_unnorm.sum()
        if B_bayes_sum > 0:
            B_bayes = B_bayes_unnorm / B_bayes_sum
        else:
            B_bayes = self.B_hat.copy()  # Fallback if all likelihoods are zero
        
        # Line 14: Apply restless mixture with epistemic vigilance γ
        # B̂^(t+1)(φ) = (1-γ) * B^Bayes(φ) + γ / |F|
        gamma_mix = self.config.gamma_mixture
        uniform_prior = np.ones(self.n_principles) / self.n_principles
        self.B_hat = (1 - gamma_mix) * B_bayes + gamma_mix * uniform_prior
        
        # Line 15: Update confidence
        K = self.n_principles
        max_entropy = np.log(K)
        self.c = 1.0 - entropy(self.B_hat) / max_entropy if max_entropy > 0 else 0.0
        
        # Line 16: Update cooperation weight
        self.varpi = self.theta * self.c
        
        # Line 17: Update reservation utility
        self.d_i = U_F[a_i_VB, a_j_actual]
        
        self.history['theta'].append(self.theta)
        self.history['c'].append(self.c)
        self.history['varpi'].append(self.varpi)
        self.history['B_hat'].append(self.B_hat.copy())
        self.history['d_i'].append(self.d_i)
        self.history['actions'].append((a_i_VB, a_j_actual))
    
    def select_action_with_noise(
        self,
        game: Game,
    ) -> Tuple[int, dict]:
        a_i_VB, info = self.select_action(game)
        
        if self.rng.random() < self.config.epsilon:
            n_actions = len(game.action_space_i)
            a_i_actual = self.rng.integers(0, n_actions)
            info['trembled'] = True
            info['intended_action'] = a_i_VB
        else:
            a_i_actual = a_i_VB
            info['trembled'] = False
        
        return a_i_actual, info
    
    def observe_partner_action(
        self,
        a_j_actual: int,
        game: Game,
    ) -> int:
        if self.rng.random() < self.config.eta:
            n_actions = len(game.action_space_j)
            a_j_observed = self.rng.integers(0, n_actions)
        else:
            a_j_observed = a_j_actual
        
        return a_j_observed
    
    def play_round(
        self,
        game: Game,
        partner_action: int = None,
    ) -> Tuple[int, int, dict]:
        my_action, info = self.select_action_with_noise(game)
        
        if partner_action is None:
            partner_action = info['a_j_VB']
        
        partner_action_observed = self.observe_partner_action(partner_action, game)
        info['partner_action_actual'] = partner_action
        info['partner_action_observed'] = partner_action_observed
        
        self.update(
            game=game,
            a_i_VB=info['a_i_VB'],
            a_j_actual=partner_action_observed,
            U_i=info['U_i'],
            U_j_hat=info['U_j_hat'],
        )
        
        info['partner_action'] = partner_action
        self.history['vb_outcomes'].append((my_action, partner_action))
        
        return my_action, partner_action, info


def sample_moral_prior(rng: np.random.Generator = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    return rng.dirichlet(np.ones(3))


def create_tgmi_agent(
    moral_prior: np.ndarray = None,
    theta_0: float = 0.5,
    phi: float = 0.1,
    alpha: float = 0.5,
    beta: float = 1.0,
    xi_dev: float = 3.0,
    gamma_bargain: float = 0.5,
    gamma_mixture: float = 0.1,
    epsilon: float = 0.0,
    eta: float = 0.0,
    agent_id: str = None,
    rng: np.random.Generator = None,
) -> TGMIAgent:
    if rng is None:
        rng = np.random.default_rng()
    
    if moral_prior is None:
        moral_prior = sample_moral_prior(rng)
    
    config = TGMIConfig(
        theta_0=theta_0,
        phi=phi,
        alpha=alpha,
        beta=beta,
        xi_dev=xi_dev,
        gamma_bargain=gamma_bargain,
        gamma_mixture=gamma_mixture,
        epsilon=epsilon,
        eta=eta,
    )
    
    return TGMIAgent(
        moral_prior=moral_prior,
        config=config,
        agent_id=agent_id,
        rng=rng,
    )

@dataclass
class ConvergenceConfig:
    prob_threshold: float = 0.00000001      # Max change in dominant type's probability
    window_size: int = 5              # Consecutive rounds to check stability
    max_rounds: int = 500             # Safety limit
    min_rounds: int = 10              # Minimum rounds before checking convergence
    check_both_agents: bool = False   # True = both must converge, False = either


def get_dominant_type_info(B_hat: np.ndarray) -> Tuple[int, float]:
    dominant_type = np.argmax(B_hat)
    prob = B_hat[dominant_type]
    return int(dominant_type), float(prob)


def check_belief_convergence(
    history: Dict,
    config: ConvergenceConfig,
) -> Tuple[bool, str]:
    B_hat_history = history['B_hat']
    
    if len(B_hat_history) < config.window_size + 1:
        return False, "insufficient_rounds"
    
    recent_beliefs = B_hat_history[-(config.window_size + 1):]
    
    dominant_types = []
    dominant_probs = []
    for B in recent_beliefs:
        dtype, prob = get_dominant_type_info(B)
        dominant_types.append(dtype)
        dominant_probs.append(prob)
    
    if len(set(dominant_types)) != 1:
        return False, "dominant_type_unstable"
    
    max_prob_change = max(
        abs(dominant_probs[i] - dominant_probs[i-1]) 
        for i in range(1, len(dominant_probs))
    )
    
    if max_prob_change < config.prob_threshold:
        dominant_type_name = list(FairnessPrinciple)[dominant_types[0]].name
        return True, f"converged_to_{dominant_type_name}_p={dominant_probs[-1]:.3f}"
    
    return False, "probability_unstable"


def simulate_interaction(
    agent_i: TGMIAgent,
    agent_j: TGMIAgent,
    game: Game,
    n_rounds: int = None,
    conv_config: ConvergenceConfig = None,
) -> Dict:
    if conv_config is None:
        conv_config = ConvergenceConfig()
    
    use_fixed_rounds = n_rounds is not None
    max_t = n_rounds if use_fixed_rounds else conv_config.max_rounds
    
    agent_i.reset()
    agent_j.reset()
    
    actions_i = []
    actions_j = []
    payoffs_i = []
    payoffs_j = []
    trembles_i = []
    trembles_j = []
    
    converged = False
    convergence_reason = "max_rounds" if not use_fixed_rounds else "fixed_rounds"
    convergence_round = max_t
    who_converged = None
    
    for t in range(max_t):
        a_i, info_i = agent_i.select_action_with_noise(game)
        a_j, info_j = agent_j.select_action_with_noise(game)
        
        a_j_observed_by_i = agent_i.observe_partner_action(a_j, game)
        a_i_observed_by_j = agent_j.observe_partner_action(a_i, game)
        
        agent_i.update(
            game=game,
            a_i_VB=info_i['a_i_VB'],
            a_j_actual=a_j_observed_by_i,
            U_i=info_i['U_i'],
            U_j_hat=info_i['U_j_hat'],
        )
        agent_j.update(
            game=game,
            a_i_VB=info_j['a_i_VB'],
            a_j_actual=a_i_observed_by_j,
            U_i=info_j['U_i'],
            U_j_hat=info_j['U_j_hat'],
        )
        
        actions_i.append(a_i)
        actions_j.append(a_j)
        payoffs_i.append(game.R_i[a_i, a_j])
        payoffs_j.append(game.R_j[a_i, a_j])
        trembles_i.append(info_i.get('trembled', False))
        trembles_j.append(info_j.get('trembled', False))
        
        if not use_fixed_rounds and t >= conv_config.min_rounds:
            conv_i, reason_i = check_belief_convergence(agent_i.history, conv_config)
            conv_j, reason_j = check_belief_convergence(agent_j.history, conv_config)
            
            if conv_config.check_both_agents:
                if conv_i and conv_j:
                    converged = True
                    convergence_reason = f"both: i={reason_i}, j={reason_j}"
                    convergence_round = t + 1
                    who_converged = "both"
                    break
            else:
                if conv_i:
                    converged = True
                    convergence_reason = reason_i
                    convergence_round = t + 1
                    who_converged = "agent_i"
                    break
                elif conv_j:
                    converged = True
                    convergence_reason = reason_j
                    convergence_round = t + 1
                    who_converged = "agent_j"
                    break
    
    final_dominant_i, final_prob_i = get_dominant_type_info(agent_i.B_hat)
    final_dominant_j, final_prob_j = get_dominant_type_info(agent_j.B_hat)
    
    return {
        'actions_i': np.array(actions_i),
        'actions_j': np.array(actions_j),
        'payoffs_i': np.array(payoffs_i),
        'payoffs_j': np.array(payoffs_j),
        'history_i': agent_i.history,
        'history_j': agent_j.history,
        'mean_payoff_i': np.mean(payoffs_i),
        'mean_payoff_j': np.mean(payoffs_j),
        'total_payoff': np.sum(payoffs_i) + np.sum(payoffs_j),
        'tremble_rate_i': np.mean(trembles_i),
        'tremble_rate_j': np.mean(trembles_j),
        'n_rounds': len(actions_i),
        'converged': converged,
        'convergence_reason': convergence_reason,
        'convergence_round': convergence_round,
        'who_converged': who_converged,
        'final_dominant_type_i': list(FairnessPrinciple)[final_dominant_i].name,
        'final_dominant_prob_i': final_prob_i,
        'final_dominant_type_j': list(FairnessPrinciple)[final_dominant_j].name,
        'final_dominant_prob_j': final_prob_j,
    }


def simulate_with_fixed_partner(
    agent: TGMIAgent,
    partner_strategy: Callable[[Game, int], int],
    game: Game,
    n_rounds: int = None,
    conv_config: ConvergenceConfig = None,
) -> Dict:
    if conv_config is None:
        conv_config = ConvergenceConfig()
    
    use_fixed_rounds = n_rounds is not None
    max_t = n_rounds if use_fixed_rounds else conv_config.max_rounds
    
    agent.reset()
    
    actions_agent = []
    actions_partner = []
    payoffs_agent = []
    payoffs_partner = []
    trembles = []
    
    converged = False
    convergence_reason = "max_rounds" if not use_fixed_rounds else "fixed_rounds"
    convergence_round = max_t
    
    for t in range(max_t):
        a_agent, info = agent.select_action_with_noise(game)
        
        a_partner = partner_strategy(game, t)
        
        a_partner_observed = agent.observe_partner_action(a_partner, game)
        
        agent.update(
            game=game,
            a_i_VB=info['a_i_VB'],
            a_j_actual=a_partner_observed,
            U_i=info['U_i'],
            U_j_hat=info['U_j_hat'],
        )
        
        actions_agent.append(a_agent)
        actions_partner.append(a_partner)
        payoffs_agent.append(game.R_i[a_agent, a_partner])
        payoffs_partner.append(game.R_j[a_agent, a_partner])
        trembles.append(info.get('trembled', False))
        
        if not use_fixed_rounds and t >= conv_config.min_rounds:
            conv, reason = check_belief_convergence(agent.history, conv_config)
            if conv:
                converged = True
                convergence_reason = reason
                convergence_round = t + 1
                break
    
    final_dominant, final_prob = get_dominant_type_info(agent.B_hat)
    
    return {
        'actions_agent': np.array(actions_agent),
        'actions_partner': np.array(actions_partner),
        'payoffs_agent': np.array(payoffs_agent),
        'payoffs_partner': np.array(payoffs_partner),
        'history': agent.history,
        'mean_payoff_agent': np.mean(payoffs_agent),
        'mean_payoff_partner': np.mean(payoffs_partner),
        'tremble_rate': np.mean(trembles),
        'n_rounds': len(actions_agent),
        'converged': converged,
        'convergence_reason': convergence_reason,
        'convergence_round': convergence_round,
        'final_dominant_type': list(FairnessPrinciple)[final_dominant].name,
        'final_dominant_prob': final_prob,
    }


def always_cooperate_strategy(game: Game, t: int) -> int:
    return len(game.action_space_j) - 1


def always_defect_strategy(game: Game, t: int) -> int:
    return 0


def random_strategy(game: Game, t: int, rng: np.random.Generator = None) -> int:
    if rng is None:
        rng = np.random.default_rng()
    return rng.integers(0, len(game.action_space_j))


def tit_for_tat_strategy(last_opponent_action: List[int]) -> Callable:
    def strategy(game: Game, t: int) -> int:
        if t == 0:
            return len(game.action_space_j) - 1
        else:
            return last_opponent_action[0]
    return strategy


def deceptive_strategy(cooperate_rounds: int = 10) -> Callable:
    def strategy(game: Game, t: int) -> int:
        max_action = len(game.action_space_j) - 1
        if t < cooperate_rounds:
            return max_action
        else:
            return 0
    return strategy


# main function ----------->>>>
if __name__ == "__main__":
    mgg = MoralGameGenerator(num_actions=11)
    game = mgg.sample_game()
    
    print(f"Game archetype: {game.archetype.name}")
    print(f"Action space size: {len(game.action_space_i)}")
    
    rng = np.random.default_rng(42)

    prior_1 = np.array([0.6, 0.2, 0.2])  # Max-sum, Equal-split, Rawls
    agent_1 = create_tgmi_agent(
        moral_prior=prior_1,
        theta_0=0.5,
        phi=0.1,
        beta=1.0,
        epsilon=0.05,  # 5% action error
        eta=0.05,      # 5% observation error
        agent_id="Agent_1",
        rng=rng,
    )
    
    prior_2 = np.array([0.2, 0.6, 0.2])
    agent_2 = create_tgmi_agent(
        moral_prior=prior_2,
        theta_0=0.5,
        phi=0.1,
        beta=1.0,
        epsilon=0.05,
        eta=0.05,
        agent_id="Agent_2",
        rng=rng,
    )
    
    print("\nAgent 1 moral prior:", prior_1)
    print("Agent 2 moral prior:", prior_2)
    
    print("\n=== Demo 1: Convergence-based stopping (default) ===")
    conv_config = ConvergenceConfig(
        prob_threshold=0.01,
        window_size=5,
        max_rounds=500,
        min_rounds=10,
    )
    results = simulate_interaction(agent_1, agent_2, game, conv_config=conv_config)
    
    print(f"Converged: {results['converged']}")
    print(f"Convergence reason: {results['convergence_reason']}")
    print(f"Rounds to convergence: {results['convergence_round']}")
    print(f"Total rounds played: {results['n_rounds']}")
    print(f"Mean payoff Agent 1: {results['mean_payoff_i']:.4f}")
    print(f"Mean payoff Agent 2: {results['mean_payoff_j']:.4f}")
    print(f"Final dominant type Agent 1: {results['final_dominant_type_i']} (p={results['final_dominant_prob_i']:.3f})")
    print(f"Final dominant type Agent 2: {results['final_dominant_type_j']} (p={results['final_dominant_prob_j']:.3f})")
    
    print("\n=== Demo 2: Fixed rounds mode (n_rounds=50) ===")
    results_fixed = simulate_interaction(agent_1, agent_2, game, n_rounds=50)
    
    print(f"Converged: {results_fixed['converged']}")
    print(f"Convergence reason: {results_fixed['convergence_reason']}")
    print(f"Total rounds played: {results_fixed['n_rounds']}")
    print(f"Mean payoff Agent 1: {results_fixed['mean_payoff_i']:.4f}")
    print(f"Mean payoff Agent 2: {results_fixed['mean_payoff_j']:.4f}")
    
    print("\n=== Demo 3: Against fixed strategies (convergence mode) ===")
    
    agent_test = create_tgmi_agent(
        moral_prior=np.array([0.4, 0.3, 0.3]),
        theta_0=0.5,
        rng=rng,
    )
    
    game_test = mgg.sample_game()
    
    results_coop = simulate_with_fixed_partner(
        agent_test,
        always_cooperate_strategy,
        game_test,
    )
    print(f"\nVs Always Cooperate (convergence mode):")
    print(f"  Converged: {results_coop['converged']} ({results_coop['convergence_reason']})")
    print(f"  Rounds: {results_coop['n_rounds']}")
    print(f"  Inferred type: {results_coop['final_dominant_type']} (p={results_coop['final_dominant_prob']:.3f})")
    
    agent_test.reset()
    results_coop_fixed = simulate_with_fixed_partner(
        agent_test,
        always_cooperate_strategy,
        game_test,
        n_rounds=30,  
    )
    print(f"\nVs Always Cooperate (fixed 30 rounds):")
    print(f"  Converged: {results_coop_fixed['converged']} ({results_coop_fixed['convergence_reason']})")
    print(f"  Rounds: {results_coop_fixed['n_rounds']}")
    print(f"  Inferred type: {results_coop_fixed['final_dominant_type']} (p={results_coop_fixed['final_dominant_prob']:.3f})")
