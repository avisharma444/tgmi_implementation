import numpy as np
import scipy.special
from typing import Dict, List, Tuple, Callable, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from collections import defaultdict
import sys
import os

_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
sys.path.insert(0, _parent_dir)

from tgmi.agent import (
    TGMIAgent, TGMIConfig, create_tgmi_agent,
    sample_moral_prior, FairnessPrinciple, FAIRNESS_KEYS,
    simulate_interaction, simulate_with_fixed_partner,
    always_cooperate_strategy, always_defect_strategy,
    deceptive_strategy,
    ConvergenceConfig, check_belief_convergence,
)
from mgg.generator import MoralGameGenerator, Game, Archetype

EPSILON = 0.05  # 5% action error 
ETA = 0.05      # 5% observation error 

DEFAULT_CONV_CONFIG = ConvergenceConfig(
    prob_threshold=0.01,
    window_size=5,
    max_rounds=500,
    min_rounds=10,
    check_both_agents=False,
)

class AgentTypeEnum:
    SELFISH = 0     
    ALTRUISTIC = 1   
    RECIPROCAL = 2  


def softmax(utilities: np.ndarray, beta: float) -> np.ndarray:
    if beta == np.inf:
        max_u = utilities.max()
        probs = (utilities == max_u).astype(float)
        return probs / probs.sum()
    else:
        return scipy.special.softmax(beta * utilities)


def add_tremble(probs: np.ndarray, tremble: float) -> np.ndarray:
    if tremble == 0 or np.isnan(tremble):
        return probs
    
    n = len(probs)
    if n <= 1:
        return probs
    
    tremble_matrix = (
        (1 - np.eye(n)) * tremble / (n - 1) +
        np.eye(n) * (1 - tremble)
    )
    return tremble_matrix.T @ probs


class BRAgent:
    def __init__(
        self,
        prior: float = 0.5,  # Prior probability partner is reciprocal
        beta: float = 5.0,   # Softmax temperature
        tremble: float = 0.0,  # Action noise for likelihood computation
        agent_id: str = None,
        rng: np.random.Generator = None,
    ):
        self.rng = rng if rng is not None else np.random.default_rng()
        self.agent_id = agent_id
        self.beta = beta
        self.tremble = tremble
        
        self.n_types = 3
        
        non_recip_prior = (1 - prior) / 2
        self.pop_prior = np.array([non_recip_prior, non_recip_prior, prior])
        
        self._initialize_state()
        
        self.history = {
            'beliefs': [self.belief.copy()],
            'actions': [],
            'partner_actions': [],
            'log_likelihoods': [],
        }
    
    def _initialize_state(self):
        self.belief = self.pop_prior.copy()
        self.log_likelihood = np.zeros(self.n_types)
    
    def reset(self):
        self._initialize_state()
        self.history = {
            'beliefs': [self.belief.copy()],
            'actions': [],
            'partner_actions': [],
            'log_likelihoods': [],
        }
    
    def _compute_type_utility(
        self,
        agent_type: int,
        game: Game,
        a_i: int,
        a_j: int,
        belief_partner_reciprocal: float = 0.5,
    ) -> float:
        own_payoff = game.R_i[a_i, a_j]
        partner_payoff = game.R_j[a_i, a_j]
        
        if agent_type == AgentTypeEnum.SELFISH:
            return own_payoff
        elif agent_type == AgentTypeEnum.ALTRUISTIC:
            return own_payoff + partner_payoff
        elif agent_type == AgentTypeEnum.RECIPROCAL:
            return own_payoff + belief_partner_reciprocal * partner_payoff
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
    
    def _compute_action_probs_for_type(
        self,
        agent_type: int,
        game: Game,
        belief_partner_reciprocal: float = 0.5,
    ) -> np.ndarray:
        n_actions_i = len(game.action_space_i)
        n_actions_j = len(game.action_space_j)
        
        utilities = np.zeros(n_actions_i)
        
        for a_i in range(n_actions_i):
            expected_u = 0.0
            for a_j in range(n_actions_j):
                expected_u += self._compute_type_utility(
                    agent_type, game, a_i, a_j, belief_partner_reciprocal
                )
            utilities[a_i] = expected_u / n_actions_j
        
        return add_tremble(softmax(utilities, self.beta), self.tremble)
    
    def compute_utility(
        self,
        game: Game,
        a_i: int,
        a_j: int,
    ) -> float:
        belief_partner_reciprocal = self.belief[AgentTypeEnum.RECIPROCAL]
        return self._compute_type_utility(
            AgentTypeEnum.RECIPROCAL, game, a_i, a_j, belief_partner_reciprocal
        )
    
    def select_action(
        self,
        game: Game,
    ) -> Tuple[int, dict]:
        n_actions_i = len(game.action_space_i)
        n_actions_j = len(game.action_space_j)
        
        utilities = np.zeros(n_actions_i)
        
        for a_i in range(n_actions_i):
            expected_u = 0.0
            for a_j in range(n_actions_j):
                expected_u += self.compute_utility(game, a_i, a_j)
            utilities[a_i] = expected_u / n_actions_j
        
        probs = softmax(utilities, self.beta)
        action = self.rng.choice(n_actions_i, p=probs)
        
        return action, {'utilities': utilities, 'probs': probs}
    
    def update(
        self,
        game: Game,
        my_action: int,
        partner_action: int,
    ):
        likelihoods = np.zeros(self.n_types)
        
        for agent_type in range(self.n_types):
            action_probs = self._compute_action_probs_for_type(
                agent_type, game, 
                belief_partner_reciprocal=self.belief[AgentTypeEnum.RECIPROCAL]
            )
            likelihoods[agent_type] = action_probs[partner_action]
        
        likelihoods = np.clip(likelihoods, 1e-10, 1.0)
        
        self.log_likelihood += np.log(likelihoods)
        
        log_prior = np.log(self.pop_prior + 1e-10)
        log_posterior = log_prior + self.log_likelihood
        
        log_posterior -= np.max(log_posterior) 
        self.belief = np.exp(log_posterior)
        self.belief = self.belief / self.belief.sum()
        
        self.history['beliefs'].append(self.belief.copy())
        self.history['actions'].append(my_action)
        self.history['partner_actions'].append(partner_action)
        self.history['log_likelihoods'].append(self.log_likelihood.copy())


def create_br_agent(
    prior: float = 0.5,
    beta: float = 5.0,
    tremble: float = 0.0,
    agent_id: str = None,
    rng: np.random.Generator = None,
) -> BRAgent:
    return BRAgent(
        prior=prior,
        beta=beta,
        tremble=tremble,
        agent_id=agent_id,
        rng=rng,
    )

@dataclass
class ExperimentResult:
    agent_i_type: str
    agent_j_type: str
    game_archetype: str
    n_rounds: int
    payoffs_i: np.ndarray
    payoffs_j: np.ndarray
    actions_i: np.ndarray
    actions_j: np.ndarray
    mean_payoff_i: float
    mean_payoff_j: float
    total_welfare: float
    cooperation_rate: float
    fairness_scores: Dict[str, float]
    converged: bool = False
    convergence_reason: str = None
    convergence_round: int = None


def compute_cooperation_rate(
    actions: np.ndarray,
    n_actions: int,
) -> float:
    max_action = n_actions - 1
    threshold = max_action * 0.5
    return np.mean(actions >= threshold)


def compute_mean_fairness(
    game: Game,
    actions_i: np.ndarray,
    actions_j: np.ndarray,
) -> Dict[str, float]:
    fairness_scores = {}
    
    for key in ['max_sum', 'equal_split', 'rawls']:
        if key in game.F:
            scores = [game.F[key][a_i, a_j] for a_i, a_j in zip(actions_i, actions_j)]
            fairness_scores[key] = np.mean(scores)
    
    return fairness_scores


def run_tgmi_vs_tgmi(
    prior_i: np.ndarray,
    prior_j: np.ndarray,
    game: Game,
    n_rounds: int = None,
    conv_config: ConvergenceConfig = None,
    config: TGMIConfig = None,
    epsilon: float = None,
    eta: float = None,
    rng: np.random.Generator = None,
) -> ExperimentResult:
    if rng is None:
        rng = np.random.default_rng()
    
    if epsilon is None:
        epsilon = EPSILON
    if eta is None:
        eta = ETA
    
    agent_i = create_tgmi_agent(moral_prior=prior_i, epsilon=epsilon, eta=eta, rng=rng)
    agent_j = create_tgmi_agent(moral_prior=prior_j, epsilon=epsilon, eta=eta, rng=rng)
    
    if config:
        agent_i.config = config
        agent_j.config = config
    
    results = simulate_interaction(agent_i, agent_j, game, n_rounds=n_rounds, conv_config=conv_config)
    
    fairness = compute_mean_fairness(game, results['actions_i'], results['actions_j'])
    coop_rate = 0.5 * (
        compute_cooperation_rate(results['actions_i'], len(game.action_space_i)) +
        compute_cooperation_rate(results['actions_j'], len(game.action_space_j))
    )
    
    return ExperimentResult(
        agent_i_type='TGMI',
        agent_j_type='TGMI',
        game_archetype=game.archetype.name,
        n_rounds=results['n_rounds'],
        payoffs_i=results['payoffs_i'],
        payoffs_j=results['payoffs_j'],
        actions_i=results['actions_i'],
        actions_j=results['actions_j'],
        mean_payoff_i=results['mean_payoff_i'],
        mean_payoff_j=results['mean_payoff_j'],
        total_welfare=results['total_payoff'],
        cooperation_rate=coop_rate,
        fairness_scores=fairness,
        converged=results.get('converged', False),
        convergence_reason=results.get('convergence_reason'),
        convergence_round=results.get('convergence_round'),
    )


def run_br_vs_br(
    game: Game,
    n_rounds: int = 100,
    prior_i: float = 0.5,
    prior_j: float = 0.5,
    rng: np.random.Generator = None,
) -> ExperimentResult:
    if rng is None:
        rng = np.random.default_rng()
    
    agent_i = create_br_agent(prior=prior_i, rng=rng)
    agent_j = create_br_agent(prior=prior_j, rng=rng)
    
    actions_i, actions_j = [], []
    payoffs_i, payoffs_j = [], []
    
    for t in range(n_rounds):
        a_i, _ = agent_i.select_action(game)
        a_j, _ = agent_j.select_action(game)
        
        agent_i.update(game, a_i, a_j)
        agent_j.update(game, a_j, a_i)
        
        actions_i.append(a_i)
        actions_j.append(a_j)
        payoffs_i.append(game.R_i[a_i, a_j])
        payoffs_j.append(game.R_j[a_i, a_j])
    
    actions_i = np.array(actions_i)
    actions_j = np.array(actions_j)
    payoffs_i = np.array(payoffs_i)
    payoffs_j = np.array(payoffs_j)
    
    fairness = compute_mean_fairness(game, actions_i, actions_j)
    coop_rate = 0.5 * (
        compute_cooperation_rate(actions_i, len(game.action_space_i)) +
        compute_cooperation_rate(actions_j, len(game.action_space_j))
    )
    
    return ExperimentResult(
        agent_i_type='BR',
        agent_j_type='BR',
        game_archetype=game.archetype.name,
        n_rounds=n_rounds,
        payoffs_i=payoffs_i,
        payoffs_j=payoffs_j,
        actions_i=actions_i,
        actions_j=actions_j,
        mean_payoff_i=np.mean(payoffs_i),
        mean_payoff_j=np.mean(payoffs_j),
        total_welfare=np.sum(payoffs_i) + np.sum(payoffs_j),
        cooperation_rate=coop_rate,
        fairness_scores=fairness,
    )


def run_tgmi_vs_br(
    prior_i: np.ndarray,
    game: Game,
    n_rounds: int = None,
    conv_config: ConvergenceConfig = None,
    tgmi_config: TGMIConfig = None,
    br_prior: float = 0.5,
    epsilon: float = None,
    eta: float = None,
    rng: np.random.Generator = None,
) -> ExperimentResult:
    if rng is None:
        rng = np.random.default_rng()
    
    if conv_config is None:
        conv_config = ConvergenceConfig()
    
    if epsilon is None:
        epsilon = EPSILON
    if eta is None:
        eta = ETA
    
    use_fixed_rounds = n_rounds is not None
    max_t = n_rounds if use_fixed_rounds else conv_config.max_rounds
    
    agent_tgmi = create_tgmi_agent(moral_prior=prior_i, epsilon=epsilon, eta=eta, rng=rng)
    agent_br = create_br_agent(prior=br_prior, rng=rng)
    
    if tgmi_config:
        agent_tgmi.config = tgmi_config
    
    actions_tgmi, actions_br = [], []
    payoffs_tgmi, payoffs_br = [], []
    
    converged = False
    convergence_reason = "max_rounds" if not use_fixed_rounds else "fixed_rounds"
    convergence_round = max_t
    
    for t in range(max_t):
        a_tgmi, info_tgmi = agent_tgmi.select_action(game)
        a_br, _ = agent_br.select_action(game)
        
        agent_tgmi.update(
            game=game,
            a_i_VB=a_tgmi,
            a_j_actual=a_br,
            U_i=info_tgmi['U_i'],
            U_j_hat=info_tgmi['U_j_hat'],
        )
        
        agent_br.update(game, a_br, a_tgmi)
        
        actions_tgmi.append(a_tgmi)
        actions_br.append(a_br)
        payoffs_tgmi.append(game.R_i[a_tgmi, a_br])
        payoffs_br.append(game.R_j[a_tgmi, a_br])
        
        if not use_fixed_rounds and t >= conv_config.min_rounds:
            conv, reason = check_belief_convergence(agent_tgmi.history, conv_config)
            if conv:
                converged = True
                convergence_reason = reason
                convergence_round = t + 1
                break
    
    actions_tgmi = np.array(actions_tgmi)
    actions_br = np.array(actions_br)
    payoffs_tgmi = np.array(payoffs_tgmi)
    payoffs_br = np.array(payoffs_br)
    
    fairness = compute_mean_fairness(game, actions_tgmi, actions_br)
    coop_rate = 0.5 * (
        compute_cooperation_rate(actions_tgmi, len(game.action_space_i)) +
        compute_cooperation_rate(actions_br, len(game.action_space_j))
    )
    
    return ExperimentResult(
        agent_i_type='TGMI',
        agent_j_type='BR',
        game_archetype=game.archetype.name,
        n_rounds=len(actions_tgmi),
        payoffs_i=payoffs_tgmi,
        payoffs_j=payoffs_br,
        actions_i=actions_tgmi,
        actions_j=actions_br,
        mean_payoff_i=np.mean(payoffs_tgmi),
        mean_payoff_j=np.mean(payoffs_br),
        total_welfare=np.sum(payoffs_tgmi) + np.sum(payoffs_br),
        cooperation_rate=coop_rate,
        fairness_scores=fairness,
        converged=converged,
        convergence_reason=convergence_reason,
        convergence_round=convergence_round,
    )

def run_batch_experiments(
    n_games: int = 20,
    n_rounds: int = None,
    conv_config: ConvergenceConfig = None,
    n_prior_samples: int = 10,
    rng: np.random.Generator = None,
) -> Dict[str, List[ExperimentResult]]:
    if rng is None:
        rng = np.random.default_rng()
    
    mgg = MoralGameGenerator(num_actions=11, rng=rng)
    
    results = {
        'tgmi_vs_tgmi_same': [],
        'tgmi_vs_tgmi_diff': [],
        'br_vs_br': [],
        'tgmi_vs_br': [],
    }
    
    for game_idx in range(n_games):
        game = mgg.sample_game()
        print(f"Game {game_idx + 1}/{n_games}: {game.archetype.name}")
        
        for prior_idx in range(n_prior_samples):
            prior_1 = sample_moral_prior(rng)
            prior_2 = sample_moral_prior(rng)
            
            result = run_tgmi_vs_tgmi(prior_1, prior_1, game, n_rounds=n_rounds, conv_config=conv_config, rng=rng)
            results['tgmi_vs_tgmi_same'].append(result)
            
            result = run_tgmi_vs_tgmi(prior_1, prior_2, game, n_rounds=n_rounds, conv_config=conv_config, rng=rng)
            results['tgmi_vs_tgmi_diff'].append(result)
            
            br_rounds = n_rounds if n_rounds is not None else 100
            result = run_br_vs_br(game, br_rounds, rng=rng)
            results['br_vs_br'].append(result)
            
            result = run_tgmi_vs_br(prior_1, game, n_rounds=n_rounds, conv_config=conv_config, rng=rng)
            results['tgmi_vs_br'].append(result)
    
    return results


def summarize_results(results: Dict[str, List[ExperimentResult]]) -> Dict:
    summary = {}
    
    for exp_type, exp_results in results.items():
        if not exp_results:
            continue
        
        payoffs_i = [r.mean_payoff_i for r in exp_results]
        payoffs_j = [r.mean_payoff_j for r in exp_results]
        welfare = [r.total_welfare for r in exp_results]
        coop = [r.cooperation_rate for r in exp_results]
        
        summary[exp_type] = {
            'mean_payoff_i': np.mean(payoffs_i),
            'std_payoff_i': np.std(payoffs_i),
            'mean_payoff_j': np.mean(payoffs_j),
            'std_payoff_j': np.std(payoffs_j),
            'mean_welfare': np.mean(welfare),
            'std_welfare': np.std(welfare),
            'mean_cooperation': np.mean(coop),
            'std_cooperation': np.std(coop),
            'n_experiments': len(exp_results),
        }
        
        fairness_keys = ['max_sum', 'equal_split', 'rawls']
        for key in fairness_keys:
            scores = [r.fairness_scores.get(key, 0) for r in exp_results]
            summary[exp_type][f'fairness_{key}'] = np.mean(scores)
    
    return summary


def plot_experiment_comparison(
    summary: Dict,
    save_path: str = None,
):
    exp_types = list(summary.keys())
    n_exp = len(exp_types)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    ax = axes[0, 0]
    x = np.arange(n_exp)
    width = 0.35
    payoffs_i = [summary[et]['mean_payoff_i'] for et in exp_types]
    payoffs_j = [summary[et]['mean_payoff_j'] for et in exp_types]
    ax.bar(x - width/2, payoffs_i, width, label='Agent I')
    ax.bar(x + width/2, payoffs_j, width, label='Agent J')
    ax.set_xticks(x)
    ax.set_xticklabels(exp_types, rotation=45, ha='right')
    ax.set_ylabel('Mean Payoff')
    ax.set_title('Mean Payoffs by Experiment Type')
    ax.legend()
    
    ax = axes[0, 1]
    welfare = [summary[et]['mean_welfare'] for et in exp_types]
    ax.bar(x, welfare)
    ax.set_xticks(x)
    ax.set_xticklabels(exp_types, rotation=45, ha='right')
    ax.set_ylabel('Total Welfare')
    ax.set_title('Total Welfare by Experiment Type')
    
    # Cooperation rate
    ax = axes[1, 0]
    coop = [summary[et]['mean_cooperation'] for et in exp_types]
    ax.bar(x, coop)
    ax.set_xticks(x)
    ax.set_xticklabels(exp_types, rotation=45, ha='right')
    ax.set_ylabel('Cooperation Rate')
    ax.set_title('Cooperation Rate by Experiment Type')
    ax.set_ylim([0, 1])
    
    ax = axes[1, 1]
    width = 0.25
    fairness_keys = ['max_sum', 'equal_split', 'rawls']
    for i, key in enumerate(fairness_keys):
        scores = [summary[et].get(f'fairness_{key}', 0) for et in exp_types]
        ax.bar(x + (i - 1) * width, scores, width, label=key)
    ax.set_xticks(x)
    ax.set_xticklabels(exp_types, rotation=45, ha='right')
    ax.set_ylabel('Fairness Score')
    ax.set_title('Fairness Scores by Experiment Type')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
    
    plt.show()


def plot_trust_dynamics(
    agent: TGMIAgent,
    title: str = "TGMI Trust Dynamics",
    save_path: str = None,
):
    history = agent.history
    n_rounds = len(history['theta']) - 1
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    ax = axes[0, 0]
    ax.plot(history['theta'], 'b-', linewidth=2)
    ax.set_xlabel('Round')
    ax.set_ylabel('Trust (θ)')
    ax.set_title('Trust Evolution')
    ax.set_ylim([0, 1])
    
    ax = axes[0, 1]
    ax.plot(history['c'], 'g-', linewidth=2)
    ax.set_xlabel('Round')
    ax.set_ylabel('Confidence (c)')
    ax.set_title('Confidence Evolution')
    ax.set_ylim([0, 1])
    
    ax = axes[1, 0]
    ax.plot(history['varpi'], 'r-', linewidth=2)
    ax.set_xlabel('Round')
    ax.set_ylabel('Cooperation Weight (ϖ)')
    ax.set_title('Cooperation Weight Evolution')
    ax.set_ylim([0, 1])
    
    ax = axes[1, 1]
    beliefs = np.array(history['B_hat'])
    for i, label in enumerate(['Max-Sum', 'Equal-Split', 'Rawls']):
        ax.plot(beliefs[:, i], label=label, linewidth=2)
    ax.set_xlabel('Round')
    ax.set_ylabel('Belief Probability')
    ax.set_title('Belief about Partner\'s Norms')
    ax.legend()
    ax.set_ylim([0, 1])
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    
    plt.show()

def run_main_experiments():
    print("=" * 60)
    print("TGMI Experiments - Generating Plots")
    print("=" * 60)
    
    rng = np.random.default_rng(42)
    mgg = MoralGameGenerator(num_actions=11, rng=rng)
    
    prior_utilitarian = np.array([0.7, 0.15, 0.15])
    prior_egalitarian = np.array([0.15, 0.7, 0.15])
    prior_rawlsian = np.array([0.15, 0.15, 0.7])
    
    print("\n--- Running Batch Experiments ---")
    batch_results = run_batch_experiments(
        n_games=10,
        n_prior_samples=5,
        rng=rng,
    )
    summary = summarize_results(batch_results)
    
    print("Generating: tgmi_comparison.png")
    plot_experiment_comparison(summary, save_path='outputs/plots/tgmi_comparison.png')
    
    print("\n--- Trust vs Defector ---")
    game_demo = mgg.sample_game()
    agent_defector = create_tgmi_agent(moral_prior=prior_utilitarian, rng=rng)
    
    simulate_with_fixed_partner(
        agent_defector,
        always_defect_strategy,
        game_demo,
    )
    
    print("Generating: trust_vs_defector.png")
    plot_trust_dynamics(
        agent_defector, 
        "TGMI vs Always-Defect Partner", 
        save_path='outputs/plots/trust_vs_defector.png'
    )
    
    print("\n--- Trust vs Deceptive Agent ---")
    game_deceptive = mgg.sample_game()
    
    agent_deceptive = create_tgmi_agent(
        moral_prior=prior_egalitarian,
        theta_0=0.5, 
        alpha=0.7,  # Higher self-anchoring to see belief pull-back effect
        rng=rng
    )
    
    simulate_with_fixed_partner(
        agent_deceptive,
        deceptive_strategy(cooperate_rounds=15),
        game_deceptive,
        n_rounds=60,
    )
    
    print("Generating: trust_vs_deceptive.png")
    plot_trust_dynamics(
        agent_deceptive, 
        "TGMI (Egalitarian) vs Deceptive Agent (switches at round 15)",
        save_path='outputs/plots/trust_vs_deceptive.png'
    )
    
    print("\n" + "=" * 60)
    print("Done! Generated 3 plots in outputs/plots/")
    print("=" * 60)
    
    return batch_results, summary


if __name__ == "__main__":
    os.makedirs('outputs/plots', exist_ok=True)
    
    results, summary = run_main_experiments()
