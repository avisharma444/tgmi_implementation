"""
Figure 3C: Steady-state abundance of TGMI as a function of action error ε_a
============================================================================

Tests TGMI robustness to noisy behavior.

Key insight:
- With probability ε_a, an agent's chosen action is replaced by a random action
- TGMI should be robust because trust updates depend on fairness deviation, not single defections
- Reciprocity strategies (TFT) are famously brittle under noise

Fixed parameters:
- Game length T = 20
- Population size N = 10
- No perception noise (ε_p = 0)

Sweep: ε_a ∈ {0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4}
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import sys
import os

# Add parent directory to path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
sys.path.insert(0, _parent_dir)

from tgmi.agent import (
    TGMIAgent, TGMIConfig, create_tgmi_agent,
    FairnessPrinciple, FAIRNESS_KEYS,
)
from MGG.generator import MoralGameGenerator, Game


# =============================================================================
# Strategy Implementations
# =============================================================================

class AllC:
    """Always Cooperate."""
    name = "AllC"
    
    def __init__(self, rng=None, epsilon=0.0):
        self.rng = rng if rng is not None else np.random.default_rng()
        self.epsilon = epsilon
    
    def reset(self): pass
    
    def select_action(self, game: Game, history=None) -> int:
        n = len(game.action_space_j)
        action = n - 1  # Cooperate = max action
        if self.rng.random() < self.epsilon:
            return self.rng.integers(0, n)
        return action


class AllD:
    """Always Defect."""
    name = "AllD"
    
    def __init__(self, rng=None, epsilon=0.0):
        self.rng = rng if rng is not None else np.random.default_rng()
        self.epsilon = epsilon
    
    def reset(self): pass
    
    def select_action(self, game: Game, history=None) -> int:
        n = len(game.action_space_j)
        if self.rng.random() < self.epsilon:
            return self.rng.integers(0, n)
        return 0  # Defect = min action


class TFT:
    """Tit-for-Tat."""
    name = "TFT"
    
    def __init__(self, rng=None, epsilon=0.0):
        self.rng = rng if rng is not None else np.random.default_rng()
        self.epsilon = epsilon
    
    def reset(self): pass
    
    def select_action(self, game: Game, history=None) -> int:
        n = len(game.action_space_j)
        if history is None or len(history) == 0:
            action = n - 1  # Cooperate first
        else:
            _, partner_last = history[-1]
            action = partner_last
        
        if self.rng.random() < self.epsilon:
            return self.rng.integers(0, n)
        return action


class TGMIStrategy:
    """TGMI agent wrapped as a strategy."""
    name = "TGMI"
    
    def __init__(self, rng=None, epsilon=0.0, eta=0.0):
        self.rng = rng if rng is not None else np.random.default_rng()
        self.epsilon = epsilon
        self.eta = eta
        self.agent = None
        self._create_agent()
    
    def _create_agent(self):
        self.agent = create_tgmi_agent(
            moral_prior=None,
            theta_0=0.5,
            phi=0.15,
            beta=1.0,
            xi_dev=5.0,
            gamma_bargain=0.5,
            gamma_mixture=0.1,
            epsilon=self.epsilon,
            eta=self.eta,
            rng=self.rng,
        )
    
    def reset(self):
        self._create_agent()
    
    def select_action(self, game: Game, history=None) -> int:
        action, _ = self.agent.select_action(game)
        return action
    
    def get_action_with_info(self, game: Game):
        return self.agent.select_action(game)
    
    def update(self, game: Game, my_action: int, partner_action: int, info: dict):
        partner_observed = self.agent.observe_partner_action(partner_action, game)
        self.agent.update(
            game=game,
            a_i_VB=info.get('a_i_VB', my_action),
            a_j_actual=partner_observed,
            U_i=info['U_i'],
            U_j_hat=info['U_j_hat'],
        )


# Strategy types for the simulation
STRATEGY_TYPES = ['TGMI', 'TFT', 'AllC', 'AllD']


def create_strategy(name: str, rng=None, epsilon=0.05, eta=0.0):
    """Factory to create strategies."""
    if rng is None:
        rng = np.random.default_rng()
    
    if name == 'TGMI':
        return TGMIStrategy(rng=rng, epsilon=epsilon, eta=eta)
    elif name == 'TFT':
        return TFT(rng=rng, epsilon=epsilon)
    elif name == 'AllC':
        return AllC(rng=rng, epsilon=epsilon)
    elif name == 'AllD':
        return AllD(rng=rng, epsilon=epsilon)
    else:
        raise ValueError(f"Unknown strategy: {name}")


# =============================================================================
# Pairwise Interaction
# =============================================================================

def play_match(
    strat_i,
    strat_j,
    game: Game,
    T: int,
    omega: float = 0.5,
) -> Tuple[float, float]:
    """
    Play T rounds and return fitness for both players.
    Fitness = (1-ω)*payoff + ω*fairness
    """
    strat_i.reset()
    strat_j.reset()
    
    history_i, history_j = [], []
    total_payoff_i, total_payoff_j = 0.0, 0.0
    total_fairness_i, total_fairness_j = 0.0, 0.0
    
    for t in range(T):
        # Get actions
        if isinstance(strat_i, TGMIStrategy):
            action_i, info_i = strat_i.get_action_with_info(game)
        else:
            action_i = strat_i.select_action(game, history_i)
            info_i = None
        
        if isinstance(strat_j, TGMIStrategy):
            action_j, info_j = strat_j.get_action_with_info(game)
        else:
            action_j = strat_j.select_action(game, history_j)
            info_j = None
        
        # Payoffs
        payoff_i = game.R_i[action_i, action_j]
        payoff_j = game.R_j[action_i, action_j]
        
        # Fairness (equal split)
        fairness = game.F['equal_split'][action_i, action_j]
        
        total_payoff_i += payoff_i
        total_payoff_j += payoff_j
        total_fairness_i += fairness
        total_fairness_j += fairness
        
        # Update histories
        history_i.append((action_i, action_j))
        history_j.append((action_j, action_i))
        
        # Update TGMI agents
        if isinstance(strat_i, TGMIStrategy) and info_i:
            strat_i.update(game, action_i, action_j, info_i)
        if isinstance(strat_j, TGMIStrategy) and info_j:
            strat_j.update(game, action_j, action_i, info_j)
    
    # Fitness
    fitness_i = (1 - omega) * total_payoff_i / T + omega * total_fairness_i / T
    fitness_j = (1 - omega) * total_payoff_j / T + omega * total_fairness_j / T
    
    return fitness_i, fitness_j


def compute_payoff_matrix(
    strategy_types: List[str],
    mgg: MoralGameGenerator,
    T: int,
    omega: float = 0.5,
    n_samples: int = 20,
    epsilon: float = 0.05,
    rng=None,
    games: List[Game] = None,
) -> np.ndarray:
    """Compute expected payoff matrix."""
    if rng is None:
        rng = np.random.default_rng()
    
    n_types = len(strategy_types)
    payoffs = np.zeros((n_types, n_types))
    
    # Use pre-sampled games if provided
    if games is None:
        games = [mgg.sample_game() for _ in range(n_samples)]
    
    for i, type_i in enumerate(strategy_types):
        for j, type_j in enumerate(strategy_types):
            total = 0.0
            for game in games:
                s_i = create_strategy(type_i, rng=rng, epsilon=epsilon)
                s_j = create_strategy(type_j, rng=rng, epsilon=epsilon)
                f_i, _ = play_match(s_i, s_j, game, T, omega)
                total += f_i
            payoffs[i, j] = total / len(games)
    
    return payoffs


# =============================================================================
# Monte Carlo Moran Process
# =============================================================================

def run_moran_monte_carlo(
    payoff_matrix: np.ndarray,
    N: int = 10,
    selection_strength: float = 2.0,
    mutation_rate: float = 0.001,
    n_generations: int = 5000,
    burn_in: int = 1000,
    rng=None,
) -> np.ndarray:
    """
    Run Moran process via Monte Carlo simulation.
    Returns estimated stationary abundance for each type.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    n_types = payoff_matrix.shape[0]
    
    # Initialize population (uniform)
    population = np.zeros(n_types, dtype=int)
    for i in range(N):
        population[i % n_types] += 1
    
    abundance_samples = []
    
    for gen in range(n_generations):
        # Compute fitness for each type
        type_counts = population
        total_pop = N
        
        # Average payoff for each type
        fitness = np.zeros(n_types)
        for i in range(n_types):
            if type_counts[i] > 0:
                for j in range(n_types):
                    fitness[i] += payoff_matrix[i, j] * type_counts[j] / total_pop
        
        # Selection probabilities (exponential fitness)
        selection_probs = np.zeros(n_types)
        for i in range(n_types):
            selection_probs[i] = type_counts[i] * np.exp(selection_strength * fitness[i])
        
        if selection_probs.sum() > 0:
            selection_probs /= selection_probs.sum()
        else:
            selection_probs = type_counts / total_pop
        
        # Death probabilities (uniform)
        death_probs = type_counts / total_pop
        
        # Select reproducer
        if rng.random() < mutation_rate:
            # Mutation: random type
            reproducer = rng.integers(0, n_types)
        else:
            reproducer = rng.choice(n_types, p=selection_probs)
        
        # Select individual to die
        dying = rng.choice(n_types, p=death_probs)
        
        # Update population
        if population[dying] > 0:
            population[dying] -= 1
            population[reproducer] += 1
        
        # Record abundance after burn-in
        if gen >= burn_in:
            abundance_samples.append(population.copy() / N)
    
    # Return mean abundance
    return np.mean(abundance_samples, axis=0)


# =============================================================================
# Figure 3C Configuration and Simulation
# =============================================================================

@dataclass
class Figure3CConfig:
    N: int = 10
    T: int = 20  # Fixed game length
    selection_strength: float = 2.0
    mutation_rate: float = 0.001
    omega: float = 0.5
    n_samples: int = 50
    n_generations: int = 10000
    burn_in: int = 3000
    epsilon_values: List[float] = None  # Action error values to sweep
    seed: int = 42


def run_figure_3c_simulations(config: Figure3CConfig = None, verbose=True) -> Dict:
    """Run Figure 3C: TGMI abundance vs action error ε_a."""
    if config is None:
        config = Figure3CConfig()
    
    if config.epsilon_values is None:
        config.epsilon_values = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
    
    strategy_types = STRATEGY_TYPES
    tgmi_idx = strategy_types.index('TGMI')
    
    # Pre-sample games ONCE for all ε_a values (ensures consistency)
    master_rng = np.random.default_rng(config.seed)
    mgg = MoralGameGenerator(num_actions=11, rng=master_rng)
    games = [mgg.sample_game() for _ in range(config.n_samples)]
    
    if verbose:
        print(f"Pre-sampled {len(games)} games for consistent payoff estimation")
        print(f"Fixed game length T = {config.T}")
    
    abundances = []
    all_abundances = []
    
    for eps_a in config.epsilon_values:
        if verbose:
            print(f"ε_a={eps_a:.2f}: Computing payoff matrix...")
        
        # Use fresh RNG for each ε_a
        rng = np.random.default_rng(config.seed + int(eps_a * 1000))
        
        # Compute payoff matrix with this action error
        payoff_matrix = compute_payoff_matrix(
            strategy_types, mgg, config.T,
            omega=config.omega,
            n_samples=config.n_samples,
            epsilon=eps_a,  # This is the sweep variable!
            rng=rng,
            games=games,
        )
        
        if verbose:
            print(f"  Payoff matrix:\n{np.round(payoff_matrix, 3)}")
        
        # Run multiple Monte Carlo Moran processes and average
        n_mc_runs = 20
        abundance_runs = []
        
        for mc_run in range(n_mc_runs):
            mc_rng = np.random.default_rng(config.seed + int(eps_a * 1000) + mc_run * 100)
            abundance = run_moran_monte_carlo(
                payoff_matrix,
                N=config.N,
                selection_strength=config.selection_strength,
                mutation_rate=config.mutation_rate,
                n_generations=config.n_generations,
                burn_in=config.burn_in,
                rng=mc_rng,
            )
            abundance_runs.append(abundance)
        
        # Average across runs
        mean_abundance = np.mean(abundance_runs, axis=0)
        
        abundances.append(mean_abundance[tgmi_idx])
        all_abundances.append(mean_abundance)
        
        if verbose:
            print(f"  Abundances: {dict(zip(strategy_types, np.round(mean_abundance, 3)))}")
            print(f"  TGMI abundance: {mean_abundance[tgmi_idx]:.4f}")
    
    return {
        'epsilon_values': np.array(config.epsilon_values),
        'abundances': np.array(abundances),
        'all_abundances': np.array(all_abundances),
        'strategy_types': strategy_types,
        'config': config,
    }


# =============================================================================
# Plotting
# =============================================================================

def plot_figure_3c(results: Dict, save_path: str = None):
    """Plot Figure 3C: TGMI abundance vs action error."""
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
    })
    
    epsilon_values = results['epsilon_values']
    abundances = results['abundances']
    n_types = len(results['strategy_types'])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot TGMI abundance
    ax.plot(epsilon_values, abundances, 'o-',
            color='#2E86AB', linewidth=2.5, markersize=10,
            markerfacecolor='white', markeredgewidth=2.5,
            label='TGMI')
    
    # Reference lines
    ax.axhline(y=1/n_types, color='gray', linestyle='--',
               linewidth=1.5, alpha=0.7, label=f'Neutral (1/{n_types})')
    ax.axhline(y=0.5, color='red', linestyle=':',
               linewidth=1.5, alpha=0.7, label='Majority')
    
    ax.set_xlabel(r'Action error $\varepsilon_a$')
    ax.set_ylabel('Steady-state TGMI abundance')
    ax.set_title('C', fontweight='bold', loc='left', fontsize=16)
    
    ax.set_xlim(-0.02, 0.45)
    ax.set_ylim(0, 0.8)
    
    ax.legend(loc='upper right', framealpha=0.95)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to {save_path}")
    
    return fig


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("Running Figure 3C simulations...")
    print("=" * 60)
    print("Testing TGMI robustness to action noise")
    print("Strategy types:", STRATEGY_TYPES)
    print("=" * 60)
    
    config = Figure3CConfig(
        N=10,
        T=20,  # Fixed game length
        selection_strength=2.0,
        mutation_rate=0.001,
        omega=0.5,
        n_samples=100,
        n_generations=20000,
        burn_in=5000,
        epsilon_values=[0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4],
        seed=42,
    )
    
    results = run_figure_3c_simulations(config, verbose=True)
    
    print("\n" + "=" * 60)
    print("RESULTS: TGMI Abundance vs Action Error ε_a")
    print("=" * 60)
    for eps, ab in zip(results['epsilon_values'], results['abundances']):
        marker = "✓ ROBUST" if ab > 0.5 else ("~" if ab > 0.25 else "✗")
        print(f"  ε_a={eps:.2f}: {ab:.4f} {marker}")
    
    # Robustness analysis
    low_noise_avg = np.mean(results['abundances'][:3])  # ε_a ≤ 0.1
    high_noise_avg = np.mean(results['abundances'][-2:])  # ε_a ≥ 0.3
    print(f"\nLow noise avg (ε_a ≤ 0.1): {low_noise_avg:.4f}")
    print(f"High noise avg (ε_a ≥ 0.3): {high_noise_avg:.4f}")
    
    # Save plot
    output_dir = os.path.join(_parent_dir, 'outputs', 'plots')
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'figure_3c.png')
    
    fig = plot_figure_3c(results, save_path=save_path)
    plt.show()
    
    print("\nDone!")
