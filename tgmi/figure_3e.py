"""
Figure 3E: Mean Population Welfare as a function of (T, ε_a)
=============================================================

This panel answers: "How good are the outcomes for the population?"

Key insight:
- Welfare measures how well off agents are, not just who dominates
- TGMI dominance should correlate with higher welfare
- This validates that TGMI is good both evolutionarily AND socially

Welfare = average fitness across agents in steady state
Φ_i = (1/T) * Σ[(1-ω)*R_i + ω*U_F_i]
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
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
# Strategy Implementations (same as 3B/3C/3D)
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
        action = n - 1
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
        return 0


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
            action = n - 1
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
    """Play T rounds and return fitness for both players."""
    strat_i.reset()
    strat_j.reset()
    
    history_i, history_j = [], []
    total_payoff_i, total_payoff_j = 0.0, 0.0
    total_fairness_i, total_fairness_j = 0.0, 0.0
    
    for t in range(T):
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
        
        payoff_i = game.R_i[action_i, action_j]
        payoff_j = game.R_j[action_i, action_j]
        fairness = game.F['equal_split'][action_i, action_j]
        
        total_payoff_i += payoff_i
        total_payoff_j += payoff_j
        total_fairness_i += fairness
        total_fairness_j += fairness
        
        history_i.append((action_i, action_j))
        history_j.append((action_j, action_i))
        
        if isinstance(strat_i, TGMIStrategy) and info_i:
            strat_i.update(game, action_i, action_j, info_i)
        if isinstance(strat_j, TGMIStrategy) and info_j:
            strat_j.update(game, action_j, action_i, info_j)
    
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
# Monte Carlo Moran Process with Welfare Tracking
# =============================================================================

def run_moran_monte_carlo_with_welfare(
    payoff_matrix: np.ndarray,
    N: int = 10,
    selection_strength: float = 2.0,
    mutation_rate: float = 0.001,
    n_generations: int = 5000,
    burn_in: int = 1000,
    rng=None,
) -> Tuple[np.ndarray, float]:
    """
    Run Moran process and return both abundance AND mean welfare.
    
    Welfare = mean fitness across agents, averaged over steady-state samples.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    n_types = payoff_matrix.shape[0]
    
    population = np.zeros(n_types, dtype=int)
    for i in range(N):
        population[i % n_types] += 1
    
    abundance_samples = []
    welfare_samples = []
    
    for gen in range(n_generations):
        type_counts = population
        total_pop = N
        
        # Compute fitness for each type
        fitness = np.zeros(n_types)
        for i in range(n_types):
            if type_counts[i] > 0:
                for j in range(n_types):
                    fitness[i] += payoff_matrix[i, j] * type_counts[j] / total_pop
        
        # Compute population welfare = mean fitness across agents
        # Welfare = Σ (n_i / N) * fitness_i
        pop_welfare = 0.0
        for i in range(n_types):
            pop_welfare += (type_counts[i] / total_pop) * fitness[i]
        
        # Selection probabilities
        selection_probs = np.zeros(n_types)
        for i in range(n_types):
            selection_probs[i] = type_counts[i] * np.exp(selection_strength * fitness[i])
        
        if selection_probs.sum() > 0:
            selection_probs /= selection_probs.sum()
        else:
            selection_probs = type_counts / total_pop
        
        death_probs = type_counts / total_pop
        
        if rng.random() < mutation_rate:
            reproducer = rng.integers(0, n_types)
        else:
            reproducer = rng.choice(n_types, p=selection_probs)
        
        dying = rng.choice(n_types, p=death_probs)
        
        if population[dying] > 0:
            population[dying] -= 1
            population[reproducer] += 1
        
        # Record after burn-in
        if gen >= burn_in:
            abundance_samples.append(population.copy() / N)
            welfare_samples.append(pop_welfare)
    
    mean_abundance = np.mean(abundance_samples, axis=0)
    mean_welfare = np.mean(welfare_samples)
    
    return mean_abundance, mean_welfare


# =============================================================================
# Figure 3E Configuration and Simulation
# =============================================================================

@dataclass
class Figure3EConfig:
    N: int = 10
    selection_strength: float = 2.0
    mutation_rate: float = 0.001
    omega: float = 0.5
    n_samples: int = 50
    n_generations: int = 10000
    burn_in: int = 3000
    n_mc_runs: int = 10
    T_values: List[int] = None
    epsilon_values: List[float] = None
    seed: int = 42


def run_figure_3e_simulations(config: Figure3EConfig = None, verbose=True) -> Dict:
    """Run Figure 3E: Population welfare over (T, ε_a) grid."""
    if config is None:
        config = Figure3EConfig()
    
    if config.T_values is None:
        config.T_values = [1, 2, 5, 10, 20, 50]
    if config.epsilon_values is None:
        config.epsilon_values = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]
    
    strategy_types = STRATEGY_TYPES
    tgmi_idx = strategy_types.index('TGMI')
    
    n_T = len(config.T_values)
    n_eps = len(config.epsilon_values)
    
    # Pre-sample games
    master_rng = np.random.default_rng(config.seed)
    mgg = MoralGameGenerator(num_actions=11, rng=master_rng)
    games = [mgg.sample_game() for _ in range(config.n_samples)]
    
    if verbose:
        print(f"Pre-sampled {len(games)} games")
        print(f"Grid size: {n_T} T values × {n_eps} ε_a values = {n_T * n_eps} points")
    
    # Store results
    welfare_grid = np.zeros((n_eps, n_T))
    abundance_grid = np.zeros((n_eps, n_T))  # Also track TGMI abundance for comparison
    
    total_points = n_T * n_eps
    point_idx = 0
    
    for i, eps_a in enumerate(config.epsilon_values):
        for j, T in enumerate(config.T_values):
            point_idx += 1
            if verbose:
                print(f"[{point_idx}/{total_points}] T={T}, ε_a={eps_a:.2f}...", end=" ")
            
            rng = np.random.default_rng(config.seed + i * 1000 + j * 10)
            
            # Compute payoff matrix
            payoff_matrix = compute_payoff_matrix(
                strategy_types, mgg, T,
                omega=config.omega,
                n_samples=config.n_samples,
                epsilon=eps_a,
                rng=rng,
                games=games,
            )
            
            # Run multiple Moran processes
            welfare_runs = []
            abundance_runs = []
            
            for mc_run in range(config.n_mc_runs):
                mc_rng = np.random.default_rng(config.seed + i * 1000 + j * 10 + mc_run)
                abundance, welfare = run_moran_monte_carlo_with_welfare(
                    payoff_matrix,
                    N=config.N,
                    selection_strength=config.selection_strength,
                    mutation_rate=config.mutation_rate,
                    n_generations=config.n_generations,
                    burn_in=config.burn_in,
                    rng=mc_rng,
                )
                welfare_runs.append(welfare)
                abundance_runs.append(abundance[tgmi_idx])
            
            mean_welfare = np.mean(welfare_runs)
            mean_abundance = np.mean(abundance_runs)
            
            welfare_grid[i, j] = mean_welfare
            abundance_grid[i, j] = mean_abundance
            
            if verbose:
                print(f"Welfare={mean_welfare:.3f}, TGMI={mean_abundance:.2f}")
    
    return {
        'T_values': np.array(config.T_values),
        'epsilon_values': np.array(config.epsilon_values),
        'welfare_grid': welfare_grid,
        'abundance_grid': abundance_grid,
        'strategy_types': strategy_types,
        'config': config,
    }


# =============================================================================
# Plotting
# =============================================================================

def plot_figure_3e(results: Dict, save_path: str = None):
    """
    Plot Figure 3E: Mean population welfare heatmap.
    
    Continuous colormap showing welfare across (T, ε_a) grid.
    No red overlay - pure welfare visualization.
    """
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
    })
    
    T_values = results['T_values']
    epsilon_values = results['epsilon_values']
    W = results['welfare_grid']  # Welfare
    
    n_T = len(T_values)
    n_eps = len(epsilon_values)
    
    fig, ax = plt.subplots(figsize=(9, 7))
    
    # Plot welfare heatmap with viridis
    im = ax.imshow(
        W,
        aspect='auto',
        origin='lower',
        cmap='viridis',
        extent=[0.5, n_T + 0.5, -0.5, n_eps - 0.5],
    )
    
    # Add text annotations for each cell (no asterisks)
    for i, eps in enumerate(epsilon_values):
        for j, T in enumerate(T_values):
            welfare = W[i, j]
            # Use white text on dark cells, black on light
            text_color = 'white' if welfare < 0.8 else 'black'
            ax.text(j + 1, i, f'{welfare:.2f}',
                   ha='center', va='center',
                   fontsize=10, fontweight='bold',
                   color=text_color,
                   zorder=3)
    
    # Set tick labels
    ax.set_xticks(np.arange(1, n_T + 1))
    ax.set_xticklabels([str(T) for T in T_values])
    ax.set_yticks(np.arange(n_eps))
    ax.set_yticklabels([f'{eps:.2f}' for eps in epsilon_values])
    
    # Labels
    ax.set_xlabel('Game length $T$')
    ax.set_ylabel(r'Action error $\varepsilon_a$')
    ax.set_title('E', fontweight='bold', loc='left', fontsize=16)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Mean population welfare', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to {save_path}")
    
    return fig


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("Running Figure 3E simulations...")
    print("=" * 60)
    print("Mean Population Welfare Analysis")
    print("Strategy types:", STRATEGY_TYPES)
    print("=" * 60)
    
    config = Figure3EConfig(
        N=10,
        selection_strength=2.0,
        mutation_rate=0.001,
        omega=0.5,
        n_samples=50,
        n_generations=10000,
        burn_in=3000,
        n_mc_runs=10,
        T_values=[1, 2, 5, 10, 20, 50],
        epsilon_values=[0.0, 0.05, 0.1, 0.15, 0.2, 0.3],
        seed=42,
    )
    
    results = run_figure_3e_simulations(config, verbose=True)
    
    print("\n" + "=" * 60)
    print("RESULTS: Population Welfare Grid (ε_a × T)")
    print("=" * 60)
    
    header = "ε_a \\ T |" + "|".join([f" {T:5d} " for T in results['T_values']]) + "|"
    print(header)
    print("-" * len(header))
    
    for i, eps in enumerate(results['epsilon_values']):
        row = f" {eps:.2f}   |"
        for j in range(len(results['T_values'])):
            welfare = results['welfare_grid'][i, j]
            tgmi = results['abundance_grid'][i, j]
            marker = "*" if tgmi > 0.5 else " "
            row += f" {welfare:.3f}{marker}|"
        print(row)
    
    print("\n* indicates TGMI majority (>0.5)")
    
    # Correlation analysis
    welfare_flat = results['welfare_grid'].flatten()
    abundance_flat = results['abundance_grid'].flatten()
    correlation = np.corrcoef(welfare_flat, abundance_flat)[0, 1]
    print(f"\nCorrelation between TGMI abundance and welfare: {correlation:.3f}")
    
    # Save plot
    output_dir = os.path.join(_parent_dir, 'outputs', 'plots')
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'figure_3e.png')
    
    fig = plot_figure_3e(results, save_path=save_path)
    plt.show()
    
    print("\nDone!")
