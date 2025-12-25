"""
Figure 3B: Steady-state abundance of TGMI as a function of game length T
=========================================================================

Implements the Moran process evolutionary dynamics from SI S3.5:
1. Population of N=10 agents with different strategy types
2. Each generation: agents play T rounds and accumulate fitness
3. Softmax selection + mutation determines reproduction
4. Compute stationary distribution via power iteration
5. Extract TGMI's expected abundance in stationarity

Strategy types: TGMI, TFT, GTFT, WSLS, Forgiver, AllC, AllD
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from itertools import combinations_with_replacement
from scipy.special import comb
import sys
import os

# Add parent directory to path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
sys.path.insert(0, _parent_dir)

from tgmi.agent import (
    TGMIAgent, TGMIConfig, create_tgmi_agent,
    FairnessPrinciple, FAIRNESS_KEYS, normalized,
)
from MGG.generator import MoralGameGenerator, Game


# =============================================================================
# Strategy Implementations (Classic IPD-style strategies adapted for MGG)
# =============================================================================

class BaseStrategy:
    """Base class for all strategies."""
    
    def __init__(self, rng: np.random.Generator = None, epsilon: float = 0.0):
        self.rng = rng if rng is not None else np.random.default_rng()
        self.epsilon = epsilon
        self.name = "Base"
    
    def reset(self):
        """Reset any internal state."""
        pass
    
    def select_action(self, game: Game, history: List[Tuple[int, int]] = None) -> int:
        """Select an action given the game and history of (my_action, partner_action)."""
        raise NotImplementedError
    
    def _apply_noise(self, action: int, n_actions: int) -> int:
        """Apply action noise (trembling hand)."""
        if self.rng.random() < self.epsilon:
            return self.rng.integers(0, n_actions)
        return action


class AllC(BaseStrategy):
    """Always Cooperate: always choose the highest action (full cooperation)."""
    
    def __init__(self, rng=None, epsilon=0.0):
        super().__init__(rng, epsilon)
        self.name = "AllC"
    
    def select_action(self, game: Game, history=None) -> int:
        max_action = len(game.action_space_j) - 1
        return self._apply_noise(max_action, len(game.action_space_j))


class AllD(BaseStrategy):
    """Always Defect: always choose the lowest action (full defection)."""
    
    def __init__(self, rng=None, epsilon=0.0):
        super().__init__(rng, epsilon)
        self.name = "AllD"
    
    def select_action(self, game: Game, history=None) -> int:
        return self._apply_noise(0, len(game.action_space_j))


class TFT(BaseStrategy):
    """Tit-for-Tat: cooperate first, then copy opponent's last action."""
    
    def __init__(self, rng=None, epsilon=0.0):
        super().__init__(rng, epsilon)
        self.name = "TFT"
    
    def select_action(self, game: Game, history=None) -> int:
        n_actions = len(game.action_space_j)
        max_action = n_actions - 1
        
        if history is None or len(history) == 0:
            # First round: cooperate
            action = max_action
        else:
            # Copy opponent's last action
            _, partner_last = history[-1]
            action = partner_last
        
        return self._apply_noise(action, n_actions)


class GTFT(BaseStrategy):
    """Generous Tit-for-Tat: like TFT but occasionally forgives defection."""
    
    def __init__(self, rng=None, epsilon=0.0, generosity=0.33):
        super().__init__(rng, epsilon)
        self.name = "GTFT"
        self.generosity = generosity
    
    def select_action(self, game: Game, history=None) -> int:
        n_actions = len(game.action_space_j)
        max_action = n_actions - 1
        
        if history is None or len(history) == 0:
            action = max_action
        else:
            _, partner_last = history[-1]
            # Forgive with probability 'generosity'
            if partner_last < max_action * 0.5 and self.rng.random() < self.generosity:
                action = max_action
            else:
                action = partner_last
        
        return self._apply_noise(action, n_actions)


class WSLS(BaseStrategy):
    """Win-Stay Lose-Shift: repeat if payoff was good, switch otherwise."""
    
    def __init__(self, rng=None, epsilon=0.0):
        super().__init__(rng, epsilon)
        self.name = "WSLS"
        self.last_payoff = None
        self.last_action = None
    
    def reset(self):
        self.last_payoff = None
        self.last_action = None
    
    def select_action(self, game: Game, history=None) -> int:
        n_actions = len(game.action_space_j)
        max_action = n_actions - 1
        
        if history is None or len(history) == 0 or self.last_payoff is None:
            # First round: cooperate
            action = max_action
        else:
            # Check if last outcome was "good" (above median payoff)
            median_payoff = 0.5  # Normalized payoffs are in [0, 1]
            
            if self.last_payoff >= median_payoff:
                # Win: stay with last action
                action = self.last_action
            else:
                # Lose: shift to opposite
                if self.last_action > max_action * 0.5:
                    action = 0  # Shift to defect
                else:
                    action = max_action  # Shift to cooperate
        
        return self._apply_noise(action, n_actions)
    
    def update_payoff(self, action: int, payoff: float):
        """Update internal state with last payoff."""
        self.last_action = action
        self.last_payoff = payoff


class Forgiver(BaseStrategy):
    """Forgiver: like TFT but requires 2 consecutive defections before retaliating."""
    
    def __init__(self, rng=None, epsilon=0.0):
        super().__init__(rng, epsilon)
        self.name = "Forgiver"
        self.defection_count = 0
    
    def reset(self):
        self.defection_count = 0
    
    def select_action(self, game: Game, history=None) -> int:
        n_actions = len(game.action_space_j)
        max_action = n_actions - 1
        threshold = max_action * 0.5
        
        if history is None or len(history) == 0:
            action = max_action
        else:
            _, partner_last = history[-1]
            
            if partner_last < threshold:
                self.defection_count += 1
            else:
                self.defection_count = 0
            
            # Only retaliate after 2 consecutive defections
            if self.defection_count >= 2:
                action = 0
            else:
                action = max_action
        
        return self._apply_noise(action, n_actions)


class TGMIStrategy(BaseStrategy):
    """TGMI agent wrapped as a strategy for the Moran process."""
    
    def __init__(self, rng=None, epsilon=0.0, eta=0.0, config=None):
        super().__init__(rng, epsilon)
        self.name = "TGMI"
        self.eta = eta
        self.config = config
        self.agent = None
        self._initialize_agent()
    
    def _initialize_agent(self):
        """Create a fresh TGMI agent."""
        config = self.config if self.config else TGMIConfig(
            theta_0=0.5,
            phi=0.2,
            beta=1.0,
            xi_dev=8.0,
            gamma_bargain=0.5,
            gamma_mixture=0.1,
            epsilon=self.epsilon,
            eta=self.eta,
        )
        self.agent = create_tgmi_agent(
            moral_prior=None,  # Random prior
            theta_0=config.theta_0,
            phi=config.phi,
            beta=config.beta,
            xi_dev=config.xi_dev,
            gamma_bargain=config.gamma_bargain,
            gamma_mixture=config.gamma_mixture,
            epsilon=self.epsilon,
            eta=self.eta,
            rng=self.rng,
        )
    
    def reset(self):
        """Reset the TGMI agent."""
        self._initialize_agent()
    
    def select_action(self, game: Game, history=None) -> int:
        """Select action using TGMI's VB mechanism."""
        action, info = self.agent.select_action(game)
        return action  # Noise is handled internally by agent
    
    def update(self, game: Game, my_action: int, partner_action: int, info: dict):
        """Update TGMI's beliefs after observing partner's action."""
        # Observe partner action (with noise if configured)
        partner_observed = self.agent.observe_partner_action(partner_action, game)
        
        # Get VB info if not provided
        if 'U_i' not in info:
            _, info = self.agent.select_action(game)
        
        self.agent.update(
            game=game,
            a_i_VB=info.get('a_i_VB', my_action),
            a_j_actual=partner_observed,
            U_i=info['U_i'],
            U_j_hat=info['U_j_hat'],
        )


# =============================================================================
# Strategy Factory
# =============================================================================

STRATEGY_TYPES = ['TGMI', 'TFT', 'GTFT', 'WSLS', 'Forgiver', 'AllC', 'AllD']

def create_strategy(
    strategy_type: str,
    rng: np.random.Generator = None,
    epsilon: float = 0.05,
    eta: float = 0.0,
    tgmi_config: TGMIConfig = None,
) -> BaseStrategy:
    """Create a strategy instance by name."""
    if rng is None:
        rng = np.random.default_rng()
    
    if strategy_type == 'TGMI':
        return TGMIStrategy(rng=rng, epsilon=epsilon, eta=eta, config=tgmi_config)
    elif strategy_type == 'TFT':
        return TFT(rng=rng, epsilon=epsilon)
    elif strategy_type == 'GTFT':
        return GTFT(rng=rng, epsilon=epsilon)
    elif strategy_type == 'WSLS':
        return WSLS(rng=rng, epsilon=epsilon)
    elif strategy_type == 'Forgiver':
        return Forgiver(rng=rng, epsilon=epsilon)
    elif strategy_type == 'AllC':
        return AllC(rng=rng, epsilon=epsilon)
    elif strategy_type == 'AllD':
        return AllD(rng=rng, epsilon=epsilon)
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")


# =============================================================================
# Pairwise Interaction & Fitness Computation
# =============================================================================

def play_match(
    strategy_i: BaseStrategy,
    strategy_j: BaseStrategy,
    game: Game,
    T: int,
    omega: float = 0.5,
) -> Tuple[float, float]:
    """
    Play T rounds between two strategies and compute fitness.
    
    Fitness is fairness-weighted return (SI S3.5):
    Φ_i = (1/T) * Σ [(1-ω)*R_i + ω*U_F_i]
    
    Args:
        strategy_i, strategy_j: The two strategies
        game: The game being played
        T: Number of rounds
        omega: Fairness weight (0=payoff-only, 1=fairness-only)
    
    Returns:
        (fitness_i, fitness_j): Fitness values for both strategies
    """
    strategy_i.reset()
    strategy_j.reset()
    
    history_i = []  # (my_action, partner_action)
    history_j = []
    
    total_payoff_i = 0.0
    total_payoff_j = 0.0
    total_fairness_i = 0.0
    total_fairness_j = 0.0
    
    for t in range(T):
        # Select actions
        if isinstance(strategy_i, TGMIStrategy):
            action_i, info_i = strategy_i.agent.select_action(game)
        else:
            action_i = strategy_i.select_action(game, history_i)
            info_i = {}
        
        if isinstance(strategy_j, TGMIStrategy):
            action_j, info_j = strategy_j.agent.select_action(game)
        else:
            action_j = strategy_j.select_action(game, history_j)
            info_j = {}
        
        # Get payoffs
        payoff_i = game.R_i[action_i, action_j]
        payoff_j = game.R_j[action_i, action_j]
        
        # Compute fairness utility (using Equal-Split as the fairness metric)
        fairness_i = game.F['equal_split'][action_i, action_j]
        fairness_j = game.F['equal_split'][action_i, action_j]  # Symmetric
        
        total_payoff_i += payoff_i
        total_payoff_j += payoff_j
        total_fairness_i += fairness_i
        total_fairness_j += fairness_j
        
        # Update histories
        history_i.append((action_i, action_j))
        history_j.append((action_j, action_i))
        
        # Update WSLS internal state
        if isinstance(strategy_i, WSLS):
            strategy_i.update_payoff(action_i, payoff_i)
        if isinstance(strategy_j, WSLS):
            strategy_j.update_payoff(action_j, payoff_j)
        
        # Update TGMI beliefs
        if isinstance(strategy_i, TGMIStrategy):
            if 'U_i' in info_i:
                strategy_i.update(game, action_i, action_j, info_i)
        if isinstance(strategy_j, TGMIStrategy):
            if 'U_i' in info_j:
                strategy_j.update(game, action_j, action_i, info_j)
    
    # Compute fitness: Φ = (1/T) * [(1-ω)*R + ω*U_F]
    fitness_i = (1.0 / T) * ((1 - omega) * total_payoff_i + omega * total_fairness_i)
    fitness_j = (1.0 / T) * ((1 - omega) * total_payoff_j + omega * total_fairness_j)
    
    return fitness_i, fitness_j


def compute_payoff_matrix(
    strategy_types: List[str],
    mgg: MoralGameGenerator,
    T: int,
    omega: float = 0.5,
    n_samples: int = 50,
    epsilon: float = 0.05,
    eta: float = 0.0,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Compute the expected payoff matrix for all strategy pairs.
    
    Args:
        strategy_types: List of strategy names
        mgg: Moral Game Generator
        T: Rounds per match
        omega: Fairness weight
        n_samples: Number of game samples for averaging
        epsilon: Action noise
        eta: Observation noise
        rng: Random generator
    
    Returns:
        Payoff matrix A where A[i,j] = expected fitness of type i vs type j
    """
    if rng is None:
        rng = np.random.default_rng()
    
    n_types = len(strategy_types)
    payoff_matrix = np.zeros((n_types, n_types))
    
    for i, type_i in enumerate(strategy_types):
        for j, type_j in enumerate(strategy_types):
            total_fitness_i = 0.0
            
            for _ in range(n_samples):
                # Sample a new game
                game = mgg.sample_game()
                
                # Create fresh strategies
                strat_i = create_strategy(type_i, rng=rng, epsilon=epsilon, eta=eta)
                strat_j = create_strategy(type_j, rng=rng, epsilon=epsilon, eta=eta)
                
                # Play match
                fitness_i, _ = play_match(strat_i, strat_j, game, T, omega)
                total_fitness_i += fitness_i
            
            payoff_matrix[i, j] = total_fitness_i / n_samples
    
    return payoff_matrix


# =============================================================================
# Moran Process Implementation
# =============================================================================

def enumerate_states(N: int, n_types: int) -> List[Tuple[int, ...]]:
    """
    Enumerate all possible population states.
    A state is a tuple (n_0, n_1, ..., n_{k-1}) where n_i is count of type i.
    Sum of all n_i = N.
    
    Returns:
        List of all valid state tuples
    """
    states = []
    
    def generate(remaining: int, types_left: int, current: List[int]):
        if types_left == 1:
            states.append(tuple(current + [remaining]))
            return
        
        for count in range(remaining + 1):
            generate(remaining - count, types_left - 1, current + [count])
    
    generate(N, n_types, [])
    return states


def state_to_index(state: Tuple[int, ...], states: List[Tuple[int, ...]]) -> int:
    """Get index of a state in the states list."""
    return states.index(state)


def compute_moran_transition_matrix(
    payoff_matrix: np.ndarray,
    N: int,
    selection_strength: float = 2.0,
    mutation_rate: float = 1e-3,
) -> Tuple[np.ndarray, List[Tuple[int, ...]]]:
    """
    Compute the Moran process transition matrix.
    
    The Moran process:
    1. Compute fitness of each type from payoff matrix
    2. Select reproducer with probability ∝ exp(s * fitness)
    3. Select random individual to die
    4. With probability μ, mutate to random type
    
    Args:
        payoff_matrix: Expected payoffs A[i,j] = fitness of i vs j
        N: Population size
        selection_strength: s in softmax selection
        mutation_rate: μ for mutation
    
    Returns:
        (P, states): Transition matrix and list of states
    """
    n_types = payoff_matrix.shape[0]
    states = enumerate_states(N, n_types)
    n_states = len(states)
    
    # Build transition matrix
    P = np.zeros((n_states, n_states))
    
    for idx, state in enumerate(states):
        state = np.array(state)
        
        # Compute average fitness for each type in this population
        # Fitness of type i = sum over j of (n_j/N) * A[i,j]
        type_fitness = np.zeros(n_types)
        for i in range(n_types):
            if state[i] > 0:
                for j in range(n_types):
                    type_fitness[i] += (state[j] / N) * payoff_matrix[i, j]
        
        # Softmax selection probabilities (only for types present)
        present = state > 0
        if present.sum() == 0:
            continue
        
        # Compute selection probabilities
        max_fitness = type_fitness[present].max()
        exp_fitness = np.zeros(n_types)
        exp_fitness[present] = np.exp(selection_strength * (type_fitness[present] - max_fitness))
        selection_probs = np.zeros(n_types)
        if exp_fitness.sum() > 0:
            selection_probs = (state * exp_fitness) / (state * exp_fitness).sum()
        
        # Death probabilities (uniform)
        death_probs = state / N
        
        # Compute transitions
        for reproducer in range(n_types):
            if state[reproducer] == 0:
                continue
            
            p_select = selection_probs[reproducer]
            
            for dying in range(n_types):
                if state[dying] == 0:
                    continue
                
                p_die = death_probs[dying]
                
                # What types can the offspring be?
                # With prob (1-μ): same as reproducer
                # With prob μ: uniform random type
                
                for offspring_type in range(n_types):
                    if offspring_type == reproducer:
                        p_offspring = (1 - mutation_rate) + mutation_rate / n_types
                    else:
                        p_offspring = mutation_rate / n_types
                    
                    # Compute new state
                    new_state = list(state)
                    new_state[dying] -= 1
                    new_state[offspring_type] += 1
                    new_state = tuple(new_state)
                    
                    # Find new state index
                    try:
                        new_idx = states.index(new_state)
                        P[idx, new_idx] += p_select * p_die * p_offspring
                    except ValueError:
                        pass  # Invalid state
    
    # Normalize rows (should already sum to 1, but ensure numerical stability)
    row_sums = P.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    P = P / row_sums
    
    return P, states


def compute_stationary_distribution(
    P: np.ndarray,
    tol: float = 1e-9,
    max_iter: int = 10000,
) -> np.ndarray:
    """
    Compute stationary distribution using power iteration.
    
    Args:
        P: Transition matrix
        tol: Convergence tolerance
        max_iter: Maximum iterations
    
    Returns:
        Stationary distribution π where π = πP
    """
    n = P.shape[0]
    
    # Start with uniform distribution
    pi = np.ones(n) / n
    
    for _ in range(max_iter):
        pi_new = pi @ P
        
        # Normalize
        pi_new = pi_new / pi_new.sum()
        
        # Check convergence
        if np.max(np.abs(pi_new - pi)) < tol:
            return pi_new
        
        pi = pi_new
    
    print(f"Warning: Power iteration did not converge after {max_iter} iterations")
    return pi


def compute_tgmi_abundance(
    pi: np.ndarray,
    states: List[Tuple[int, ...]],
    tgmi_index: int,
    N: int,
) -> float:
    """
    Compute expected TGMI abundance under stationary distribution.
    
    x*_TGMI = Σ_n (n_TGMI / N) * π(n)
    
    Args:
        pi: Stationary distribution
        states: List of population states
        tgmi_index: Index of TGMI in the strategy types list
        N: Population size
    
    Returns:
        Expected TGMI abundance (fraction in [0, 1])
    """
    abundance = 0.0
    
    for state_idx, state in enumerate(states):
        n_tgmi = state[tgmi_index]
        abundance += (n_tgmi / N) * pi[state_idx]
    
    return abundance


# =============================================================================
# Main Simulation for Figure 3B
# =============================================================================

@dataclass
class Figure3BConfig:
    """Configuration for Figure 3B simulations."""
    N: int = 10                     # Population size
    selection_strength: float = 2.0 # Selection strength s
    mutation_rate: float = 1e-3     # Mutation rate μ
    omega: float = 0.5              # Fairness weight in fitness
    epsilon_a: float = 0.05         # Action error
    eta: float = 0.0                # Observation error (perception noise)
    n_samples: int = 50             # Samples for payoff estimation
    T_values: List[int] = None      # Game lengths to sweep
    seed: int = 42


def run_figure_3b_simulations(
    config: Figure3BConfig = None,
    verbose: bool = True,
) -> Dict:
    """
    Run Figure 3B simulations: TGMI abundance vs game length T.
    
    Returns:
        Dictionary with T values and corresponding TGMI abundances
    """
    if config is None:
        config = Figure3BConfig()
    
    if config.T_values is None:
        config.T_values = [1, 2, 5, 10, 20, 50, 100]
    
    rng = np.random.default_rng(config.seed)
    mgg = MoralGameGenerator(num_actions=11, rng=rng)
    
    strategy_types = STRATEGY_TYPES  # ['TGMI', 'TFT', 'GTFT', 'WSLS', 'Forgiver', 'AllC', 'AllD']
    tgmi_index = strategy_types.index('TGMI')
    
    T_values = config.T_values
    abundances = []
    
    for T in T_values:
        if verbose:
            print(f"Computing for T={T}...")
        
        # Compute payoff matrix
        payoff_matrix = compute_payoff_matrix(
            strategy_types=strategy_types,
            mgg=mgg,
            T=T,
            omega=config.omega,
            n_samples=config.n_samples,
            epsilon=config.epsilon_a,
            eta=config.eta,
            rng=rng,
        )
        
        if verbose:
            print(f"  Payoff matrix computed. Building transition matrix...")
        
        # Compute Moran transition matrix
        P, states = compute_moran_transition_matrix(
            payoff_matrix=payoff_matrix,
            N=config.N,
            selection_strength=config.selection_strength,
            mutation_rate=config.mutation_rate,
        )
        
        if verbose:
            print(f"  Transition matrix: {P.shape[0]} states. Computing stationary distribution...")
        
        # Compute stationary distribution
        pi = compute_stationary_distribution(P, tol=1e-9)
        
        # Compute TGMI abundance
        abundance = compute_tgmi_abundance(pi, states, tgmi_index, config.N)
        abundances.append(abundance)
        
        if verbose:
            print(f"  TGMI abundance at T={T}: {abundance:.4f}")
    
    return {
        'T_values': np.array(T_values),
        'abundances': np.array(abundances),
        'config': config,
        'strategy_types': strategy_types,
    }


# =============================================================================
# Plotting
# =============================================================================

def plot_figure_3b(
    results: Dict,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 6),
) -> plt.Figure:
    """
    Plot Figure 3B: TGMI abundance vs game length T.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
    })
    
    T_values = results['T_values']
    abundances = results['abundances']
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot TGMI abundance
    ax.plot(
        T_values,
        abundances,
        'o-',
        color='#2E86AB',
        linewidth=2.5,
        markersize=8,
        markerfacecolor='white',
        markeredgewidth=2,
        label='TGMI',
    )
    
    # Reference line at 1/n_types (random baseline)
    n_types = len(results['strategy_types'])
    ax.axhline(
        y=1.0/n_types,
        color='gray',
        linestyle='--',
        linewidth=1.5,
        alpha=0.7,
        label=f'Random baseline (1/{n_types})',
    )
    
    # Reference line at 0.5 (majority threshold)
    ax.axhline(
        y=0.5,
        color='red',
        linestyle=':',
        linewidth=1.5,
        alpha=0.7,
        label='Majority threshold',
    )
    
    ax.set_xlabel('Game length $T$')
    ax.set_ylabel('Steady-state TGMI abundance')
    ax.set_title('B', fontweight='bold', loc='left', fontsize=16)
    
    # Use log scale for x-axis since T spans orders of magnitude
    ax.set_xscale('log')
    ax.set_xlim(0.8, 150)
    ax.set_ylim(0, 1.0)
    
    ax.legend(loc='lower right', framealpha=0.95)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to {save_path}")
    
    return fig


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    print("Running Figure 3B simulations...")
    print("=" * 50)
    print("This computes TGMI abundance vs game length T")
    print("Using Moran process with N=10 agents")
    print("Strategy types:", STRATEGY_TYPES)
    print("=" * 50)
    
    config = Figure3BConfig(
        N=10,
        selection_strength=2.0,
        mutation_rate=1e-3,
        omega=0.5,
        epsilon_a=0.05,
        eta=0.0,
        n_samples=30,  # Fewer samples for faster runtime
        T_values=[1, 2, 5, 10, 20, 50, 100],
        seed=42,
    )
    
    results = run_figure_3b_simulations(config, verbose=True)
    
    print("\n" + "=" * 50)
    print("RESULTS: TGMI Abundance vs Game Length T")
    print("=" * 50)
    for T, abundance in zip(results['T_values'], results['abundances']):
        status = "MAJORITY" if abundance > 0.5 else ""
        print(f"  T={T:3d}: {abundance:.4f} {status}")
    
    # Create output directory
    output_dir = os.path.join(_parent_dir, 'outputs', 'plots')
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot and save figure
    save_path = os.path.join(output_dir, 'figure_3b.png')
    fig = plot_figure_3b(results, save_path=save_path)
    
    plt.show()
    
    print("\nDone!")
