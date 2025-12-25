"""
Figure 3A: Intragenerational belief and trust updating
=====================================================

Simulates a probe TGMI agent interacting with three partner types:
1. Another TGMI agent
2. A Fairness-Aligned agent (maximizes Equal-Split fairness)
3. A Self-Interested agent (maximizes own material payoff)

For each partner type, we run 1000 independent simulations over 20 rounds,
tracking the probe agent's beliefs over the partner's moral priors and trust.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Callable, Optional
from dataclasses import dataclass
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
# Partner Type Implementations
# =============================================================================

class FairnessAlignedPartner:
    """
    Partner that maximizes fairness according to a fixed moral principle.
    Default: Equal-Split (the most "fair" principle in conventional sense).
    
    This partner doesn't learn or update - it just picks the action that
    maximizes F_phi(a_i, a_j) for its fixed principle phi.
    """
    
    def __init__(
        self,
        principle: FairnessPrinciple = FairnessPrinciple.EQUAL_SPLIT,
        rng: np.random.Generator = None,
        epsilon: float = 0.0,  # Action noise
    ):
        self.principle = principle
        self.rng = rng if rng is not None else np.random.default_rng()
        self.epsilon = epsilon
        
        # True moral prior: concentrated on the chosen principle
        self._true_prior = np.zeros(3)
        self._true_prior[principle.value] = 1.0
    
    @property
    def true_moral_prior(self) -> np.ndarray:
        """Returns the partner's true moral prior (for plotting dashed lines)."""
        return self._true_prior.copy()
    
    def select_action(self, game: Game, probe_action: int) -> int:
        """
        Select action that maximizes fairness for the given principle.
        
        Args:
            game: The game being played
            probe_action: The action taken by the probe agent (a_i)
        
        Returns:
            The fairness-maximizing action a_j
        """
        F_table = game.F
        key = FAIRNESS_KEYS[self.principle]
        F_phi = F_table[key]
        
        # F_phi has shape (n_ai, n_aj) - we want to maximize over a_j given a_i
        fairness_values = F_phi[probe_action, :]
        
        # Argmax with tie-breaking
        best_action = int(np.argmax(fairness_values))
        
        # Apply action noise
        if self.rng.random() < self.epsilon:
            n_actions = len(game.action_space_j)
            best_action = self.rng.integers(0, n_actions)
        
        return best_action


class SelfInterestedPartner:
    """
    Partner that maximizes its own material payoff R_j, ignoring VB.
    This is a truly exploitative agent that ONLY cares about its own payoff.
    
    Key behavior: argmax_{a_j} R_j(a_i, a_j)
    This will systematically deviate from fair/cooperative actions,
    causing the probe agent's trust to decay.
    """
    
    def __init__(
        self,
        rng: np.random.Generator = None,
        epsilon: float = 0.0,  # Action noise
    ):
        self.rng = rng if rng is not None else np.random.default_rng()
        self.epsilon = epsilon
        
        # True moral prior: Pure Max-Sum but with ZERO weight on partner
        # This represents pure selfishness - we visualize as [1, 0, 0]
        # but the agent ONLY maximizes its own payoff, not joint welfare
        self._true_prior = np.array([1.0, 0.0, 0.0])  # Max-Sum (selfish)
    
    @property
    def true_moral_prior(self) -> np.ndarray:
        """Returns the partner's 'true' moral prior for visualization."""
        return self._true_prior.copy()
    
    def select_action(self, game: Game, probe_action: int) -> int:
        """
        Select action that maximizes OWN material payoff while MINIMIZING 
        fairness to the probe. This is explicitly exploitative.
        
        The key insight: to make trust decay, we need the partner's action
        to systematically deviate from what fairness principles would suggest.
        
        Strategy: Choose the action that minimizes Equal-Split fairness
        (maximizes inequality in payoffs, favoring self).
        
        Args:
            game: The game being played
            probe_action: The action taken by the probe agent (a_i)
        
        Returns:
            The exploitative action a_j
        """
        R_j = game.R_j  # Shape: (n_ai, n_aj) - partner's payoff
        R_i = game.R_i  # Shape: (n_ai, n_aj) - probe's payoff
        
        # Exploitative utility: maximize (own payoff - probe payoff)
        # This explicitly seeks to exploit the probe
        exploitative_utility = R_j[probe_action, :] - R_i[probe_action, :]
        
        # Argmax with tie-breaking
        best_action = int(np.argmax(exploitative_utility))
        
        # Apply action noise
        if self.rng.random() < self.epsilon:
            n_actions = len(game.action_space_j)
            best_action = self.rng.integers(0, n_actions)
        
        return best_action


# =============================================================================
# Simulation Functions
# =============================================================================

def run_probe_vs_tgmi(
    probe: TGMIAgent,
    partner: TGMIAgent,
    game: Game,
    n_rounds: int = 20,
) -> Dict:
    """
    Run probe TGMI agent against another TGMI partner.
    Both agents update their beliefs and trust.
    
    Returns:
        Dictionary with belief and trust trajectories
    """
    probe.reset()
    partner.reset()
    
    # Log tensors: [n_rounds+1, n_principles] for beliefs, [n_rounds+1] for trust
    n_principles = probe.n_principles
    beliefs = np.zeros((n_rounds + 1, n_principles))
    trusts = np.zeros(n_rounds + 1)
    
    # Initial state (t=0)
    beliefs[0] = probe.B_hat.copy()
    trusts[0] = probe.theta
    
    for t in range(n_rounds):
        # Both agents select actions simultaneously
        a_i, info_i = probe.select_action_with_noise(game)
        a_j, info_j = partner.select_action_with_noise(game)
        
        # Observation with noise
        a_j_observed = probe.observe_partner_action(a_j, game)
        a_i_observed = partner.observe_partner_action(a_i, game)
        
        # Update both agents
        probe.update(
            game=game,
            a_i_VB=info_i['a_i_VB'],
            a_j_actual=a_j_observed,
            U_i=info_i['U_i'],
            U_j_hat=info_i['U_j_hat'],
        )
        partner.update(
            game=game,
            a_i_VB=info_j['a_i_VB'],
            a_j_actual=a_i_observed,
            U_i=info_j['U_i'],
            U_j_hat=info_j['U_j_hat'],
        )
        
        # Log probe's state after update
        beliefs[t + 1] = probe.B_hat.copy()
        trusts[t + 1] = probe.theta
    
    return {
        'beliefs': beliefs,
        'trusts': trusts,
        'partner_true_prior': partner.B_i.copy(),  # Partner's actual moral prior
    }


def run_probe_vs_fixed_partner(
    probe: TGMIAgent,
    partner,  # FairnessAlignedPartner or SelfInterestedPartner
    game: Game,
    n_rounds: int = 20,
) -> Dict:
    """
    Run probe TGMI agent against a fixed-strategy partner.
    Only the probe updates beliefs and trust.
    
    Returns:
        Dictionary with belief and trust trajectories
    """
    probe.reset()
    
    n_principles = probe.n_principles
    beliefs = np.zeros((n_rounds + 1, n_principles))
    trusts = np.zeros(n_rounds + 1)
    
    # Initial state (t=0)
    beliefs[0] = probe.B_hat.copy()
    trusts[0] = probe.theta
    
    for t in range(n_rounds):
        # Probe selects action
        a_i, info_i = probe.select_action_with_noise(game)
        
        # Partner responds based on its strategy
        a_j = partner.select_action(game, a_i)
        
        # Observation with noise
        a_j_observed = probe.observe_partner_action(a_j, game)
        
        # Update probe
        probe.update(
            game=game,
            a_i_VB=info_i['a_i_VB'],
            a_j_actual=a_j_observed,
            U_i=info_i['U_i'],
            U_j_hat=info_i['U_j_hat'],
        )
        
        # Log probe's state after update
        beliefs[t + 1] = probe.B_hat.copy()
        trusts[t + 1] = probe.theta
    
    return {
        'beliefs': beliefs,
        'trusts': trusts,
        'partner_true_prior': partner.true_moral_prior,
    }


def run_figure_3a_simulations(
    n_runs: int = 1000,
    n_rounds: int = 20,
    seed: int = 42,
    epsilon_a: float = 0.05,  # Action noise
    eta: float = 0.05,        # Observation noise
    verbose: bool = True,
) -> Dict:
    """
    Run all simulations for Figure 3A.
    
    Args:
        n_runs: Number of independent simulation runs per partner type
        n_rounds: Number of rounds per simulation
        seed: Random seed for reproducibility
        epsilon_a: Action error rate
        eta: Observation error rate
        verbose: Print progress
    
    Returns:
        Dictionary with all logged tensors:
        - beliefs: [3, n_runs, n_rounds+1, 3] - beliefs over partner's moral prior
        - trusts: [3, n_runs, n_rounds+1] - trust values
        - true_priors: [3, n_runs, 3] - partner's true moral priors
    """
    master_rng = np.random.default_rng(seed)
    
    # Tensor shapes
    n_partner_types = 3
    n_principles = 3
    
    # Output tensors
    beliefs = np.zeros((n_partner_types, n_runs, n_rounds + 1, n_principles))
    trusts = np.zeros((n_partner_types, n_runs, n_rounds + 1))
    true_priors = np.zeros((n_partner_types, n_runs, n_principles))
    
    # Game generator
    mgg = MoralGameGenerator(num_actions=11, rng=master_rng)
    
    partner_names = ['TGMI', 'Fairness-Aligned', 'Self-Interested']
    
    for run in range(n_runs):
        if verbose and (run + 1) % 100 == 0:
            print(f"Run {run + 1}/{n_runs}")
        
        # Generate seeds for this run
        run_seed = master_rng.integers(0, 2**31)
        
        # Sample a new game for each run
        game = mgg.sample_game()
        
        # --- Partner Type 0: TGMI Partner ---
        rng_0 = np.random.default_rng(run_seed)
        probe_0 = create_tgmi_agent(
            moral_prior=None,  # Will be sampled
            theta_0=0.5,
            phi=0.2,            # Faster trust learning for visible dynamics
            beta=1.0,
            xi_dev=8.0,         # Higher sensitivity to deviations
            gamma_bargain=0.5,
            gamma_mixture=0.1,
            epsilon=epsilon_a,
            eta=eta,
            rng=rng_0,
        )
        partner_tgmi = create_tgmi_agent(
            moral_prior=None,  # Will be sampled
            theta_0=0.5,
            phi=0.2,
            beta=1.0,
            xi_dev=8.0,
            gamma_bargain=0.5,
            gamma_mixture=0.1,
            epsilon=epsilon_a,
            eta=eta,
            rng=rng_0,
        )
        
        result_0 = run_probe_vs_tgmi(probe_0, partner_tgmi, game, n_rounds)
        beliefs[0, run] = result_0['beliefs']
        trusts[0, run] = result_0['trusts']
        true_priors[0, run] = result_0['partner_true_prior']
        
        # --- Partner Type 1: Fairness-Aligned Partner ---
        rng_1 = np.random.default_rng(run_seed)
        probe_1 = create_tgmi_agent(
            moral_prior=None,
            theta_0=0.5,
            phi=0.2,            # Faster trust learning for visible dynamics
            beta=1.0,
            xi_dev=8.0,         # Higher sensitivity to deviations
            gamma_bargain=0.5,
            gamma_mixture=0.1,
            epsilon=epsilon_a,
            eta=eta,
            rng=rng_1,
        )
        partner_fair = FairnessAlignedPartner(
            principle=FairnessPrinciple.EQUAL_SPLIT,
            rng=rng_1,
            epsilon=epsilon_a,
        )
        
        result_1 = run_probe_vs_fixed_partner(probe_1, partner_fair, game, n_rounds)
        beliefs[1, run] = result_1['beliefs']
        trusts[1, run] = result_1['trusts']
        true_priors[1, run] = result_1['partner_true_prior']
        
        # --- Partner Type 2: Self-Interested Partner ---
        rng_2 = np.random.default_rng(run_seed)
        probe_2 = create_tgmi_agent(
            moral_prior=None,
            theta_0=0.5,
            phi=0.2,            # Faster trust learning for visible dynamics
            beta=1.0,
            xi_dev=8.0,         # Higher sensitivity to deviations
            gamma_bargain=0.5,
            gamma_mixture=0.1,
            epsilon=epsilon_a,
            eta=eta,
            rng=rng_2,
        )
        partner_selfish = SelfInterestedPartner(
            rng=rng_2,
            epsilon=epsilon_a,
        )
        
        result_2 = run_probe_vs_fixed_partner(probe_2, partner_selfish, game, n_rounds)
        beliefs[2, run] = result_2['beliefs']
        trusts[2, run] = result_2['trusts']
        true_priors[2, run] = result_2['partner_true_prior']
    
    return {
        'beliefs': beliefs,  # [3, n_runs, n_rounds+1, 3]
        'trusts': trusts,    # [3, n_runs, n_rounds+1]
        'true_priors': true_priors,  # [3, n_runs, 3]
        'n_runs': n_runs,
        'n_rounds': n_rounds,
        'partner_names': partner_names,
    }


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_figure_3a(
    results: Dict,
    save_path: Optional[str] = None,
    show_individual_traces: bool = True,
    n_traces_to_show: int = 100,  # Number of individual traces to show (for clarity)
    trace_alpha: float = 0.08,
    figsize: Tuple[float, float] = (14, 8),
) -> plt.Figure:
    """
    Plot Figure 3A: Belief and trust trajectories for three partner types.
    
    Layout:
        3 columns (TGMI | Fairness-Aligned | Self-Interested)
        2 rows (Beliefs over moral priors | Trust)
    """
    # Set publication-quality style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'figure.dpi': 150,
    })
    
    beliefs = results['beliefs']  # [3, n_runs, n_rounds+1, 3]
    trusts = results['trusts']    # [3, n_runs, n_rounds+1]
    true_priors = results['true_priors']  # [3, n_runs, 3]
    n_runs = results['n_runs']
    n_rounds = results['n_rounds']
    partner_names = results['partner_names']
    
    # Time axis (rounds 1 to T, as in paper)
    rounds = np.arange(n_rounds + 1)
    
    # Principle names and colors (colorblind-friendly palette)
    principle_names = ['Max-Sum', 'Equal-Split', 'Rawls']
    belief_colors = ['#D55E00', '#0072B2', '#009E73']  # Orange, Blue, Green
    trust_color = '#CC79A7'  # Pink/Purple
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    # Random state for reproducible trace selection
    trace_rng = np.random.default_rng(123)
    
    for col, partner_type in enumerate(range(3)):
        # --- Row 0: Beliefs ---
        ax_belief = axes[0, col]
        
        for phi_idx in range(3):
            # Individual traces (faint)
            if show_individual_traces:
                indices = trace_rng.choice(n_runs, min(n_traces_to_show, n_runs), replace=False)
                for run_idx in indices:
                    ax_belief.plot(
                        rounds,
                        beliefs[partner_type, run_idx, :, phi_idx],
                        color=belief_colors[phi_idx],
                        alpha=trace_alpha,
                        linewidth=0.5,
                        zorder=1,
                    )
            
            # Mean trajectory (dark, on top)
            mean_belief = beliefs[partner_type, :, :, phi_idx].mean(axis=0)
            ax_belief.plot(
                rounds,
                mean_belief,
                color=belief_colors[phi_idx],
                linewidth=2.5,
                label=principle_names[phi_idx],
                zorder=3,
            )
        
        # Dashed lines for true moral priors
        mean_true_prior = true_priors[partner_type].mean(axis=0)
        
        if partner_type == 0:
            # TGMI partner: Show mean true prior across runs as dashed lines
            # (Partner priors vary per run, so we show the average)
            for phi_idx in range(3):
                ax_belief.axhline(
                    y=mean_true_prior[phi_idx],
                    color=belief_colors[phi_idx],
                    linestyle='--',
                    linewidth=1.5,
                    alpha=0.6,
                    zorder=2,
                )
        else:
            # Fixed partners: show true prior as horizontal dashed lines
            for phi_idx in range(3):
                if mean_true_prior[phi_idx] > 0.01:  # Only show non-zero
                    ax_belief.axhline(
                        y=mean_true_prior[phi_idx],
                        color=belief_colors[phi_idx],
                        linestyle='--',
                        linewidth=2.0,
                        alpha=0.8,
                        zorder=2,
                    )
        
        ax_belief.set_xlim(0, n_rounds)
        ax_belief.set_ylim(-0.02, 1.05)
        ax_belief.set_xlabel('Round $t$')
        if col == 0:
            ax_belief.set_ylabel('Belief $\\hat{B}_{i \\to j}(\\phi)$')
        ax_belief.set_title(f'{partner_names[col]} Partner', fontweight='bold')
        ax_belief.spines['top'].set_visible(False)
        ax_belief.spines['right'].set_visible(False)
        
        # Add legend with better placement
        if col == 2:
            ax_belief.legend(loc='upper right', framealpha=0.95, fontsize=8)
        
        # --- Row 1: Trust ---
        ax_trust = axes[1, col]
        
        # Add baseline reference at 0.5 (initial trust)
        ax_trust.axhline(y=0.5, color='gray', linestyle=':', linewidth=1.0, alpha=0.5, zorder=0)
        
        # Individual traces (faint)
        if show_individual_traces:
            indices = trace_rng.choice(n_runs, min(n_traces_to_show, n_runs), replace=False)
            for run_idx in indices:
                ax_trust.plot(
                    rounds,
                    trusts[partner_type, run_idx, :],
                    color=trust_color,
                    alpha=trace_alpha,
                    linewidth=0.5,
                    zorder=1,
                )
        
        # Mean trajectory (dark, thicker for emphasis)
        mean_trust = trusts[partner_type].mean(axis=0)
        ax_trust.plot(
            rounds,
            mean_trust,
            color=trust_color,
            linewidth=3.0,
            label='Trust $\\tau_i$',
            zorder=3,
        )
        
        # Also show 25th-75th percentile band
        q25 = np.percentile(trusts[partner_type], 25, axis=0)
        q75 = np.percentile(trusts[partner_type], 75, axis=0)
        ax_trust.fill_between(
            rounds,
            q25,
            q75,
            color=trust_color,
            alpha=0.2,
            zorder=2,
        )
        
        ax_trust.set_xlim(0, n_rounds)
        ax_trust.set_ylim(-0.02, 1.05)
        ax_trust.set_xlabel('Round $t$')
        if col == 0:
            ax_trust.set_ylabel('Trust $\\tau_i$')
        ax_trust.spines['top'].set_visible(False)
        ax_trust.spines['right'].set_visible(False)
        
        if col == 0:
            ax_trust.legend(loc='lower right', framealpha=0.9)
    
    plt.tight_layout()
    
    # Add panel label
    fig.text(0.01, 0.99, 'A', fontsize=18, fontweight='bold', va='top', ha='left')
    
    # Add overall figure title
    fig.suptitle(
        'Intragenerational belief and trust updating',
        fontsize=14,
        fontweight='bold',
        y=1.02,
    )
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to {save_path}")
    
    return fig


def print_summary_statistics(results: Dict):
    """Print summary statistics for the simulation results."""
    beliefs = results['beliefs']
    trusts = results['trusts']
    n_rounds = results['n_rounds']
    partner_names = results['partner_names']
    principle_names = ['Max-Sum', 'Equal-Split', 'Rawls']
    
    print("\n" + "="*70)
    print("FIGURE 3A SUMMARY STATISTICS")
    print("="*70)
    
    for p_idx, p_name in enumerate(partner_names):
        print(f"\n--- {p_name} Partner ---")
        
        # Final beliefs (at round T)
        final_beliefs = beliefs[p_idx, :, -1, :]  # [n_runs, 3]
        print(f"  Final beliefs (round {n_rounds}):")
        for phi_idx, phi_name in enumerate(principle_names):
            mean_b = final_beliefs[:, phi_idx].mean()
            std_b = final_beliefs[:, phi_idx].std()
            print(f"    {phi_name}: {mean_b:.3f} ± {std_b:.3f}")
        
        # Final trust
        final_trust = trusts[p_idx, :, -1]  # [n_runs]
        print(f"  Final trust: {final_trust.mean():.3f} ± {final_trust.std():.3f}")
        
        # Dominant inferred type
        dominant_types = np.argmax(final_beliefs, axis=1)
        for phi_idx, phi_name in enumerate(principle_names):
            pct = (dominant_types == phi_idx).mean() * 100
            print(f"    % runs inferring {phi_name}: {pct:.1f}%")


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    print("Running Figure 3A simulations...")
    print("=" * 50)
    
    # Run simulations
    results = run_figure_3a_simulations(
        n_runs=1000,
        n_rounds=20,
        seed=42,
        epsilon_a=0.05,
        eta=0.05,
        verbose=True,
    )
    
    # Print summary statistics
    print_summary_statistics(results)
    
    # Create output directory
    output_dir = os.path.join(_parent_dir, 'outputs', 'plots')
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot and save figure
    save_path = os.path.join(output_dir, 'figure_3a.png')
    fig = plot_figure_3a(
        results,
        save_path=save_path,
        show_individual_traces=True,
        n_traces_to_show=50,
        trace_alpha=0.1,
    )
    
    plt.show()
    
    print("\nDone!")
