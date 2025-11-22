"""
Visualization script for the Moral Game Generator (MGG).

Creates heatmaps showing payoffs and fairness functions for each archetype.
"""

import numpy as np
import matplotlib.pyplot as plt
from mgg import MoralGameGenerator, Archetype
from config import DEFAULT_HYPERPARAMS


def visualize_game(game, title="Game Visualization"):
    """Create a visualization of a game's payoffs and fairness functions."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"{title}\nArchetype: {game.archetype.name}", fontsize=14, fontweight='bold')
    
    # Payoff matrices
    im0 = axes[0, 0].imshow(game.R_i, cmap='viridis', origin='lower', vmin=0, vmax=1)
    axes[0, 0].set_title("Payoff R_i (Player i)")
    axes[0, 0].set_xlabel("Player j action index")
    axes[0, 0].set_ylabel("Player i action index")
    plt.colorbar(im0, ax=axes[0, 0])
    
    im1 = axes[0, 1].imshow(game.R_j, cmap='viridis', origin='lower', vmin=0, vmax=1)
    axes[0, 1].set_title("Payoff R_j (Player j)")
    axes[0, 1].set_xlabel("Player j action index")
    axes[0, 1].set_ylabel("Player i action index")
    plt.colorbar(im1, ax=axes[0, 1])
    
    # Total payoff
    total_payoff = game.R_i + game.R_j
    im2 = axes[0, 2].imshow(total_payoff, cmap='viridis', origin='lower')
    axes[0, 2].set_title("Total Payoff (R_i + R_j)")
    axes[0, 2].set_xlabel("Player j action index")
    axes[0, 2].set_ylabel("Player i action index")
    plt.colorbar(im2, ax=axes[0, 2])
    
    # Fairness functions
    im3 = axes[1, 0].imshow(game.F["max_sum"], cmap='RdYlGn', origin='lower', vmin=0, vmax=1)
    axes[1, 0].set_title("F_Max-Sum")
    axes[1, 0].set_xlabel("Player j action index")
    axes[1, 0].set_ylabel("Player i action index")
    plt.colorbar(im3, ax=axes[1, 0])
    
    im4 = axes[1, 1].imshow(game.F["equal_split"], cmap='RdYlGn', origin='lower', vmin=0, vmax=1)
    axes[1, 1].set_title("F_Equal-Split")
    axes[1, 1].set_xlabel("Player j action index")
    axes[1, 1].set_ylabel("Player i action index")
    plt.colorbar(im4, ax=axes[1, 1])
    
    im5 = axes[1, 2].imshow(game.F["rawls"], cmap='RdYlGn', origin='lower', vmin=0, vmax=1)
    axes[1, 2].set_title("F_Rawls (min payoff)")
    axes[1, 2].set_xlabel("Player j action index")
    axes[1, 2].set_ylabel("Player i action index")
    plt.colorbar(im5, ax=axes[1, 2])
    
    plt.tight_layout()
    return fig


def main():
    """Generate visualizations for each archetype."""
    rng = np.random.default_rng(42)
    mgg = MoralGameGenerator(rng=rng, num_actions=21)
    
    # Generate one game per archetype
    archetype_probs = {
        Archetype.DILEMMA: [1, 0, 0, 0],
        Archetype.ASSURANCE: [0, 1, 0, 0],
        Archetype.BARGAIN: [0, 0, 1, 0],
        Archetype.PUBLIC_GOODS: [0, 0, 0, 1],
    }
    
    for archetype, probs in archetype_probs.items():
        mgg.archetype_probs = np.array(probs)
        game = mgg.sample_game()
        
        fig = visualize_game(game, f"Sample {archetype.name} Game")
        plt.savefig(f"phase1_viz_{archetype.name.lower()}.png", dpi=150, bbox_inches='tight')
        print(f"Saved visualization for {archetype.name}")
    
    print("\n" + "="*60)
    print("Phase 1 Visualization Complete!")
    print("="*60)
    print("\nGenerated files:")
    for archetype in Archetype:
        print(f"  - phase1_viz_{archetype.name.lower()}.png")
    
    # Print statistics
    print("\n" + "="*60)
    print("Game Statistics (100 random samples)")
    print("="*60)
    
    mgg.archetype_probs = np.ones(4) / 4  # uniform
    payoff_ranges = []
    fairness_ranges = {phi: [] for phi in ["max_sum", "equal_split", "rawls"]}
    
    for _ in range(100):
        game = mgg.sample_game()
        payoff_ranges.append((game.R_i.min(), game.R_i.max(), game.R_j.min(), game.R_j.max()))
        for phi in fairness_ranges:
            fairness_ranges[phi].append((game.F[phi].min(), game.F[phi].max()))
    
    print(f"\nPayoff ranges (R_i and R_j):")
    print(f"  Min: {np.mean([p[0] for p in payoff_ranges]):.3f}")
    print(f"  Max: {np.mean([p[1] for p in payoff_ranges]):.3f}")
    
    for phi in fairness_ranges:
        mins = [f[0] for f in fairness_ranges[phi]]
        maxs = [f[1] for f in fairness_ranges[phi]]
        print(f"\nF_{phi} ranges:")
        print(f"  Min: {np.mean(mins):.3f} ± {np.std(mins):.3f}")
        print(f"  Max: {np.mean(maxs):.3f} ± {np.std(maxs):.3f}")


if __name__ == "__main__":
    main()
