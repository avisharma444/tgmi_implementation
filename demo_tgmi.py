"""
Demonstration of TGMI agent interactions.

Shows how trust and beliefs evolve over repeated interactions between:
1. Two TGMI agents with different moral priors
2. TGMI vs. consistently fair partner
3. TGMI vs. exploitative partner (simulated)
"""

import numpy as np
import matplotlib.pyplot as plt
from mgg import MoralGameGenerator, sample_moral_prior
from tgmi_agent import TGMI_Agent, tgmi_round
from config import Hyperparameters


def simulate_tgmi_pair(agent_i, agent_j, mgg, n_rounds=30):
    """
    Simulate repeated interactions between two TGMI agents.
    
    Returns:
        Dictionary with time series data for both agents.
    """
    data = {
        'round': [],
        'tau_i': [],
        'tau_j': [],
        'c_i': [],
        'c_j': [],
        'kappa_i': [],
        'kappa_j': [],
        'd_i': [],
        'd_j': [],
        'R_i': [],
        'R_j': [],
        'U_i': [],
        'U_j': [],
        'UF_i': [],
        'UF_j': [],
        'deviation_i': [],
        'deviation_j': [],
        'B_hat_i': [],
        'B_hat_j': [],
        'archetype': [],
    }
    
    for t in range(n_rounds):
        game = mgg.sample_game()
        R_i, R_j, info = tgmi_round(game, agent_i, agent_j)
        
        ps_i = agent_i.partners[agent_j.id]
        ps_j = agent_j.partners[agent_i.id]
        
        data['round'].append(t)
        data['tau_i'].append(ps_i.tau)
        data['tau_j'].append(ps_j.tau)
        data['c_i'].append(ps_i.c)
        data['c_j'].append(ps_j.c)
        data['kappa_i'].append(ps_i.kappa)
        data['kappa_j'].append(ps_j.kappa)
        data['d_i'].append(ps_i.d)
        data['d_j'].append(ps_j.d)
        data['R_i'].append(R_i)
        data['R_j'].append(R_j)
        data['U_i'].append(info['U_i_vb'])
        data['U_j'].append(info['U_j_vb'])
        data['UF_i'].append(info['UF_i_vb'])
        data['UF_j'].append(info['UF_j_vb'])
        data['deviation_i'].append(info['d_i'])
        data['deviation_j'].append(info['d_j'])
        data['B_hat_i'].append(ps_i.B_hat.copy())
        data['B_hat_j'].append(ps_j.B_hat.copy())
        data['archetype'].append(info['archetype'])
    
    return data


def plot_tgmi_dynamics(data, agent_i, agent_j, title="TGMI Dynamics"):
    """Plot key dynamics from TGMI interaction."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    rounds = data['round']
    
    # 1. Trust evolution
    axes[0, 0].plot(rounds, data['tau_i'], label='τ_i (i→j)', marker='o', markersize=3)
    axes[0, 0].plot(rounds, data['tau_j'], label='τ_j (j→i)', marker='s', markersize=3)
    axes[0, 0].set_xlabel('Round')
    axes[0, 0].set_ylabel('Trust τ')
    axes[0, 0].set_title('Trust Evolution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Confidence evolution
    axes[0, 1].plot(rounds, data['c_i'], label='c_i', marker='o', markersize=3)
    axes[0, 1].plot(rounds, data['c_j'], label='c_j', marker='s', markersize=3)
    axes[0, 1].set_xlabel('Round')
    axes[0, 1].set_ylabel('Confidence c')
    axes[0, 1].set_title('Confidence Evolution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. κ (trust-confidence gate) evolution
    axes[1, 0].plot(rounds, data['kappa_i'], label='κ_i', marker='o', markersize=3)
    axes[1, 0].plot(rounds, data['kappa_j'], label='κ_j', marker='s', markersize=3)
    axes[1, 0].set_xlabel('Round')
    axes[1, 0].set_ylabel('κ (Trust-Confidence Gate)')
    axes[1, 0].set_title('Trust-Confidence Gate κ = τ × c')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Payoffs
    axes[1, 1].plot(rounds, data['R_i'], label='R_i', marker='o', markersize=3, alpha=0.7)
    axes[1, 1].plot(rounds, data['R_j'], label='R_j', marker='s', markersize=3, alpha=0.7)
    axes[1, 1].set_xlabel('Round')
    axes[1, 1].set_ylabel('Payoff')
    axes[1, 1].set_title('Payoffs Over Time')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 5. Moral beliefs evolution (agent i's belief about j)
    B_hat_i_array = np.array(data['B_hat_i'])
    axes[2, 0].plot(rounds, B_hat_i_array[:, 0], label='Max-Sum', marker='o', markersize=3)
    axes[2, 0].plot(rounds, B_hat_i_array[:, 1], label='Equal-Split', marker='s', markersize=3)
    axes[2, 0].plot(rounds, B_hat_i_array[:, 2], label='Rawls', marker='^', markersize=3)
    axes[2, 0].axhline(agent_j.B[0], color='C0', linestyle='--', alpha=0.5, label='j\'s true Max-Sum')
    axes[2, 0].axhline(agent_j.B[1], color='C1', linestyle='--', alpha=0.5, label='j\'s true Equal-Split')
    axes[2, 0].axhline(agent_j.B[2], color='C2', linestyle='--', alpha=0.5, label='j\'s true Rawls')
    axes[2, 0].set_xlabel('Round')
    axes[2, 0].set_ylabel('Belief Weight')
    axes[2, 0].set_title('Agent i\'s Belief about j (B̂_i→j)')
    axes[2, 0].legend(fontsize=8, loc='best')
    axes[2, 0].grid(True, alpha=0.3)
    
    # 6. Fairness deviations
    axes[2, 1].plot(rounds, data['deviation_i'], label='d_i (deviation from i\'s perspective)', 
                   marker='o', markersize=3)
    axes[2, 1].plot(rounds, data['deviation_j'], label='d_j (deviation from j\'s perspective)', 
                   marker='s', markersize=3)
    axes[2, 1].set_xlabel('Round')
    axes[2, 1].set_ylabel('Fairness Deviation')
    axes[2, 1].set_title('Fairness Deviations Over Time')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def main():
    """Run TGMI demonstrations."""
    rng = np.random.default_rng(42)
    hyper = Hyperparameters()
    mgg = MoralGameGenerator(rng=rng, num_actions=21)
    
    print("="*70)
    print("TGMI Agent Demonstration")
    print("="*70)
    
    # ========================================================================
    # Scenario 1: Two TGMI agents with different moral priors
    # ========================================================================
    print("\n" + "="*70)
    print("Scenario 1: Two TGMI agents with different moral priors")
    print("="*70)
    
    # Utilitarian agent (values max-sum)
    B_i = np.array([0.7, 0.2, 0.1])
    agent_i = TGMI_Agent(agent_id=1, moral_prior=B_i, hyperparams=hyper, 
                        rng=np.random.default_rng(42))
    
    # Egalitarian agent (values equal-split)
    B_j = np.array([0.1, 0.7, 0.2])
    agent_j = TGMI_Agent(agent_id=2, moral_prior=B_j, hyperparams=hyper, 
                        rng=np.random.default_rng(43))
    
    print(f"\nAgent i (Utilitarian): B_i = {B_i}")
    print(f"Agent j (Egalitarian):  B_j = {B_j}")
    
    data_scenario1 = simulate_tgmi_pair(agent_i, agent_j, mgg, n_rounds=30)
    
    print(f"\nFinal trust: τ_i = {data_scenario1['tau_i'][-1]:.3f}, τ_j = {data_scenario1['tau_j'][-1]:.3f}")
    print(f"Final confidence: c_i = {data_scenario1['c_i'][-1]:.3f}, c_j = {data_scenario1['c_j'][-1]:.3f}")
    print(f"Final κ: κ_i = {data_scenario1['kappa_i'][-1]:.3f}, κ_j = {data_scenario1['kappa_j'][-1]:.3f}")
    
    # Plot
    fig1 = plot_tgmi_dynamics(
        data_scenario1, agent_i, agent_j,
        title="Scenario 1: Utilitarian vs Egalitarian TGMI Agents"
    )
    fig1.savefig("phase2_demo_scenario1.png", dpi=150, bbox_inches='tight')
    print("\nSaved: phase2_demo_scenario1.png")
    
    # ========================================================================
    # Scenario 2: Two similar TGMI agents (should build high trust)
    # ========================================================================
    print("\n" + "="*70)
    print("Scenario 2: Two TGMI agents with similar moral priors")
    print("="*70)
    
    B_i2 = np.array([0.5, 0.3, 0.2])
    B_j2 = np.array([0.45, 0.35, 0.2])
    
    agent_i2 = TGMI_Agent(agent_id=3, moral_prior=B_i2, hyperparams=hyper, 
                         rng=np.random.default_rng(44))
    agent_j2 = TGMI_Agent(agent_id=4, moral_prior=B_j2, hyperparams=hyper, 
                         rng=np.random.default_rng(45))
    
    print(f"\nAgent i: B_i = {B_i2}")
    print(f"Agent j: B_j = {B_j2}")
    
    data_scenario2 = simulate_tgmi_pair(agent_i2, agent_j2, mgg, n_rounds=30)
    
    print(f"\nFinal trust: τ_i = {data_scenario2['tau_i'][-1]:.3f}, τ_j = {data_scenario2['tau_j'][-1]:.3f}")
    print(f"Final confidence: c_i = {data_scenario2['c_i'][-1]:.3f}, c_j = {data_scenario2['c_j'][-1]:.3f}")
    
    fig2 = plot_tgmi_dynamics(
        data_scenario2, agent_i2, agent_j2,
        title="Scenario 2: Similar TGMI Agents (Should Build High Trust)"
    )
    fig2.savefig("phase2_demo_scenario2.png", dpi=150, bbox_inches='tight')
    print("Saved: phase2_demo_scenario2.png")
    
    # ========================================================================
    # Scenario 3: Random moral priors (diverse population sample)
    # ========================================================================
    print("\n" + "="*70)
    print("Scenario 3: Random moral priors from population")
    print("="*70)
    
    B_i3 = sample_moral_prior(rng)
    B_j3 = sample_moral_prior(rng)
    
    agent_i3 = TGMI_Agent(agent_id=5, moral_prior=B_i3, hyperparams=hyper, 
                         rng=np.random.default_rng(46))
    agent_j3 = TGMI_Agent(agent_id=6, moral_prior=B_j3, hyperparams=hyper, 
                         rng=np.random.default_rng(47))
    
    print(f"\nAgent i: B_i = [{B_i3[0]:.3f}, {B_i3[1]:.3f}, {B_i3[2]:.3f}]")
    print(f"Agent j: B_j = [{B_j3[0]:.3f}, {B_j3[1]:.3f}, {B_j3[2]:.3f}]")
    
    data_scenario3 = simulate_tgmi_pair(agent_i3, agent_j3, mgg, n_rounds=30)
    
    print(f"\nFinal trust: τ_i = {data_scenario3['tau_i'][-1]:.3f}, τ_j = {data_scenario3['tau_j'][-1]:.3f}")
    
    fig3 = plot_tgmi_dynamics(
        data_scenario3, agent_i3, agent_j3,
        title="Scenario 3: Random Moral Priors"
    )
    fig3.savefig("phase2_demo_scenario3.png", dpi=150, bbox_inches='tight')
    print("Saved: phase2_demo_scenario3.png")
    
    # ========================================================================
    # Summary Statistics
    # ========================================================================
    print("\n" + "="*70)
    print("Summary Statistics Across Scenarios")
    print("="*70)
    
    scenarios = [
        ("Different priors", data_scenario1),
        ("Similar priors", data_scenario2),
        ("Random priors", data_scenario3),
    ]
    
    for name, data in scenarios:
        avg_payoff_i = np.mean(data['R_i'])
        avg_payoff_j = np.mean(data['R_j'])
        avg_trust_i = np.mean(data['tau_i'])
        avg_trust_j = np.mean(data['tau_j'])
        avg_deviation_i = np.mean(data['deviation_i'])
        avg_deviation_j = np.mean(data['deviation_j'])
        
        print(f"\n{name}:")
        print(f"  Avg payoff: i={avg_payoff_i:.3f}, j={avg_payoff_j:.3f}")
        print(f"  Avg trust: i={avg_trust_i:.3f}, j={avg_trust_j:.3f}")
        print(f"  Avg deviation: i={avg_deviation_i:.3f}, j={avg_deviation_j:.3f}")
    
    print("\n" + "="*70)
    print("Phase 2 Demonstration Complete!")
    print("="*70)
    print("\nKey observations:")
    print("1. Trust evolves based on fairness compliance")
    print("2. Beliefs converge toward partners' true moral preferences")
    print("3. Confidence increases as beliefs become more certain")
    print("4. κ gates the influence of partner beliefs on moral utility")
    print("5. Similar moral priors lead to higher trust and cooperation")


if __name__ == "__main__":
    main()
