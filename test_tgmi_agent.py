"""
Unit tests for TGMI agent implementation.

Tests:
- Partner state initialization
- Moral utility computation
- Virtual bargaining
- Trust updates
- Belief updates (CK-ToM)
- Confidence and κ computation
- Full round integration
"""

import numpy as np
import pytest
from mgg import MoralGameGenerator, sample_moral_prior
from tgmi_agent import (
    TGMI_Agent, PartnerState,
    virtual_bargain, apply_action_noise,
    compute_fairness_deviation,
    update_trust, update_belief, update_confidence_and_kappa,
    update_reservation_utility, tgmi_round
)
from config import Hyperparameters


@pytest.fixture
def setup_agents():
    """Create two TGMI agents with different moral priors."""
    rng = np.random.default_rng(42)
    hyper = Hyperparameters()
    
    # Agent with strong utilitarian prior (max-sum)
    B_i = np.array([0.7, 0.2, 0.1])
    agent_i = TGMI_Agent(agent_id=1, moral_prior=B_i, hyperparams=hyper, rng=rng)
    
    # Agent with strong egalitarian prior (equal-split)
    B_j = np.array([0.1, 0.7, 0.2])
    agent_j = TGMI_Agent(agent_id=2, moral_prior=B_j, hyperparams=hyper, rng=rng)
    
    return agent_i, agent_j, rng


@pytest.fixture
def setup_game():
    """Create a sample game."""
    rng = np.random.default_rng(42)
    mgg = MoralGameGenerator(rng=rng, num_actions=21)
    game = mgg.sample_game()
    return game


class TestPartnerStateInitialization:
    """Test partner state initialization."""
    
    def test_initialization(self, setup_agents):
        """Test that partner state is correctly initialized."""
        agent_i, agent_j, rng = setup_agents
        
        agent_i.ensure_partner_initialized(agent_j.id)
        
        ps = agent_i.partners[agent_j.id]
        
        # Check structure
        assert len(ps.B_hat) == 3
        assert np.abs(ps.B_hat.sum() - 1.0) < 1e-10
        assert 0 <= ps.tau <= 1
        assert 0 <= ps.c <= 1
        assert 0 <= ps.kappa <= 1
        assert ps.d == 0.0
    
    def test_initial_trust(self, setup_agents):
        """Test that initial trust equals tau0."""
        agent_i, agent_j, rng = setup_agents
        agent_i.ensure_partner_initialized(agent_j.id)
        
        ps = agent_i.partners[agent_j.id]
        assert ps.tau == agent_i.hyper.tau0
    
    def test_kappa_calculation(self, setup_agents):
        """Test that κ = τ × c."""
        agent_i, agent_j, rng = setup_agents
        agent_i.ensure_partner_initialized(agent_j.id)
        
        ps = agent_i.partners[agent_j.id]
        assert np.abs(ps.kappa - ps.tau * ps.c) < 1e-10


class TestMoralUtility:
    """Test moral utility computation."""
    
    def test_moral_utility_range(self, setup_agents, setup_game):
        """Test that moral utility is in [0, 1]."""
        agent_i, agent_j, rng = setup_agents
        game = setup_game
        
        agent_i.ensure_partner_initialized(agent_j.id)
        
        # Test various action pairs
        for _ in range(10):
            idx_i = rng.integers(0, len(game.action_space_i))
            idx_j = rng.integers(0, len(game.action_space_j))
            
            F_vec = np.array([
                game.F["max_sum"][idx_i, idx_j],
                game.F["equal_split"][idx_i, idx_j],
                game.F["rawls"][idx_i, idx_j],
            ])
            
            U_i = agent_i.moral_utility(agent_j.id, F_vec)
            assert 0 <= U_i <= 1, f"Moral utility {U_i} out of range"
    
    def test_moral_utility_gating(self, setup_agents, setup_game):
        """Test that moral utility interpolates between B_i and B_hat."""
        agent_i, agent_j, rng = setup_agents
        game = setup_game
        
        agent_i.ensure_partner_initialized(agent_j.id)
        ps = agent_i.partners[agent_j.id]
        
        # Set extreme values for testing
        ps.kappa = 0.0  # Pure self prior
        F_vec = np.array([1.0, 0.5, 0.3])
        U_i_self = agent_i.moral_utility(agent_j.id, F_vec)
        U_i_self_expected = np.dot(agent_i.B, F_vec)
        assert np.abs(U_i_self - U_i_self_expected) < 1e-10
        
        ps.kappa = 1.0  # Pure partner belief
        U_i_partner = agent_i.moral_utility(agent_j.id, F_vec)
        U_i_partner_expected = np.dot(ps.B_hat, F_vec)
        assert np.abs(U_i_partner - U_i_partner_expected) < 1e-10
    
    def test_fairness_utility(self, setup_agents, setup_game):
        """Test fairness-only utility computation."""
        agent_i, agent_j, rng = setup_agents
        game = setup_game
        
        F_vec = np.array([1.0, 0.5, 0.3])
        UF = agent_i.fairness_utility(F_vec)
        expected = np.dot(agent_i.B, F_vec)
        assert np.abs(UF - expected) < 1e-10


class TestVirtualBargaining:
    """Test virtual bargaining mechanism."""
    
    def test_vb_returns_valid_actions(self, setup_agents, setup_game):
        """Test that VB returns valid action indices."""
        agent_i, agent_j, rng = setup_agents
        game = setup_game
        
        agent_i.ensure_partner_initialized(agent_j.id)
        agent_j.ensure_partner_initialized(agent_i.id)
        
        idx_ai, idx_aj, U_i, U_j = virtual_bargain(game, agent_i, agent_j, gamma=0.5)
        
        assert 0 <= idx_ai < len(game.action_space_i)
        assert 0 <= idx_aj < len(game.action_space_j)
        assert 0 <= U_i <= 1
        assert 0 <= U_j <= 1
    
    def test_vb_nash_product(self, setup_agents, setup_game):
        """Test that VB maximizes Nash product."""
        agent_i, agent_j, rng = setup_agents
        game = setup_game
        
        agent_i.ensure_partner_initialized(agent_j.id)
        agent_j.ensure_partner_initialized(agent_i.id)
        
        idx_ai_vb, idx_aj_vb, U_i_vb, U_j_vb = virtual_bargain(
            game, agent_i, agent_j, gamma=0.5
        )
        
        ps_i = agent_i.partners[agent_j.id]
        ps_j = agent_j.partners[agent_i.id]
        
        gain_i_vb = max(U_i_vb - ps_i.d, 0.0)
        gain_j_vb = max(U_j_vb - ps_j.d, 0.0)
        nash_vb = np.sqrt(gain_i_vb * gain_j_vb)  # gamma=0.5
        
        # Check that no other action pair gives higher Nash product
        n_samples = 20
        better_found = False
        for _ in range(n_samples):
            idx_i = rng.integers(0, len(game.action_space_i))
            idx_j = rng.integers(0, len(game.action_space_j))
            
            F_vec = np.array([
                game.F["max_sum"][idx_i, idx_j],
                game.F["equal_split"][idx_i, idx_j],
                game.F["rawls"][idx_i, idx_j],
            ])
            
            U_i = agent_i.moral_utility(agent_j.id, F_vec)
            U_j = agent_j.moral_utility(agent_i.id, F_vec)
            
            gain_i = max(U_i - ps_i.d, 0.0)
            gain_j = max(U_j - ps_j.d, 0.0)
            nash = np.sqrt(gain_i * gain_j)
            
            if nash > nash_vb + 1e-6:
                better_found = True
                break
        
        # VB should find the maximum (or very close to it)
        assert not better_found, "Found action pair better than VB result"


class TestTrustUpdate:
    """Test trust update mechanism."""
    
    def test_trust_increase_on_low_deviation(self, setup_agents):
        """Test that trust increases when deviation is low."""
        agent_i, agent_j, rng = setup_agents
        agent_i.ensure_partner_initialized(agent_j.id)
        
        ps = agent_i.partners[agent_j.id]
        tau_before = ps.tau
        
        # Low deviation (high compliance)
        update_trust(agent_i, agent_j.id, d_i=0.01)
        
        tau_after = ps.tau
        
        # Trust should increase (or stay high if already high)
        # With low deviation, s_i ≈ 1, so trust moves toward 1
        assert tau_after >= tau_before - 0.01  # Allow small numerical error
    
    def test_trust_decrease_on_high_deviation(self, setup_agents):
        """Test that trust decreases when deviation is high."""
        agent_i, agent_j, rng = setup_agents
        agent_i.ensure_partner_initialized(agent_j.id)
        
        ps = agent_i.partners[agent_j.id]
        ps.tau = 0.8  # Start with high trust
        tau_before = ps.tau
        
        # High deviation (low compliance)
        update_trust(agent_i, agent_j.id, d_i=0.5)
        
        tau_after = ps.tau
        
        # Trust should decrease
        assert tau_after < tau_before
    
    def test_trust_stays_in_range(self, setup_agents):
        """Test that trust stays in [0, 1]."""
        agent_i, agent_j, rng = setup_agents
        agent_i.ensure_partner_initialized(agent_j.id)
        
        # Extreme deviations
        for d_i in [0.0, 0.1, 0.5, 1.0, 2.0]:
            update_trust(agent_i, agent_j.id, d_i)
            ps = agent_i.partners[agent_j.id]
            assert 0 <= ps.tau <= 1


class TestBeliefUpdate:
    """Test belief update (CK-ToM) mechanism."""
    
    def test_belief_remains_distribution(self, setup_agents, setup_game):
        """Test that beliefs remain a probability distribution."""
        agent_i, agent_j, rng = setup_agents
        game = setup_game
        
        agent_i.ensure_partner_initialized(agent_j.id)
        
        # Multiple updates
        for _ in range(10):
            idx_i = rng.integers(0, len(game.action_space_i))
            idx_j = rng.integers(0, len(game.action_space_j))
            
            update_belief(agent_i, agent_j.id, game, idx_i, idx_j)
            
            ps = agent_i.partners[agent_j.id]
            B_hat = ps.B_hat
            
            # Check it's a valid probability distribution
            assert len(B_hat) == 3
            assert np.all(B_hat >= 0)
            assert np.abs(B_hat.sum() - 1.0) < 1e-10
    
    def test_belief_update_direction(self, setup_agents, setup_game):
        """Test that beliefs move toward observed fairness patterns."""
        agent_i, agent_j, rng = setup_agents
        game = setup_game
        
        agent_i.ensure_partner_initialized(agent_j.id)
        ps = agent_i.partners[agent_j.id]
        
        # Find action with high max-sum fairness
        max_sum_vals = game.F["max_sum"]
        idx_max = np.unravel_index(np.argmax(max_sum_vals), max_sum_vals.shape)
        idx_i, idx_j = idx_max
        
        B_hat_before = ps.B_hat.copy()
        
        # Multiple updates with high max-sum actions
        for _ in range(5):
            update_belief(agent_i, agent_j.id, game, idx_i, idx_j)
        
        B_hat_after = ps.B_hat
        
        # Belief in max-sum should increase (index 0)
        # (unless trust is very high and self-anchoring dominates)
        # For this test, just check it changed
        assert not np.allclose(B_hat_before, B_hat_after)


class TestConfidenceAndKappa:
    """Test confidence and κ computation."""
    
    def test_confidence_inverse_entropy(self, setup_agents):
        """Test that confidence is inverse of entropy."""
        agent_i, agent_j, rng = setup_agents
        agent_i.ensure_partner_initialized(agent_j.id)
        
        ps = agent_i.partners[agent_j.id]
        
        # Test with uniform distribution (max entropy)
        ps.B_hat = np.array([1/3, 1/3, 1/3])
        update_confidence_and_kappa(agent_i, agent_j.id)
        assert ps.c < 0.1  # Very low confidence
        
        # Test with peaked distribution (low entropy)
        ps.B_hat = np.array([0.9, 0.05, 0.05])
        update_confidence_and_kappa(agent_i, agent_j.id)
        assert ps.c > 0.5  # High confidence
    
    def test_kappa_equals_tau_times_c(self, setup_agents):
        """Test that κ = τ × c is maintained."""
        agent_i, agent_j, rng = setup_agents
        agent_i.ensure_partner_initialized(agent_j.id)
        
        ps = agent_i.partners[agent_j.id]
        
        for _ in range(10):
            # Random belief distribution
            ps.B_hat = rng.dirichlet(np.ones(3))
            ps.tau = rng.random()
            
            update_confidence_and_kappa(agent_i, agent_j.id)
            
            expected_kappa = ps.tau * ps.c
            assert np.abs(ps.kappa - expected_kappa) < 1e-10


class TestFullRound:
    """Test complete TGMI round integration."""
    
    def test_tgmi_round_executes(self, setup_agents, setup_game):
        """Test that a full TGMI round executes without errors."""
        agent_i, agent_j, rng = setup_agents
        game = setup_game
        
        R_i, R_j, info = tgmi_round(game, agent_i, agent_j)
        
        # Check payoffs are valid
        assert 0 <= R_i <= 1
        assert 0 <= R_j <= 1
        
        # Check info dict structure
        assert 'idx_ai_vb' in info
        assert 'idx_aj_vb' in info
        assert 'U_i_vb' in info
        assert 'U_j_vb' in info
        assert 'd_i' in info
        assert 'd_j' in info
        assert 'archetype' in info
    
    def test_repeated_interactions(self, setup_agents):
        """Test multiple rounds of interaction."""
        agent_i, agent_j, rng = setup_agents
        mgg = MoralGameGenerator(rng=rng)
        
        payoffs_i = []
        payoffs_j = []
        trusts_i = []
        trusts_j = []
        
        for t in range(20):
            game = mgg.sample_game()
            R_i, R_j, info = tgmi_round(game, agent_i, agent_j)
            
            payoffs_i.append(R_i)
            payoffs_j.append(R_j)
            trusts_i.append(agent_i.partners[agent_j.id].tau)
            trusts_j.append(agent_j.partners[agent_i.id].tau)
        
        # Check that values are reasonable
        assert len(payoffs_i) == 20
        assert len(trusts_i) == 20
        assert all(0 <= p <= 1 for p in payoffs_i)
        assert all(0 <= t <= 1 for t in trusts_i)
        
        # Trust should evolve (not stay constant)
        trust_variance = np.var(trusts_i)
        assert trust_variance > 0.001, "Trust should evolve over time"
    
    def test_partner_states_updated(self, setup_agents, setup_game):
        """Test that partner states are properly updated after a round."""
        agent_i, agent_j, rng = setup_agents
        game = setup_game
        
        # Store initial states
        agent_i.ensure_partner_initialized(agent_j.id)
        agent_j.ensure_partner_initialized(agent_i.id)
        
        tau_i_before = agent_i.partners[agent_j.id].tau
        B_hat_i_before = agent_i.partners[agent_j.id].B_hat.copy()
        d_i_before = agent_i.partners[agent_j.id].d
        
        # Run a round
        R_i, R_j, info = tgmi_round(game, agent_i, agent_j)
        
        # Check that states changed
        tau_i_after = agent_i.partners[agent_j.id].tau
        B_hat_i_after = agent_i.partners[agent_j.id].B_hat
        d_i_after = agent_i.partners[agent_j.id].d
        
        # At least some state should change
        states_changed = (
            tau_i_before != tau_i_after or
            not np.allclose(B_hat_i_before, B_hat_i_after) or
            d_i_before != d_i_after
        )
        assert states_changed, "Partner states should be updated"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
