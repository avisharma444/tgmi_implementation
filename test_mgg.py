"""
Unit tests for the Moral Game Generator (MGG).

Tests:
- Payoff and fairness value ranges
- Fairness function properties
- Game sampling
- Moral prior sampling
"""

import numpy as np
import pytest
from mgg import (
    MoralGameGenerator, Archetype, Game,
    sample_moral_prior, compute_fairness_functions,
    payoff_dilemma, payoff_assurance, payoff_bargain, payoff_public_goods
)
from config import DEFAULT_HYPERPARAMS


class TestPayoffArchetypes:
    """Test individual payoff archetype functions."""
    
    def test_dilemma_structure(self):
        """Test that dilemma has defection incentive."""
        params = {'cost': 0.5, 'benefit': 1.0}
        
        # Mutual cooperation
        R_i_cc, R_j_cc = payoff_dilemma(1.0, 1.0, params)
        
        # Defection against cooperator (should be higher)
        R_i_dc, R_j_cd = payoff_dilemma(0.0, 1.0, params)
        
        # Defection is better than cooperation (given partner cooperates)
        assert R_i_dc > R_i_cc, "Defection should dominate in dilemma"
    
    def test_assurance_coordination(self):
        """Test that assurance rewards coordination."""
        params = {'threshold': 0.5, 'high': 1.0, 'low': 0.3}
        
        # Both coordinate high
        R_i_high, R_j_high = payoff_assurance(0.8, 0.8, params)
        
        # Both coordinate low
        R_i_low, R_j_low = payoff_assurance(0.2, 0.2, params)
        
        # Mismatch
        R_i_mix, R_j_mix = payoff_assurance(0.8, 0.2, params)
        
        # High coordination should be best
        assert R_i_high > R_i_low
        assert R_i_high > R_i_mix
    
    def test_bargain_compatibility(self):
        """Test bargain pays out when demands compatible."""
        params = {'pie_size': 1.0}
        
        # Compatible demands
        R_i_ok, R_j_ok = payoff_bargain(0.4, 0.5, params)
        assert R_i_ok == 0.4
        assert R_j_ok == 0.5
        
        # Incompatible demands
        R_i_bad, R_j_bad = payoff_bargain(0.7, 0.8, params)
        assert R_i_bad == 0.0
        assert R_j_bad == 0.0
    
    def test_public_goods_freeriding(self):
        """Test public goods has freeriding temptation."""
        # For freeriding to be tempting, multiplier < n_players
        # With n=2, use multiplier < 2
        params = {'multiplier': 1.8, 'endowment': 1.0}
        
        # Full contribution by both
        R_i_full, R_j_full = payoff_public_goods(1.0, 1.0, params)
        
        # Freeride (contribute 0) while partner contributes
        R_i_free, R_j_coop = payoff_public_goods(0.0, 1.0, params)
        
        # Freerider gets higher payoff (keeps endowment + half of multiplied partner contribution)
        assert R_i_free > R_i_full, "Freeriding should be tempting"


class TestFairnessFunctions:
    """Test fairness function computations."""
    
    def test_fairness_ranges(self):
        """Test that all fairness values are in [0, 1]."""
        rng = np.random.default_rng(42)
        mgg = MoralGameGenerator(rng=rng)
        
        for _ in range(100):
            game = mgg.sample_game()
            
            # Check payoff ranges
            assert game.R_i.min() >= -0.001, "R_i should be >= 0"
            assert game.R_i.max() <= 1.001, "R_i should be <= 1"
            assert game.R_j.min() >= -0.001, "R_j should be >= 0"
            assert game.R_j.max() <= 1.001, "R_j should be <= 1"
            
            # Check fairness ranges
            for phi in ["max_sum", "equal_split", "rawls"]:
                F_phi = game.F[phi]
                assert F_phi.min() >= -0.001, f"F_{phi} should be >= 0"
                assert F_phi.max() <= 1.001, f"F_{phi} should be <= 1"
    
    def test_equal_split_symmetry(self):
        """Test that equal_split is highest when payoffs are equal."""
        R_i = np.array([[0.5, 0.3], [0.7, 0.8]])
        R_j = np.array([[0.5, 0.7], [0.3, 0.8]])
        R_max = 0.8
        
        F = compute_fairness_functions(R_i, R_j, R_max)
        F_eq = F["equal_split"]
        
        # (0,0) and (1,1) have equal payoffs
        assert F_eq[0, 0] == 1.0, "Equal payoffs should give F_eq = 1"
        assert F_eq[1, 1] == 1.0, "Equal payoffs should give F_eq = 1"
        
        # (0,1) and (1,0) have unequal payoffs
        assert F_eq[0, 1] < 1.0
        assert F_eq[1, 0] < 1.0
    
    def test_max_sum_monotonicity(self):
        """Test that max_sum increases with total payoff."""
        R_i = np.array([[0.2, 0.5], [0.3, 0.9]])
        R_j = np.array([[0.1, 0.3], [0.4, 0.9]])
        R_max = 0.9
        
        F = compute_fairness_functions(R_i, R_j, R_max)
        F_sum = F["max_sum"]
        
        # (1,1) has highest sum
        assert F_sum[1, 1] >= F_sum[0, 0]
        assert F_sum[1, 1] >= F_sum[0, 1]
        assert F_sum[1, 1] >= F_sum[1, 0]
    
    def test_rawls_minimum(self):
        """Test that Rawls equals normalized minimum."""
        R_i = np.array([[0.8, 0.2], [0.5, 0.6]])
        R_j = np.array([[0.2, 0.8], [0.5, 0.7]])
        R_max = 0.8
        
        F = compute_fairness_functions(R_i, R_j, R_max)
        F_rawls = F["rawls"]
        
        expected = np.minimum(R_i, R_j) / R_max
        np.testing.assert_array_almost_equal(F_rawls, expected)


class TestGameSampling:
    """Test game sampling from MGG."""
    
    def test_game_structure(self):
        """Test that sampled games have correct structure."""
        rng = np.random.default_rng(42)
        mgg = MoralGameGenerator(rng=rng)
        game = mgg.sample_game()
        
        # Check action spaces
        assert len(game.action_space_i) == 21  # default
        assert len(game.action_space_j) == 21
        assert game.action_space_i[0] == 0.0
        assert game.action_space_i[-1] == 1.0
        
        # Check payoff matrices shape
        assert game.R_i.shape == (21, 21)
        assert game.R_j.shape == (21, 21)
        
        # Check fairness functions present
        assert "max_sum" in game.F
        assert "equal_split" in game.F
        assert "rawls" in game.F
        
        # Check archetype
        assert isinstance(game.archetype, Archetype)
    
    def test_archetype_distribution(self):
        """Test that all archetypes can be sampled."""
        rng = np.random.default_rng(42)
        mgg = MoralGameGenerator(rng=rng)
        
        archetypes_seen = set()
        for _ in range(100):
            game = mgg.sample_game()
            archetypes_seen.add(game.archetype)
        
        # Should see all 4 archetypes in 100 samples
        assert len(archetypes_seen) == 4, "Should sample all archetype types"
    
    def test_custom_action_grid(self):
        """Test MGG with custom action grid."""
        rng = np.random.default_rng(42)
        custom_grid = np.linspace(0, 1, 11)
        mgg = MoralGameGenerator(action_grid=custom_grid, rng=rng)
        
        game = mgg.sample_game()
        assert len(game.action_space_i) == 11
        assert game.R_i.shape == (11, 11)


class TestMoralPriors:
    """Test moral prior sampling."""
    
    def test_prior_is_probability_distribution(self):
        """Test that sampled priors sum to 1."""
        rng = np.random.default_rng(42)
        
        for _ in range(100):
            prior = sample_moral_prior(rng)
            assert len(prior) == 3
            assert np.abs(prior.sum() - 1.0) < 1e-10
            assert np.all(prior >= 0)
            assert np.all(prior <= 1)
    
    def test_prior_diversity(self):
        """Test that priors show diversity (not all the same)."""
        rng = np.random.default_rng(42)
        priors = [sample_moral_prior(rng) for _ in range(100)]
        
        # Compute variance in first weight across samples
        first_weights = [p[0] for p in priors]
        variance = np.var(first_weights)
        
        # Should have substantial variance (not all identical)
        assert variance > 0.01, "Priors should be diverse"


def test_integration_full_pipeline():
    """Integration test: sample games and moral priors together."""
    rng = np.random.default_rng(42)
    mgg = MoralGameGenerator(rng=rng)
    
    # Create two agents with moral priors
    B_i = sample_moral_prior(rng)
    B_j = sample_moral_prior(rng)
    
    # Sample a game
    game = mgg.sample_game()
    
    # Compute moral utilities for a random action pair
    idx_ai = rng.integers(0, len(game.action_space_i))
    idx_aj = rng.integers(0, len(game.action_space_j))
    
    F_vec = np.array([
        game.F["max_sum"][idx_ai, idx_aj],
        game.F["equal_split"][idx_ai, idx_aj],
        game.F["rawls"][idx_ai, idx_aj],
    ])
    
    U_i = np.dot(B_i, F_vec)
    U_j = np.dot(B_j, F_vec)
    
    # Utilities should be in valid range
    assert 0 <= U_i <= 1
    assert 0 <= U_j <= 1


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
