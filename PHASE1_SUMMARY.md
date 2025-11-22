# Phase 1 Summary: Moral Game Generator (MGG)

## Overview

Phase 1 implements the **Moral Game Generator (MGG)**, which creates the strategic environments where TGMI agents interact. The MGG generates games with varying payoff structures and fairness landscapes, following the framework described in the paper.

## What the MGG Does

The MGG is responsible for:

1. **Generating diverse strategic games** from four payoff archetypes
2. **Computing fairness functions** that evaluate joint actions
3. **Sampling moral priors** for agents (beliefs about fairness principles)
4. **Normalizing all values** to [0, 1] for consistent comparisons

## Implementation Details

### Core Components

#### 1. Payoff Archetypes (`Archetype` enum)

Four game types representing fundamental strategic structures:

- **DILEMMA**: Prisoner's Dilemma-like games where defection dominates but mutual cooperation is better
  - Players choose cooperation level in [0, 1]
  - Payoff: benefit × partner's action - cost × own action
  - Captures tension between individual and collective benefit

- **ASSURANCE**: Coordination/Stag Hunt games rewarding mutual cooperation
  - High payoffs when both coordinate on high actions
  - Safe but lower payoff at low actions
  - Multiple equilibria with risk-dominant and payoff-dominant outcomes

- **BARGAIN**: Resource division games
  - Actions represent demands over a shared resource
  - Compatible demands → both get their share
  - Incompatible demands → disagreement, both get nothing

- **PUBLIC_GOODS**: Public goods contribution games
  - Players contribute to a public good that gets multiplied
  - Each receives equal share plus keeps what they didn't contribute
  - Freeriding temptation when multiplier < number of players

#### 2. Fairness Functions (`compute_fairness_functions`)

Three fairness principles that evaluate joint actions:

- **F_Max-Sum** = (R_i + R_j) / (2 × R_max)
  - Utilitarian principle: maximize total welfare
  - Highest when joint payoff is maximized

- **F_Equal-Split** = 1 - |R_i - R_j| / R_max
  - Egalitarian principle: minimize inequality
  - Highest (= 1) when payoffs are exactly equal
  - Decreases linearly with payoff difference

- **F_Rawls** = min(R_i, R_j) / R_max
  - Maximin principle: maximize the minimum payoff
  - Focuses on the worst-off player
  - Named after philosopher John Rawls

All fairness values normalized to [0, 1].

#### 3. Game Dataclass

```python
@dataclass
class Game:
    archetype: Archetype              # Which game type
    action_space_i: np.ndarray        # Discrete actions for player i
    action_space_j: np.ndarray        # Discrete actions for player j
    R_i: np.ndarray                   # Payoff matrix for i (shape: n × n)
    R_j: np.ndarray                   # Payoff matrix for j (shape: n × n)
    F: Dict[str, np.ndarray]          # Fairness functions (each n × n)
    R_max: float                      # Maximum payoff for normalization
```

#### 4. MoralGameGenerator Class

Main interface for game sampling:

```python
mgg = MoralGameGenerator(
    archetype_probs=np.ones(4)/4,    # Uniform over archetypes
    action_grid=np.linspace(0, 1, 21), # 21 discrete actions
    rng=np.random.default_rng(42)    # Random number generator
)

game = mgg.sample_game()  # Returns a Game object
```

**Sampling process:**
1. Sample archetype from distribution
2. Sample payoff function parameters for that archetype
3. Compute payoff matrices over discrete action grid
4. Normalize payoffs to [0, 1]
5. Compute three fairness functions
6. Return Game object

#### 5. Moral Prior Sampling

Agents' intrinsic beliefs about fairness:

```python
B_i = sample_moral_prior(rng)  # Returns [w_max_sum, w_equal_split, w_rawls]
```

- Sampled from Dirichlet(1, 1, 1) distribution
- Represents a probability distribution over the three fairness principles
- Captures moral diversity in the population

## How It Works

### Game Generation Pipeline

```
1. Sample Archetype
   ↓
2. Sample Parameters (e.g., cost, benefit, threshold)
   ↓
3. Define Payoff Function: (ai, aj) → (R_i, R_j)
   ↓
4. Evaluate on Discrete Grid (21 × 21 by default)
   ↓
5. Normalize Payoffs to [0, 1]
   ↓
6. Compute Fairness Functions (3 matrices)
   ↓
7. Return Game Object
```

### Fairness Function Design

The fairness functions are **not** arbitrary - they represent well-studied ethical principles:

- **Max-Sum**: Total welfare matters, regardless of distribution
- **Equal-Split**: Fairness = equality
- **Rawls**: Protect the most vulnerable

These create different "moral landscapes" over the action space. Different agents (with different moral priors B_i) will perceive the same game differently.

### Discretization

Continuous action spaces [0, 1] are discretized to:
- Enable exhaustive search in virtual bargaining
- Reduce computational complexity
- Default: 21 actions → 441 joint action profiles per game

## Testing and Validation

### Unit Tests (test_mgg.py)

✓ **14 tests, all passing**

Test coverage:
- Payoff archetype properties (dilemma has defection incentive, etc.)
- Fairness function ranges [0, 1]
- Fairness function properties (equal-split = 1 when R_i = R_j, etc.)
- Game sampling structure and diversity
- Moral prior properties (probability distribution, diversity)

### Visualizations (visualize_mgg.py)

Generated heatmaps for each archetype showing:
- Payoff matrices R_i and R_j
- Total payoff R_i + R_j
- Three fairness functions

**Key observations from visualizations:**
- Dilemma: Low actions (defection) dominate along the diagonal
- Assurance: High fairness in top-right corner (mutual cooperation)
- Bargain: Sharp boundary between compatible/incompatible demands
- Public Goods: Tension between contribution and free-riding

### Statistics from 100 Random Games

- Payoffs: Always in [0, 1] ✓
- F_equal_split: Max always = 1 (perfect equality always possible) ✓
- F_max_sum: Mean max = 0.767 (not all games can achieve max efficiency)
- F_rawls: Min always = 0 (some action pairs give 0 to someone) ✓

## Files Created

```
config.py              # Hyperparameters and configuration
mgg.py                 # Core MGG implementation
test_mgg.py           # Unit tests (14 tests)
visualize_mgg.py      # Visualization script
phase1_viz_*.png      # Generated visualizations (4 files)
```

## Key Design Decisions

1. **Discrete vs Continuous Actions**: Used discretization for tractability while maintaining expressiveness

2. **Fairness Normalization**: Divided Max-Sum by 2×R_max (not R_max) to keep it in [0, 1] since both players contribute

3. **Public Goods Parameters**: Set multiplier < 2 to ensure freeriding is tempting (standard in public goods literature)

4. **Symmetric Action Spaces**: Both players use the same action grid (could be generalized later)

5. **Parameter Sampling**: Each archetype has randomized parameters (cost, benefit, threshold, etc.) to create diversity within archetypes

## Connection to TGMI

The MGG creates the **fairness environment** that TGMI agents must navigate:

- **Moral priors B_i** define each agent's intrinsic values
- **Fairness functions F_φ** define the moral landscape of each game
- **TGMI** (Phase 2) will use these to compute moral utilities and make decisions

The MGG is **completely independent** of TGMI - it only knows about payoffs and fairness, not about trust, beliefs, or virtual bargaining. This modularity makes the implementation clean and testable.

## Next Steps (Phase 2)

With the MGG complete, we can now implement:
1. TGMI agent class with partner state management
2. Moral utility computation using B_i and F_φ
3. Virtual bargaining to select joint actions
4. Trust and belief update mechanisms
5. Testing on 2-agent interactions over multiple rounds

---

**Status**: ✅ Phase 1 Complete - All tests passing, visualizations generated
