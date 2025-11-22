# Phase 2 Summary: TGMI Agent Implementation

## Overview

Phase 2 implements the **Trust-Gated Moral Inference (TGMI)** agent, which is the core contribution of the paper. TGMI agents learn about their partners' moral values through interaction and use trust to gate the influence of these learned beliefs on their decision-making.

## What TGMI Does

TGMI agents:

1. **Maintain beliefs** about partners' moral norms (which fairness principles they value)
2. **Track trust** based on partners' fairness compliance over time
3. **Compute confidence** in their beliefs (inverse of uncertainty)
4. **Make decisions** using trust-confidence-gated moral utilities
5. **Update** trust, beliefs, and confidence after each interaction

The key insight: **Trust gates moral inference**. High trust → assume partner is like me (self-anchoring). Low trust → learn from evidence.

## Core Mechanisms

### 1. Partner State Management

Each agent i maintains a `PartnerState` for each partner j:

```python
@dataclass
class PartnerState:
    B_hat: np.ndarray  # Belief about j's moral norms [3,]
    tau: float         # Trust τ_i ∈ [0, 1]
    c: float           # Confidence c_i ∈ [0, 1]
    kappa: float       # κ_i = τ_i × c_i (trust-confidence gate)
    d: float           # Reservation utility (for bargaining)
```

**Initialization** (first meeting):
- B_hat ~ Dirichlet(1, 1, 1) (uniform prior)
- τ = τ₀ = 0.5 (neutral initial trust)
- c computed from entropy of B_hat
- κ = τ × c
- d = 0 (no baseline yet)

### 2. Moral Utility (Trust-Confidence-Gated)

From **Equation 2**:

```
U_i(a) = Σ_φ [(1-κ_i)B_i(φ) + κ_i B̂_i→j(φ)] F_φ(a)
```

**Interpretation:**
- When κ = 0: Pure self prior (ignore partner belief)
- When κ = 1: Pure partner belief (fully trust learned model)
- κ = τ × c: Trust and confidence jointly gate influence

**Why this works:**
- High trust + high confidence → partner belief dominates
- High trust + low confidence → self-anchoring (no strong belief to use)
- Low trust (any confidence) → stick to self values

### 3. Virtual Bargaining (Joint Action Selection)

From **Equation 3** and **Algorithm 1**:

```
(a_i^VB, a_j^VB) = argmax_{a_i,a_j} (U_i(a) - d_i)₊^γ (U_j(a) - d_j)₊^(1-γ)
```

**Nash bargaining product** with:
- γ = bargaining power (0.5 = symmetric)
- (x)₊ = max(x, 0) (can't go below reservation utility)
- d_i, d_j = reservation utilities (updated from previous round)

**Implementation:**
- Exhaustive search over discrete action space (21 × 21 = 441 profiles)
- Both agents use their moral utilities (not raw payoffs)
- Finds joint action that maximizes product of gains

**Key property:** Virtual bargaining is **cooperative** - agents coordinate on mutually beneficial outcomes according to their moral utilities.

### 4. Trust Update (Leaky Integrator)

From the **Methods section**:

```
s_i^(t) = exp(-λ_dev × d_i^(t))     # Compliance signal
τ_i^(t+1) = (1-η)τ_i^(t) + η s_i^(t)   # Leaky integrator
```

Where:
- d_i = fairness deviation (shortfall from best possible fairness)
- λ_dev = 5.0 (deviation sensitivity)
- η = 0.1 (learning rate)

**Fairness deviation** d_i:
```
d_i = max_{a_j'} U^F_i(a_i^VB, a_j') - U^F_i(a_i^VB, a_j^VB)
```

**Interpretation:**
- d_i = 0: Partner's action was optimal for my fairness → trust increases
- d_i > 0: Partner could have been fairer → trust decreases
- Measures **fairness compliance**, not payoff maximization

**Why leaky integrator?**
- Gradual adaptation (not instant)
- Forgetting factor (old violations fade)
- Stable convergence

### 5. Belief Update (CK-ToM)

From **Equation 4**:

```
B̂_i→j^(t+1)(φ) ∝ B̂_i→j^(t)(φ) × 
                  [exp(β F_φ(a^t))]^(1-ατ_i^t) × 
                  [B_i(φ)]^(ατ_i^t)
```

**Components:**
1. **Prior:** Previous belief B̂^(t)
2. **Evidence:** Fairness observed exp(β F_φ) weighted by (1-ατ)
3. **Self-anchoring:** Own prior B_i weighted by ατ

**Trust modulation:**
- **Low trust (τ → 0):** Weight evidence heavily (1-ατ → 1), ignore self (ατ → 0)
  - "I don't know them, so learn from observations"
- **High trust (τ → 1):** Weight self prior heavily (ατ → α), discount evidence
  - "I trust them, so assume they're like me"

**Parameters:**
- α = 0.5 (self-anchoring strength)
- β = 3.0 (evidence strength)

**This is CK-ToM** (Collaborative Knowledge Theory of Mind): High trust enables projection of self onto partner.

### 6. Confidence and κ Update

```
c_i = 1 - H(B̂_i→j) / log|F|     # Confidence = 1 - normalized entropy
κ_i = τ_i × c_i                  # Trust-confidence gate
```

**Shannon entropy:**
```
H(B̂) = -Σ_φ B̂(φ) log B̂(φ)
```

**Interpretation:**
- High entropy (uniform belief) → low confidence (c → 0)
- Low entropy (peaked belief) → high confidence (c → 1)
- κ = 0 when either trust or confidence is low

### 7. Reservation Utility Update

```
d_i^(t+1) = U^F_i(a^VB)
```

Sets baseline for next round's bargaining. Agents won't accept outcomes worse than what they just got (in fairness terms).

## Algorithm Flow (One Round)

From **Algorithm 1**:

```
1. Virtual Bargaining:
   Select (a_i^VB, a_j^VB) via Nash product
   
2. Action Noise (optional):
   With probability ε_a, randomize each action
   
3. Observe Outcomes:
   Payoffs R_i, R_j
   Fairness values F_φ(a^VB)
   
4. Compute Deviations:
   d_i = best fairness i could have gotten - actual fairness
   
5. Update Trust:
   τ_i ← (1-η)τ_i + η exp(-λ_dev d_i)
   
6. Update Beliefs (CK-ToM):
   B̂_i→j ∝ B̂_i→j × [exp(β F)]^(1-ατ) × [B_i]^(ατ)
   
7. Update Confidence & κ:
   c_i ← 1 - H(B̂_i→j) / log|F|
   κ_i ← τ_i × c_i
   
8. Update Reservation Utility:
   d_i ← U^F_i(a^VB)
```

## Implementation Architecture

### Files Created

```
tgmi_agent.py         # Core TGMI implementation (~650 lines)
test_tgmi_agent.py    # Unit tests (18 tests, all passing)
demo_tgmi.py          # Demonstration scripts
phase2_demo_*.png     # Generated visualizations (3 scenarios)
```

### Key Classes and Functions

**Classes:**
- `PartnerState`: State maintained about each partner
- `TGMI_Agent`: Main agent class

**Core Functions:**
- `moral_utility()`: Compute U_i(a) with trust-confidence gating
- `fairness_utility()`: Compute U^F_i(a) (self-prior only)
- `virtual_bargain()`: Nash product joint action selection
- `compute_fairness_deviation()`: Calculate d_i
- `update_trust()`: Leaky integrator trust update
- `update_belief()`: CK-ToM belief update
- `update_confidence_and_kappa()`: Confidence and gate update
- `tgmi_round()`: Execute one complete interaction

## Testing and Validation

### Unit Tests (18 tests, all passing)

✓ **Partner State Initialization:**
- Correct structure and probability distributions
- Initial trust = τ₀
- κ = τ × c relationship

✓ **Moral Utility:**
- Values in [0, 1]
- Correct interpolation between B_i and B̂
- κ = 0 → pure self prior
- κ = 1 → pure partner belief

✓ **Virtual Bargaining:**
- Returns valid actions
- Maximizes Nash product (verified via sampling)

✓ **Trust Update:**
- Increases with low deviation
- Decreases with high deviation
- Stays in [0, 1]

✓ **Belief Update (CK-ToM):**
- Maintains probability distribution
- Beliefs evolve toward observed patterns
- Trust modulates evidence vs. self-anchoring

✓ **Confidence and κ:**
- Confidence = 1 - normalized entropy
- κ = τ × c maintained

✓ **Full Round Integration:**
- Complete rounds execute successfully
- Partner states properly updated
- Trust evolves over repeated interactions

### Demonstration Scenarios

**Scenario 1: Different Moral Priors**
- Agent i: Utilitarian (70% max-sum, 20% equal-split, 10% Rawls)
- Agent j: Egalitarian (10% max-sum, 70% equal-split, 20% Rawls)
- **Result:** Trust builds to ~96%, beliefs converge, high cooperation

**Scenario 2: Similar Moral Priors**
- Both agents have similar distributions
- **Result:** Very high trust (~98%), perfect belief convergence

**Scenario 3: Random Priors**
- Sampled from Dirichlet(1, 1, 1)
- **Result:** Trust dynamics depend on moral compatibility

### Key Observations from Demonstrations

1. **Trust Evolution:**
   - Starts at τ₀ = 0.5
   - Increases rapidly when partners are fair
   - Low deviations → sustained high trust (>0.95)

2. **Belief Convergence:**
   - Beliefs move toward partners' true moral priors
   - Convergence faster with low initial trust (more evidence weighting)
   - Self-anchoring kicks in as trust increases

3. **Confidence Increases:**
   - Entropy decreases over time (beliefs become peaked)
   - Confidence c → 1 in most cases
   - κ tracks trust when confidence is high

4. **Cooperation Emerges:**
   - Virtual bargaining finds mutually beneficial outcomes
   - Average payoffs ~0.65 (vs. maximum possible 1.0)
   - Fairness deviations near zero (d_i ≈ 0)

5. **Moral Compatibility Matters:**
   - Similar priors → faster trust building
   - Different priors → still cooperate but with adaptation

## Connection to Paper Results

Our implementation replicates the key mechanisms from the paper:

- **Trust-confidence gating (κ):** Implemented as τ × c
- **Virtual bargaining:** Nash product with moral utilities
- **CK-ToM:** Self-anchoring increases with trust
- **Fairness compliance:** Trust based on deviations, not payoffs
- **Belief convergence:** Agents learn partners' moral norms

The demonstrations show qualitatively similar dynamics to **Figure 3** in the paper:
- Trust increases with cooperative partners
- Beliefs converge to true values
- Confidence increases over time

## Design Decisions

1. **Discrete Action Spaces:** Used exhaustive search (21 × 21) for virtual bargaining
   - Trade-off: Computational cost vs. optimality
   - Alternative: Gradient-based optimization for continuous spaces

2. **Symmetric Bargaining Power:** γ = 0.5 (equal power)
   - Paper mentions asymmetric cases but uses symmetric as default

3. **Action Noise:** Implemented but set ε_a = 0 by default
   - Can be enabled for robustness testing

4. **Numerical Stability:**
   - Added small epsilon (1e-12) in log computations
   - Clamped trust to [0, 1]
   - Normalized beliefs after each update

5. **Modularity:**
   - Each mechanism (trust, belief, confidence) in separate function
   - Easy to test and modify individual components
   - Clear mapping to paper's equations

## Remaining Limitations

1. **No payoff information:** TGMI agents currently don't use or track payoffs (only fairness)
   - Paper combines payoffs and fairness in evolutionary fitness

2. **Two-player only:** Implementation assumes pairwise interactions
   - Paper mentions extension to n-player games

3. **Fixed hyperparameters:** η, λ, α, β are constants
   - Could explore adaptive or learned parameters

4. **No baseline agents yet:** Only TGMI vs. TGMI tested
   - Phase 4 will add selfish, altruistic, TFT, etc.

## Next Steps (Phase 3 & 4)

**Phase 3 - Logging:**
- Comprehensive logging of all state variables
- Time series analysis tools
- Visualization of belief trajectories
- Statistical analysis of trust dynamics

**Phase 4 - Evolutionary Dynamics:**
- Baseline agent implementations
- Population-level simulation
- Moran process with selection
- Replication of paper's main results (Figure 4, 5)

---

**Status**: ✅ Phase 2 Complete - All tests passing, demonstrations successful

The TGMI agent is fully functional and exhibits the expected behaviors:
- Trust evolves based on fairness compliance ✓
- Beliefs converge through CK-ToM ✓
- Confidence increases with learning ✓
- Virtual bargaining enables cooperation ✓
- All mechanisms validated through unit tests ✓
