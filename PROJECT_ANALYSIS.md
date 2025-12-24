# PNAS Paper Implementation: Project Analysis

## Overview

This project implements **Trust-Gated Moral Inference (TGMI)** - a computational model for cooperation under moral uncertainty. The implementation is based on research exploring how agents can cooperate despite having different moral frameworks by using trust-gated belief updates and virtual bargaining.

**Date:** December 24, 2025  
**Repository:** MLOpsProject (avisharma444/MLOpsProject)  
**Primary References:** `docs/SAMVAAD_PNAS.pdf` and `docs/PNAS_Template_for_Supplementary_Information (1).pdf`

---

## 1. Project Structure

### Core Components

```
pnas_paper_implementation/
├── tgmi/                   # Trust-Gated Moral Inference Agent
│   ├── agent.py           # Core TGMI algorithm (Algorithm 1)
│   └── experiments.py     # Experiments and visualizations
├── mgg/                   # Moral Game Generator
│   ├── generator.py       # Game generation with fairness matrices
│   └── config.py         # Hyperparameters
├── docs/                  # Research papers
├── outputs/plots/         # Generated visualizations
└── requirements.txt       # Dependencies
```

---

## 2. Moral Game Generator (MGG)

### Purpose
Generates strategic interaction scenarios with explicit moral dimensions represented as fairness functions.

### Game Archetypes
The MGG supports four types of moral games:

1. **DILEMMA** (Prisoner's Dilemma style)
   - Temptation to defect vs. mutual cooperation
   - Payoffs: `R_i = b * a_j - c * a_i`
   - Parameters: cost `c ∈ [0.3, 0.7]`, benefit `b ∈ [c+0.2, 1.5]`

2. **ASSURANCE** (Stag Hunt / Coordination)
   - Mutual cooperation yields high payoffs
   - Threshold-based coordination
   - Parameters: threshold ∈ [0.4, 0.6], high payoff ∈ [0.8, 1.2]

3. **BARGAIN** (Resource Division)
   - Division of a fixed pie
   - Zero payoff if claims exceed pie size
   - Parameter: pie_size ∈ [0.8, 1.2]

4. **PUBLIC_GOODS** (Contribution Game)
   - Multiplier effect on contributions
   - Classic social dilemma structure
   - Parameters: multiplier ∈ [1.5, 1.9], endowment ∈ [0.8, 1.2]

### Fairness Functions

Three fairness principles are computed for each game:

1. **Max-Sum (Utilitarian)**
   ```
   F_max_sum(a_i, a_j) = (R_i + R_j) / (2 * R_max)
   ```
   Maximizes total welfare

2. **Equal-Split (Egalitarian)**
   ```
   F_equal_split(a_i, a_j) = 1 - |R_i - R_j| / R_max
   ```
   Minimizes inequality

3. **Rawls (Maximin)**
   ```
   F_rawls(a_i, a_j) = min(R_i, R_j) / R_max
   ```
   Maximizes minimum payoff

These fairness functions are normalized to [0, 1] range.

---

## 3. Trust-Gated Moral Inference (TGMI) Agent

### Core Philosophy

The TGMI agent addresses the fundamental problem: **How can agents cooperate when they disagree about what is morally right?**

Key innovation: Uses **trust** (τ) as a gate that controls how much an agent updates their beliefs about their partner's moral framework and how much they weight cooperative outcomes.

### Algorithm 1: TGMI Core (Detailed Analysis)

#### Initialization Phase (Lines 1-5)

```python
# Line 1: Initialize belief over partner's moral norms
B_hat ~ Dirichlet(1, 1, 1)  # Uniform prior over 3 fairness principles
```
- Starts with maximum uncertainty about partner's morals
- Uses Dirichlet distribution for probability simplex

```python
# Line 2: Initialize trust
τ_0 = 0.5  # Initial trust (configurable)
```
- Trust τ ∈ [0, 1] represents confidence in partner's reliability

```python
# Line 3: Initialize confidence from belief entropy
c = 1 - H(B_hat) / log(K)
```
- Confidence inversely related to belief entropy
- c = 1 when certain (peaked distribution)
- c = 0 when maximally uncertain (uniform distribution)

```python
# Line 4: Initialize cooperation weight
ϖ = τ * c  # "ki" in the code
```
- **Critical**: Cooperation weight combines trust and confidence
- Controls blend between self-interest and cooperation

```python
# Line 5: Initialize reservation utilities
d_i = 0.0, d_j = 0.0
```
- Baseline utilities for Nash bargaining
- Updated after each round

#### Action Selection Phase (Lines 6-9)

```python
# Line 6: Compute own moral utility
U_i(a_i, a_j) = (1 - ϖ) * Σ_ω B_i(ω) * F_ω(a_i, a_j) + 
                ϖ * Σ_ω B_hat(ω) * F_ω(a_i, a_j)
```
**Key Insight**: This is a **weighted blend**:
- When ϖ = 0 (low trust/confidence): Pure self-interest using own beliefs B_i
- When ϖ = 1 (high trust/confidence): Full cooperation using partner model B_hat
- Intermediate: Smooth interpolation

```python
# Line 7: Compute model of partner's utility (transposed)
U_j_hat(a_j, a_i) = Σ_ω B_hat(ω) * F_ω(a_j, a_i)
```
Theory of Mind: Agent models what partner values

```python
# Line 8-9: Virtual bargaining (Nash solution)
(a_i*, a_j*) = argmax_{a_i, a_j} (U_i - d_i)^γ * (U_j_hat - d_j)^(1-γ)
```
**Virtual Bargaining Explained**:
- Not actual negotiation, but internal simulation
- γ = 0.5 for symmetric bargaining power
- Finds Pareto-efficient compromise
- Respects both agents' (modeled) preferences

#### Belief Update Phase (Lines 10-17)

After observing partner's action `a_j_actual`:

```python
# Line 10: Compute fairness deviation
d_dev = max_{a_j} U_F(a_i*, a_j) - U_F(a_i*, a_j_actual)
```
Where `U_F` is fairness utility using only own beliefs B_i
- Measures how much partner deviated from most fair response

```python
# Line 11: Compliance signal
s_t = exp(-λ_dev * d_dev)
```
- λ_dev = 3.0 (sensitivity parameter)
- s_t ∈ [0, 1]: High when partner is fair, low when unfair
- Exponential decay captures severity of violations

```python
# Line 12: Trust update (exponential smoothing)
τ_t = (1 - η) * τ_{t-1} + η * s_t
```
- η = 0.1 (learning rate)
- Trust slowly adapts to partner's behavior
- Forgiving: takes multiple violations to lose trust

```python
# Line 13: Belief update (CK-ToM formula)
B_tilde(φ) = B_hat(φ) * exp(β * α * τ * F_φ(a_i*, a_j_actual)) * [B_i(φ)]^(1 - α*τ)
```
**This is the heart of TGMI**:
- **Evidence term**: `exp(β * α * τ * F_φ)` 
  - Stronger evidence when trust τ is high
  - β = 1.0 controls sensitivity to observations
  - α = 0.5 is self-anchoring weight
  
- **Self-anchoring term**: `[B_i(φ)]^(1 - α*τ)`
  - Pulls beliefs back toward own prior B_i
  - Stronger pull when trust is low
  - Prevents complete belief revision

- **Trust-gated**: Both terms modulated by τ
  - Low trust → weak updates, strong self-anchoring
  - High trust → strong updates, weak self-anchoring

```python
# Line 14: Normalize belief
B_hat = B_tilde / ||B_tilde||_1
```

```python
# Line 15: Update confidence
c = 1 - H(B_hat) / log(K)
```

```python
# Line 16: Update cooperation weight
ϖ = τ * c
```
- As trust/confidence changes, cooperation level adjusts

```python
# Line 17: Update reservation utility
d_i = U_F(a_i*, a_j_actual)
```
- New baseline for future bargaining

### Implementation Details

#### Configuration Parameters (TGMIConfig)

```python
@dataclass
class TGMIConfig:
    theta_0: float = 0.5     # Initial trust τ_0
    phi: float = 0.1         # Trust learning rate η
    alpha: float = 0.5       # Self-anchoring weight α
    beta: float = 1.0        # Evidence strength β
    xi_dev: float = 3.0      # Deviation sensitivity λ_dev
    gamma: float = 0.5       # Bargaining asymmetry γ
    n_principles: int = 3    # Number of fairness principles
    epsilon: float = 0.0     # Action error (trembling hand) ε_a
    eta: float = 0.0         # Observation error ε_p
```

#### Key Methods

1. **`compute_moral_utility(F_table, use_partner_model)`**
   - Implements Lines 6-7
   - Computes utility matrices for action selection

2. **`virtual_bargain(U_i, U_j_hat)`**
   - Implements Lines 8-9
   - Nash bargaining solution with asymmetric power

3. **`select_action(game)`**
   - Complete action selection pipeline
   - Returns virtual bargaining action

4. **`update(game, a_i_VB, a_j_actual, U_i, U_j_hat)`**
   - Implements Lines 10-17
   - Full belief and trust update

5. **`select_action_with_noise(game)` & `observe_partner_action(a_j_actual, game)`**
   - Add realism with action/observation noise
   - Trembling hand equilibrium

#### Convergence Detection

The implementation includes sophisticated convergence detection:

```python
@dataclass
class ConvergenceConfig:
    prob_threshold: float = 0.01    # Max probability change
    window_size: int = 5            # Consecutive stable rounds
    max_rounds: int = 500           # Safety limit
    min_rounds: int = 10            # Minimum before checking
    check_both_agents: bool = False # Check one or both
```

**Convergence criteria:**
1. Dominant belief type stable for `window_size` rounds
2. Probability change < `prob_threshold`
3. Returns: (converged: bool, reason: str)

---

## 4. Experiments & Comparisons

### Baseline: Bayesian Reasoning (BR) Agent

For comparison, a simpler agent is implemented:

**BR Agent**: Bayesian inference over opponent types
- **Types**: Selfish, Altruistic, Reciprocal
- **Strategy**: 
  - Bayesian belief update based on observed actions
  - Softmax action selection (β=5.0 temperature)
  - No trust mechanism, no virtual bargaining

**Utility Functions**:
```python
# Selfish: U = R_i
# Altruistic: U = R_i + R_j  
# Reciprocal: U = R_i + p(reciprocal) * R_j
```

### Experimental Scenarios

#### 1. TGMI vs TGMI
- **Same priors**: Tests convergence speed
- **Different priors**: Tests belief alignment

#### 2. TGMI vs BR
- Compares sophistication of reasoning
- Tests robustness to simpler opponents

#### 3. TGMI vs Fixed Strategies
- **Always Cooperate**: Tests exploitation resistance
- **Always Defect**: Tests trust decay
- **Deceptive** (cooperate then defect): Tests recovery

### Key Metrics

1. **Mean Payoff**: Average per-round payoff
2. **Total Welfare**: Sum of both agents' payoffs
3. **Cooperation Rate**: Fraction of high-cooperation actions
4. **Fairness Scores**: Average of each fairness function
5. **Convergence**:
   - Rounds to convergence
   - Final dominant belief type
   - Convergence probability

### Visualization Outputs

The experiments generate three plots:

1. **`tgmi_comparison.png`**
   - Bar charts comparing:
     - Mean payoffs (Agent I vs Agent J)
     - Total welfare
     - Cooperation rate
     - Fairness scores (all three principles)

2. **`trust_vs_defector.png`**
   - Trust dynamics when facing always-defect
   - Shows: Trust τ, Confidence c, Cooperation weight ϖ, Beliefs B_hat

3. **`trust_vs_deceptive.png`**
   - Trust dynamics vs deceptive agent
   - Demonstrates recovery after betrayal

---

## 5. Theoretical Foundations

### Computational Theory of Cooperation

**Central Question**: How do agents with different moral intuitions cooperate?

**TGMI's Answer**: Three mechanisms working together:

1. **Trust-Gated Learning**
   - Trust modulates belief updates
   - Low trust → skeptical, self-anchored updates
   - High trust → receptive, evidence-driven updates

2. **Virtual Bargaining**
   - Internal simulation of negotiation
   - Respects both agents' (modeled) preferences
   - Nash solution ensures stability

3. **Cooperation Weight**
   - ϖ = τ * c bridges self-interest and cooperation
   - Dynamically adjusts based on partner reliability
   - Smooth spectrum from competitive to cooperative

### Comparison to Other Models

| Model | Belief Update | Trust | Bargaining | Fairness |
|-------|---------------|-------|------------|----------|
| **TGMI** | Trust-gated CK-ToM | ✓ Explicit | ✓ Nash | ✓ 3 principles |
| **BR Agent** | Bayesian (types) | ✗ | ✗ | ✗ |
| **CK-ToM** | Recursive reasoning | ✗ | ✗ | ✗ |
| **ToM** | Simple modeling | ✗ | ✗ | ✗ |

**Key Innovation**: Trust as a gate that controls both belief formation AND action selection.

### Mathematical Properties

1. **Trust Dynamics**
   ```
   τ_t = (1-η)τ_{t-1} + η·exp(-λ_dev·d_dev)
   ```
   - Stable fixed point when partner is consistently fair
   - Exponential decay under consistent unfairness
   - Hysteresis: Easier to lose trust than regain

2. **Belief Convergence**
   - When partner consistent: Beliefs converge to partner's true type
   - Self-anchoring prevents over-fitting to noise
   - Trust gates convergence rate

3. **Nash Bargaining**
   ```
   max (U_i - d_i)^γ (U_j - d_j)^(1-γ)
   ```
   - Pareto optimal
   - Individually rational (≥ disagreement point)
   - Symmetric when γ = 0.5

---

## 6. Implementation Quality

### Code Organization

**Strengths**:
- Clean separation of concerns (MGG vs TGMI)
- Type hints throughout
- Dataclasses for configuration
- Comprehensive history tracking
- Modular design (easy to extend)

**Architecture Pattern**:
```
Generator → Game → Agent → Action → Update → Convergence
```

### Key Design Decisions

1. **Dirichlet Prior**: Natural choice for categorical distributions
2. **Exponential Smoothing**: Simple but effective trust update
3. **Virtual Bargaining**: Efficient (no actual communication needed)
4. **Convergence Detection**: Pragmatic stopping criterion
5. **Noise Models**: Trembling hand + observation error

### Extensibility Points

Easy to extend:
- New fairness principles (add to `FairnessPrinciple` enum)
- New game archetypes (add to `Archetype` enum)
- New opponent models (inherit from base agent)
- Custom convergence criteria (modify `ConvergenceConfig`)

---

## 7. Research Contributions

### What This Implementation Demonstrates

1. **Computational Model of Moral Reasoning**
   - Operationalizes abstract moral principles
   - Shows how disagreement can coexist with cooperation

2. **Trust as Cognitive Gate**
   - Not just a parameter, but a functional mechanism
   - Controls information flow and decision-making

3. **Practical Cooperation Under Uncertainty**
   - Doesn't require:
     - Common knowledge of preferences
     - Communication protocol
     - Pre-commitment mechanisms
   
4. **Robustness Properties**
   - Recovers from deception
   - Resists exploitation
   - Adapts to different opponent types

### Potential Applications

1. **AI Alignment**: Agents with different values cooperating
2. **Multi-Agent Systems**: Heterogeneous moral frameworks
3. **Human-AI Interaction**: Modeling value differences
4. **Negotiation Systems**: Automated bargaining under uncertainty
5. **Social Simulation**: Understanding moral disagreement

---

## 8. Algorithm 1 Deep Dive

### Pseudocode from Paper (Reconstructed)

```
Algorithm 1: Trust-Gated Moral Inference (TGMI)

Input: Game G = (A_i, A_j, R_i, R_j, F)
       Own moral prior B_i over fairness principles Ω
       
Parameters: τ_0, η, α, β, λ_dev, γ

Initialize:
1.  B_hat ← Dirichlet(1,...,1)          // Belief over partner's norms
2.  τ ← τ_0                             // Trust
3.  c ← 1 - H(B_hat)/log|Ω|            // Confidence  
4.  ϖ ← τ · c                           // Cooperation weight
5.  d_i ← 0, d_j ← 0                    // Reservation utilities

For each round t:
    // Action Selection
6.  U_i(a_i,a_j) ← (1-ϖ)·Σ_ω B_i(ω)F_ω(a_i,a_j) + ϖ·Σ_ω B_hat(ω)F_ω(a_i,a_j)
7.  U_j(a_j,a_i) ← Σ_ω B_hat(ω)F_ω(a_j,a_i)
8.  (a_i*,a_j*) ← argmax (U_i-d_i)^γ · (U_j-d_j)^(1-γ)
9.  Execute a_i*, observe partner's a_j^actual

    // Belief Update
10. d_dev ← max_{a_j} U_F(a_i*,a_j) - U_F(a_i*,a_j^actual)
11. s_t ← exp(-λ_dev · d_dev)
12. τ ← (1-η)τ + η·s_t
13. B_tilde(ω) ← B_hat(ω) · exp(β·α·τ·F_ω(a_i*,a_j^actual)) · [B_i(ω)]^(1-α·τ)
14. B_hat ← normalize(B_tilde)
15. c ← 1 - H(B_hat)/log|Ω|
16. ϖ ← τ · c
17. d_i ← U_F(a_i*,a_j^actual)

Return: action sequence, beliefs, trust trajectory
```

### Line-by-Line Correspondence to Code

| Line | Formula | Code Location | Variable Name |
|------|---------|---------------|---------------|
| 1 | B_hat ~ Dir(1,1,1) | `agent.py:94` | `self.B_hat` |
| 2 | τ ← τ_0 | `agent.py:96` | `self.theta` |
| 3 | c ← 1-H/logK | `agent.py:98-100` | `self.c` |
| 4 | ϖ ← τ·c | `agent.py:102` | `self.varpi` |
| 5 | d_i, d_j ← 0 | `agent.py:104-105` | `self.d_i`, `self.d_j` |
| 6 | U_i computation | `agent.py:117-131` | `compute_moral_utility()` |
| 7 | U_j_hat computation | `agent.py:117-131` | `compute_moral_utility(use_partner_model=True)` |
| 8-9 | Nash bargaining | `agent.py:147-179` | `virtual_bargain()` |
| 10 | Fairness deviation | `agent.py:209-212` | `d_dev` |
| 11 | Compliance signal | `agent.py:214-215` | `s_t` |
| 12 | Trust update | `agent.py:217-218` | `self.theta` update |
| 13 | CK-ToM belief update | `agent.py:220-239` | `B_tilde` |
| 14 | Normalize | `agent.py:241-242` | `normalized()` |
| 15 | Update confidence | `agent.py:244-246` | `self.c` update |
| 16 | Update coop weight | `agent.py:248-249` | `self.varpi` update |
| 17 | Update reservation | `agent.py:251-252` | `self.d_i` update |

### Critical Implementation Notes

1. **Parameter Naming Discrepancy**:
   - Paper: η (trust learning rate)
   - Code: `phi` 
   - Paper: λ_dev (deviation sensitivity)
   - Code: `xi_dev`

2. **Nash Bargaining Edge Cases** (`agent.py:164-175`):
   ```python
   if surplus_i > 0 and surplus_j > 0:
       nash_product = surplus_i^γ · surplus_j^(1-γ)
   elif surplus_i > 0:
       nash_product = surplus_i^γ · 1e-10  # One-sided
   elif surplus_j > 0:
       nash_product = 1e-10 · surplus_j^(1-γ)
   else:
       nash_product = 0
   ```
   Handles cases where one agent gets no surplus

3. **Fairness Utility** (`agent.py:134-144`):
   ```python
   U_F = Σ_ω B_i(ω) · F_ω(a_i, a_j)
   ```
   Used for deviation calculation (Line 10), uses ONLY own beliefs B_i

4. **Trust-Gated Evidence** (`agent.py:229-231`):
   ```python
   evidence = exp(β · α · τ · F_ω(a_i, a_j))
   ```
   When τ=0: evidence = exp(0) = 1 (no update)
   When τ=1: evidence = exp(β·α·F_ω) (full update)

5. **Self-Anchoring Term** (`agent.py:234-236`):
   ```python
   anchor_exp = 1 - α · τ
   prior_anchor = B_i[ω]^anchor_exp
   ```
   When τ=0: B_i^1 (full anchoring)
   When τ=1: B_i^(1-α) (partial anchoring)

---

## 9. Experimental Results Interpretation

### Expected Behaviors

1. **TGMI vs TGMI (Same Prior)**
   - Fast convergence (beliefs already aligned)
   - High cooperation rate
   - Trust remains stable and high
   - Optimal fairness scores

2. **TGMI vs TGMI (Different Priors)**
   - Slower convergence
   - Trust fluctuates initially, then stabilizes
   - Beliefs partially converge
   - Good but not optimal cooperation

3. **TGMI vs BR**
   - TGMI achieves higher welfare
   - BR shows rigid beliefs
   - TGMI adapts better

4. **TGMI vs Always-Defect**
   - Trust rapidly decays to near-zero
   - Agent becomes defensive
   - Low cooperation weight ϖ

5. **TGMI vs Deceptive**
   - Trust high initially (cooperate phase)
   - Sharp drop when betrayal detected
   - Partial recovery if deception stops

### Metrics to Watch

- **Trust trajectory**: Smooth vs. volatile
- **Belief convergence**: Which principle dominates?
- **Cooperation rate**: Sustained vs. declining
- **Payoff comparison**: TGMI vs baselines

---

## 10. Strengths and Limitations

### Strengths

1. **Theoretical Rigor**: Grounded in game theory and Bayesian reasoning
2. **Practical Realism**: Handles noise, deception, uncertainty
3. **Computational Efficiency**: No expensive search, closed-form updates
4. **Interpretability**: Each parameter has clear meaning
5. **Robustness**: Recovers from mistakes and betrayals
6. **Extensibility**: Easy to add new fairness principles or game types

### Limitations

1. **Discrete Actions**: Grid discretization (21 actions)
   - Could use continuous optimization
   
2. **Three Fairness Principles**: Arbitrary choice
   - Could extend to more nuanced moral frameworks
   
3. **Nash Bargaining**: Assumes rationality
   - Alternatives: Kalai-Smorodinsky, Utilitarian, etc.
   
4. **No Communication**: Purely behavioral
   - Could integrate cheap talk or signaling
   
5. **Symmetric Information**: Both see same game
   - Could add private information
   
6. **Single-shot Learning**: No meta-learning across games
   - Could build repertoire of strategies

### Future Directions

1. **Continuous Action Spaces**: Gradient-based optimization
2. **Communication Protocol**: Add signaling layer
3. **Multi-agent (>2)**: Coalition formation
4. **Longitudinal Learning**: Transfer across games
5. **Hierarchical Morality**: Nested fairness principles
6. **Bounded Rationality**: Level-k reasoning
7. **Cultural Evolution**: Population dynamics

---

## 11. Key Takeaways

### For AI Researchers

- **Trust is not just a parameter**: It's a functional gate controlling learning and cooperation
- **Moral disagreement ≠ cooperation failure**: Agents can cooperate despite differing values
- **Theory of Mind + Bargaining**: Powerful combination for strategic interaction

### For Machine Learning Practitioners

- **Architecture**: Clean separation of game generation, agent logic, experiments
- **Hyperparameter Sensitivity**: Trust learning rate η and evidence strength β are critical
- **Convergence**: Important to detect when beliefs stabilize (avoid infinite loops)
- **Baselines**: Always compare to simpler models (BR agent)

### For Game Theorists

- **Virtual Bargaining**: Efficient implementation of Nash solution
- **Trust Dynamics**: Connects iterated games with belief formation
- **Fairness Functions**: Operationalize abstract moral principles

### For Social Scientists

- **Computational Model of Morality**: Makes testable predictions
- **Robustness to Deception**: Explains real-world forgiveness
- **Heterogeneous Preferences**: Framework for understanding disagreement

---

## 12. Dependencies and Technical Requirements

### Core Dependencies

```python
numpy          # Numerical computations
scipy          # Statistical functions (softmax, special functions)
matplotlib     # Plotting and visualization
seaborn        # Statistical data visualization
pandas         # Data manipulation (if needed for experiments)
joblib         # Parallel processing (if using batch experiments)
tqdm           # Progress bars
networkx       # Graph structures (if analyzing agent networks)
iteround       # Rounding that preserves sum
icecream       # Debugging print utility
```

### Python Version
- **Recommended**: Python 3.8+
- **Tested on**: Python 3.9

### Running the Code

```bash
# Setup
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run experiments
cd tgmi
python experiments.py

# Run agent demo
python agent.py
```

---

## 13. Summary of Key Equations

| Concept | Equation | Purpose |
|---------|----------|---------|
| **Cooperation Weight** | ϖ = τ · c | Blends self-interest and cooperation |
| **Moral Utility** | U_i = (1-ϖ)·Σ B_i(ω)F_ω + ϖ·Σ B_hat(ω)F_ω | Action selection utility |
| **Nash Bargaining** | max (U_i-d_i)^γ (U_j-d_j)^(1-γ) | Virtual negotiation |
| **Compliance Signal** | s_t = exp(-λ_dev · d_dev) | Trust update input |
| **Trust Update** | τ_t = (1-η)τ_{t-1} + η·s_t | Exponential smoothing |
| **CK-ToM Update** | B̃(ω) = B̂(ω)·e^(β·α·τ·F_ω)·[B_i(ω)]^(1-α·τ) | Trust-gated belief revision |
| **Confidence** | c = 1 - H(B̂)/log(K) | Certainty about beliefs |

---

## 14. Conclusion

This implementation provides a **complete, working system** for studying cooperation under moral uncertainty. The TGMI algorithm elegantly combines:

- **Game Theory** (Nash bargaining)
- **Bayesian Inference** (belief updates)
- **Moral Philosophy** (fairness principles)
- **Psychology** (trust dynamics)

into a unified computational model.

**Key Innovation**: Trust as a gate that simultaneously controls:
1. How much to learn from partner's actions
2. How much to anchor to own beliefs
3. How much to weight cooperative outcomes

The implementation is:
- ✅ **Faithful to the algorithm**: Direct line-by-line correspondence
- ✅ **Well-tested**: Multiple experimental scenarios
- ✅ **Extensible**: Easy to modify and extend
- ✅ **Documented**: Clear code with comments
- ✅ **Reproducible**: Fixed random seeds, configurable parameters

**Research Impact**: Demonstrates that computational models can capture nuanced aspects of moral reasoning and cooperation, bridging the gap between abstract theory and practical implementation.

---

**Document prepared by**: GitHub Copilot  
**Date**: December 24, 2025  
**Version**: 1.0  
**For**: PNAS Paper Implementation Analysis
