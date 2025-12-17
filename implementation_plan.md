Perfect, this updated Algorithm 1 is concrete enough that we can now design a clean implementation plan around it. I’ll:

* Map the updated algorithm line-by-line into what you’d code.
* Spell out the exact components your codebase needs (classes, functions, hyperparams).
* Then outline how to plug in a **Bayesian Reciprocator (BR)** agent for comparison.

I’ll use your updated pseudo-code and notation as ground truth. 

---

## 1. Decode the updated Algorithm 1 into “what we need to implement”

Here’s a cleaned version of the algorithm in plain math / code-ish notation.

### 1.1. Inputs / primitives

From the header: 

* Fairness principles:
  (F = {\omega_1, \dots, \omega_K}).
* Payoff function:
  (R_i(a_i, a_j)) (you might not use it directly in Alg 1, but it’s part of the overall model).
* Intrinsic moral prior of agent (i):
  (B_i(\omega)) (a probability distribution over (F)).

You also need:

* Discrete action spaces (A_i, A_j) (e.g., ({C,D}) or a larger set).
* Fairness score functions
  (F_\omega(a_i, a_j) \in [0,1]) for each norm (\omega \in F).
  (These are the “norm utilities” defined in the Methods.)

---

### 1.2. Initialization (lines 1–4)

1. **Belief over partner’s norms** (\hat B_{i\to j}(\omega)):

   [
   \hat B_{i\to j}^{(0)} \sim \text{Dirichlet}(1,\dots,1) \quad\Rightarrow\quad
   \mathbb E[\hat B_{i\to j}^{(0)}(\omega)] = \tfrac{1}{|F|}
   ]

2. **Trust and confidence**:

   [
   \vartheta_i^{(0)} = \vartheta_0, \quad
   c_i^{(0)} = 1 - \frac{H(\hat B_{i\to j}^{(0)})}{\log |F|}
   ]

   where (H) is Shannon entropy.

3. **Effective cooperation weight**:

   [
   \varpi_i^{(0)} = \vartheta_i^{(0)} , c_i^{(0)}.
   ]

4. **Reservation utilities**:

   [
   d_i^{(0)} = 0,\quad d_j^{(0)} = 0.
   ]

---

### 1.3. Main loop over rounds (t=1,\dots,T) (line 5)

For each round:

#### (a) Trust–confidence–gated moral utilities (lines 6–8)

Loop over all joint actions ((a_i,a_j) \in A_i \times A_j).

1. **Agent i’s own moral utility** (line 7):

   [
   U_i(a_i, a_j)
   = \sum_{\omega \in F}
   \big[(1 - \varpi_i) B_i(\omega) + \varpi_i \hat B_{i\to j}(\omega)\big]
   , F_\omega(a_i, a_j).
   ]

   * ((1-\varpi_i)): weight on *own* moral prior.
   * (\varpi_i): weight on inferred partner norms.

2. **Agent i’s internal model of partner j’s utility** (line 8):

   [
   \hat U^{(i)}*j(a_j, a_i)
   = \sum*{\omega \in F}
   \hat B_{i\to j}(\omega) , F_\omega(a_j, a_i).
   ]

   * Crucial updates reflected here:

     * This is **(\hat U_j)**, the internal model, not the true (U_j).
     * It uses **only** (\hat B_{i\to j}), not (B_j), and relies on the CK-ToM symmetry (\hat B_{i\to j} \approx \hat B_{j\to i}). 

#### (b) Virtual bargaining (VB) step (line 8 block)

Joint VB solution (conceptual coordination):

[
(a_i^{VB}, a_j^{VB}) = \arg\max_{a_i,a_j}
\Big[
(U_i(a_i,a_j) - d_i)^+

* \rho , (\hat U^{(i)}_j(a_j,a_i) - d_j)^+
  \Big],
  ]

where:

* ((x)^+ = \max(x,0)),
* (\rho \in (0,1)) is the bargaining asymmetry weight (“ϱ” in the text). 

Agent (i) will **intend** to play (a_i^{VB}). The partner’s action might be:

* literally (a_j^{VB}) (if partner is also TGMI and they coordinate), or
* something else (e.g., if partner is a BR agent or noisy).

#### (c) Fairness deviation & trust update (lines 9–12)

9. **Fairness-only utility under own moral prior** (line 9):

[
U_i^F(a_i, a_j)
= \sum_{\omega \in F} B_i(\omega) F_\omega(a_i, a_j).
]

10. **Fairness deviation** (line 10):

[
d_i^{(t)} =
\max_{a_j' \in A_j} U_i^F(a_i^{VB}, a_j') ;-;
U_i^F(a_i^{VB}, a_j^{VB}).
]

Interpretation: shortfall between:

* the *best* fairness outcome i could get if j behaved optimally (from i’s fairness perspective), and
* the fairness outcome predicted by VB (or realized, depending on how you plug in actual vs VB actions). 

11. **Compliance signal** (line 11):

[
s_i^{(t)} = \exp(-\xi_{\text{dev}} , d_i^{(t)}),
]

where (\xi_{\text{dev}}) (your “ς_dev”) controls sensitivity to fairness deviations.

12. **Trust update** (line 12):

[
\vartheta_i^{(t+1)}
= (1 - \phi) , \vartheta_i^{(t)}

* \phi , s_i^{(t)},
  ]

with (\phi \in (0,1)) a trust learning rate.

#### (d) Belief update & re-gating (lines 13–17)

13. **Belief update for each norm (\omega)** (line 13):

[
\tilde B_{i\to j}^{(t+1)}(\omega)
=================================

\hat B_{i\to j}^{(t)}(\omega)
\cdot
\exp\big(\beta , \vartheta_i^{(t)} F_\omega(a_i^{VB}, a_j^{VB})\big)
\cdot
\big(B_i(\omega)\big)^{,1 - \vartheta_i^{(t)} \varpi_i^{(t)}}
]

* First factor: previous belief.
* Second factor: fairness evidence term, scaled by trust ((\vartheta_i^t)) and sensitivity (\beta).
* Third factor: **self-anchoring** on own prior (B_i(\omega)) with exponent (1 - \vartheta_i \varpi_i).
  This is where the “weights being swapped” shows up: trust & cooperation weight control the balance between new fairness evidence vs prior anchoring. 

(Exact constants (\beta), exponent details you’ll treat as hyperparams following the text.)

14. **Normalize** (line 14):

[
\hat B_{i\to j}^{(t+1)}(\omega)
= \frac{\tilde B_{i\to j}^{(t+1)}(\omega)}
{\sum_{\omega' \in F} \tilde B_{i\to j}^{(t+1)}(\omega')}.
]

15. **Update confidence** (line 15):

[
c_i^{(t+1)}
= 1 - \frac{H\big(\hat B_{i\to j}^{(t+1)}\big)}{\log |F|}.
]

16. **Update cooperation weight** (line 16):

[
\varpi_i^{(t+1)}
= \vartheta_i^{(t+1)} , c_i^{(t+1)}.
]

17. **Update reservation utility** (line 17):

[
d_i^{(t+1)} = U_i^F(a_i^{VB}, a_j^{VB}).
]

---

## 2. Concrete implementation plan (what to actually code)

Think in terms of **three modules**:

1. **Fairness & games**
2. **TGMI agent implementation**
3. **Experiment harness + BR comparison**

### 2.1. Module 1 – Fairness principles & game environment

**(a) Represent fairness principles**

In Python pseudocode:

```python
from enum import Enum
import numpy as np

class FairnessPrinciple(Enum):
    MAX_SUM = 0
    EQUAL_SPLIT = 1
    RAWLS = 2
    # add more if needed

F = list(FairnessPrinciple)
K = len(F)
```

**(b) Implement F_ω(a_i, a_j)**

You’ll need a function:

```python
def fairness_score(phi, payoff_i, payoff_j):
    if phi == FairnessPrinciple.MAX_SUM:
        total = payoff_i + payoff_j
        # Normalize appropriately, e.g., min–max over all joint actions
        return normalize_total(total)
    elif phi == FairnessPrinciple.EQUAL_SPLIT:
        diff = abs(payoff_i - payoff_j)
        return 1.0 - normalize_diff(diff)
    elif phi == FairnessPrinciple.RAWLS:
        m = min(payoff_i, payoff_j)
        return normalize_min(m)
```

Under the hood, you’ll precompute:

* All joint payoffs (R_i(a_i,a_j), R_j(a_i,a_j)),
* Then F_ω(a_i,a_j) for each ω and joint action and cache them for speed.

**(c) Game environment wrapper**

A simple `Game` class:

```python
class RepeatedGame:
    def __init__(self, payoff_matrix_i, payoff_matrix_j, actions_i, actions_j, fairness_table):
        self.R_i = payoff_matrix_i   # shape [n_ai, n_aj]
        self.R_j = payoff_matrix_j   # shape [n_ai, n_aj]
        self.actions_i = actions_i
        self.actions_j = actions_j
        self.F = fairness_table      # shape [K, n_ai, n_aj]
```

You can sample many games if you later want to reproduce the Moral Game Generator.

---

### 2.2. Module 2 – TGMI agent

Design a `TGMI_Agent` class whose internal state mirrors Algorithm 1 exactly.

**State variables:**

```python
class TGMI_Agent:
    def __init__(self, B_i, F, n_actions_i, n_actions_j,
                 theta0=0.5, phi=0.1, beta=1.0, xi_dev=1.0):
        self.F = F                     # list of fairness principles
        self.B_i = np.array(B_i)       # intrinsic moral prior over F, shape [K]
        
        # 1. Belief over partner's norms
        self.B_hat = np.random.dirichlet(np.ones(len(F)))  # B̂_{i→j}
        
        # 2. Trust and confidence
        self.theta = theta0            # ϑ_i
        self.c = 1.0 - entropy(self.B_hat) / np.log(len(F))
        
        # 3. Effective cooperation weight
        self.varpi = self.theta * self.c   # ϖ_i
        
        # 4. Reservation utilities
        self.d_i = 0.0
        self.d_j = 0.0                 # use fixed or symmetrical update
        
        # Params
        self.phi = phi                 # trust update step
        self.beta = beta               # fairness evidence strength
        self.xi_dev = xi_dev           # dev sensitivity
        self.n_actions_i = n_actions_i
        self.n_actions_j = n_actions_j
```

**Helper functions:**

* `entropy(p)`: Shannon entropy.
* `compute_U_i(F_table)`: compute own moral utility per joint action.
* `compute_Uj_hat(F_table)`: compute internal model of partner’s utility.
* `virtual_bargain(U_i, Uj_hat)`: choose `(a_i_VB, a_j_VB)` via argmax.
* `update_trust_and_beliefs(...)`: implement lines 9–17.

**Step function for one round:**

```python
def select_action(self, game):
    # game.F has shape [K, n_ai, n_aj]

    # 1. Compute trust–confidence–gated utilities U_i and U_j_hat
    K, n_ai, n_aj = game.F.shape
    U_i = np.zeros((n_ai, n_aj))
    Uj_hat = np.zeros((n_aj, n_ai))   # note the swapped arguments
    
    for a_i in range(n_ai):
        for a_j in range(n_aj):
            # fairness scores vector over norms
            F_vec = game.F[:, a_i, a_j]      # shape [K]

            mixture_i = (1 - self.varpi) * self.B_i + self.varpi * self.B_hat
            U_i[a_i, a_j] = np.dot(mixture_i, F_vec)

            # internal partner utility; note order (a_j, a_i)
            F_vec_partner = game.F[:, a_j, a_i]
            Uj_hat[a_j, a_i] = np.dot(self.B_hat, F_vec_partner)
    
    # 2. Virtual bargaining argmax
    a_i_VB, a_j_VB = self.virtual_bargain(U_i, Uj_hat)

    # In a symmetric setup, agent i would *intend* to play a_i_VB:
    return a_i_VB, (a_i_VB, a_j_VB, U_i, Uj_hat)
```

**Virtual bargaining implementation:**

```python
def virtual_bargain(self, U_i, Uj_hat, rho=0.5):
    best_val = -np.inf
    best_pair = (0, 0)
    n_ai, n_aj = U_i.shape

    for a_i in range(n_ai):
        for a_j in range(n_aj):
            val_i = max(U_i[a_i, a_j] - self.d_i, 0.0)
            val_j = max(Uj_hat[a_j, a_i] - self.d_j, 0.0)
            objective = val_i + rho * val_j
            if objective > best_val:
                best_val = objective
                best_pair = (a_i, a_j)

    return best_pair
```

**After environment returns partner’s action** (if you differentiate between VB-intended and actual):

```python
def update_after_round(self, game, a_i_VB, a_j_real, U_i, Uj_hat):
    # 9. Fairness-only utility UF_i
    n_ai, n_aj = U_i.shape
    UF = np.zeros((n_ai, n_aj))
    for a_i in range(n_ai):
        for a_j in range(n_aj):
            # B_i . F_ω(a_i,a_j)
            F_vec = game.F[:, a_i, a_j]
            UF[a_i, a_j] = np.dot(self.B_i, F_vec)
    
    # 10. Fairness deviation
    # ideal fairness: best over j’s actions holding a_i_VB fixed
    best_over_j = UF[a_i_VB, :].max()
    realized = UF[a_i_VB, a_j_real]    # or a_j_VB if you stick to the algorithm literally
    d_dev = best_over_j - realized     # d_i^{(t)}

    # 11. Compliance
    s_t = np.exp(-self.xi_dev * d_dev)

    # 12. Trust update
    self.theta = (1 - self.phi) * self.theta + self.phi * s_t

    # 13. Belief update over norms
    # fairness evidence at VB outcome (you can also use actual outcome)
    F_vb = game.F[:, a_i_VB, a_j_real]  # vector over ω

    B_tilde = np.zeros_like(self.B_hat)
    for idx, omega in enumerate(self.F):
        evid = np.exp(self.beta * self.theta * F_vb[idx])
        prior_anchor = self.B_i[idx] ** (1 - self.theta * self.varpi)
        B_tilde[idx] = self.B_hat[idx] * evid * prior_anchor

    # 14. Normalize
    self.B_hat = B_tilde / B_tilde.sum()

    # 15. Update confidence
    self.c = 1.0 - entropy(self.B_hat) / np.log(len(self.F))

    # 16. Update cooperation weight
    self.varpi = self.theta * self.c

    # 17. Update reservation utility
    self.d_i = UF[a_i_VB, a_j_real]
```

You can tweak whether you use VB outcome `(a_i_VB,a_j_VB)` or actual outcome `(a_i_real,a_j_real)` in steps 10, 13, and 17; the algorithm text uses VB forms, but for realistic play you’ll usually plug in actual actions.

---

### 2.3. Module 3 – Experiment harness & BR comparison

Now to your question: **“should we compare this agent with the BR agent, and if yes, how?”**
Yes, and this is the natural place to slot that in.

#### (a) Implement BR agent from the earlier paper

A `BR_Agent` roughly needs: 

* A set of **types** (Selfish, BR, Cooperative, etc.).
* A belief vector over partner’s type.
* A **type-specific utility** function (e.g. linear combination of payoffs).
* A Bayesian update rule for the belief over types given observed behavior.
* A best-response or virtual-bargaining-like policy based on its expected utilities.

You can:

* Keep the exact equations from the BR paper for consistency.
* Wrap it in a similar interface: `choose_action(game)` and `update_after_round(...)`.

#### (b) Scenarios to run

For each game (or distribution of games):

1. **Homogeneous populations**:

   * TGMI vs TGMI,
   * BR vs BR.

2. **Cross-type matchups**:

   * TGMI vs BR,
   * TGMI vs Selfish, AllD, ZD-Extort, etc.

3. **Moral heterogeneity**:

   * Draw agent moral priors (B_i) from different regions of the simplex (utilitarian-heavy vs Rawls-heavy vs equality-heavy).
   * Run many matches and watch:

     * trust trajectories (\vartheta_i(t)),
     * cooperation frequencies,
     * fairness scores and payoff distributions.

#### (c) Metrics to log

For each pair of agents and game distribution:

* **Cooperation rate**: fraction of rounds with “cooperative” actions (however you define it per game).
* **Average payoffs**: (\mathbb E[R_i + R_j]) and distribution across agents.
* **Fairness scores**: (\mathbb E[F_\omega(a)]) for key norms.
* **Trust & belief dynamics**:

  * trajectory of (\vartheta_i(t)),
  * entropy of (\hat B_{i\to j}(t)),
  * evolution of (\varpi_i(t)).

You’d expect:

* Under **shared or mildly different moral priors**, both BR and TGMI can sustain cooperation.
* Under **strong moral heterogeneity**, BR tends to misclassify “morally different but fair” behavior as defection, tanking cooperation; TGMI uses the norm-space beliefs (\hat B_{i\to j}) + trust gate to preserve cooperation more often.

That comparison directly probes the **value of TGMI’s CK-ToM over moral norms** vs BR’s simpler utility-matching.

---

## 3. What you should expect to *observe* when you run this

If the code matches Algorithm 1:

* Against **fair, compatible partners**:

  * (\vartheta_i) will drift upwards toward 1.
  * Entropy of (\hat B_{i\to j}) will decrease; (c_i) will rise.
  * (\varpi_i = \vartheta_i c_i) becomes large → actions more strongly reflect shared moral reasoning.
  * Joint outcomes will cluster on fairness-high, payoff-decent actions.

* Against **exploitative or misaligned partners**:

  * Fairness deviation (d_i^{(t)}) will be large → (s_i^{(t)}) small.
  * Trust (\vartheta_i) will decrease.
  * Belief updates may remain high-entropy → (c_i) low.
  * (\varpi_i) stays small → fallback to own prior, limiting exploitation.

* In **noisy environments**:

  * (\vartheta_i) may fluctuate but stay moderate if the partner is usually fair.
  * Beliefs will adjust gradually, reflecting a stable but not fully confident moral model.

These dynamics are exactly what the paper claims mathematically and in the simulation section, so if your curves look qualitatively similar, you’re on the right track.

---

If you want next, we can:

* Tighten this into a **minimal runnable prototype** (with concrete fairness functions and a tiny 2×2 game),
* Or sketch the BR agent’s update and action-selection loop so you can slot it directly into the same experiment harness.
