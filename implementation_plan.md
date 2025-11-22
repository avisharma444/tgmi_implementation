A **step-by-step implementation plan** for TGMI following the guidelines.

I’ll assume Python + NumPy, but the structure is language-agnostic. I’ll organize by the recommended sequence:

1. Moral Game Generator (MGG)
2. TGMI agent
3. Logging
4. Moran process

For each, I’ll give: data structures, function signatures, and a build-&-test order.

---

## 0. High-level architecture

Suggested module layout:

* `mgg.py` – Moral Game Generator (payoff archetypes, fairness functions, game sampling)
* `tgmi_agent.py` – TGMI agent class + partner-state management
* `simulation.py` – intragenerational rollouts of pairs in MGG
* `logging_utils.py` – run logging helpers
* `moran.py` – evolutionary loop over population
* `config.py` – hyperparameters (η, λ_dev, α, β, γ, ε_a, ω, etc.)

Key design principles:

* Keep **MGG completely independent** of TGMI (it should just know payoffs & fairness functions).
* TGMI should depend only on: fairness values Fϕ(aᵢ,aⱼ), agents’ moral priors Bᵢ, beliefs B̂ᵢ→ⱼ, and game action spaces.

---

## 1. Implement the Moral Game Generator (MGG)

### 1.1 Core objects and types

Define an enum for payoff archetypes:

```python
class Archetype(Enum):
    DILEMMA = 1
    ASSURANCE = 2
    BARGAIN = 3
    PUBLIC_GOODS = 4
```

Define a `Game` object representing a *single* sampled interaction:

```python
@dataclass
class Game:
    archetype: Archetype
    action_space_i: np.ndarray  # e.g. 1D array of discrete actions for i
    action_space_j: np.ndarray  # same for j
    R_i: np.ndarray             # shape (n_ai, n_aj)
    R_j: np.ndarray             # shape (n_ai, n_aj)
    F: Dict[str, np.ndarray]    # e.g. {"max_sum": ..., "equal_split": ..., "rawls": ...}, each (n_ai, n_aj)
    R_max: float                # max payoff used for normalization
```

You can later extend to 3-player games; start with 2-player.

### 1.2 Action space design

Paper uses continuous action spaces; for implementation, start with a **discretized grid**:

```python
action_space = np.linspace(0.0, 1.0, num=21)  # 21 actions from 0 to 1
```

Use the same grid for both players initially; later you can generalize.

### 1.3 Payoff archetypes (Ri(aᵢ, aⱼ))

From the paper: each game draws a payoff function from Dilemma / Assurance / Bargain / Public-Goods, with payoffs normalized to [0,1].

Implementation plan:

* Create four functions:

```python
def payoff_dilemma(ai, aj, params) -> Tuple[float, float]:
    ...

def payoff_assurance(ai, aj, params) -> Tuple[float, float]:
    ...

def payoff_bargain(ai, aj, params) -> Tuple[float, float]:
    ...

def payoff_public_goods(ai, aj, params) -> Tuple[float, float]:
    ...
```

Each returns `(R_i, R_j)` for scalar actions `ai`, `aj`.

* Wrap them into a factory:

```python
def sample_payoff_function(archetype: Archetype, rng, config) -> Callable:
    # sample any needed parameters (e.g., benefit/cost ratios, thresholds)
    # return a closure payoff(ai, aj) -> (R_i, R_j)
```

**Where to get exact formulas?**
The exact parameterizations are in the Supplementary Appendix; the main text only says the archetypes and that `R_i ∈ [0,1]`.
So in the implementation plan:

* Step 1: stub with simple, textbook-like functions (donation game, stag-hunt, bargaining over a pie, public goods).
* Step 2: once you’re happy with the TGMI loop, replace stub functions with the exact equations from the supplement.

### 1.4 Fairness functions Fϕ(aᵢ, aⱼ)

From the paper, fairness mappings:

[
F_{\text{Max-Sum}} = \frac{R_i + R_j}{R_{\max}},\quad
F_{\text{Equal-Split}} = 1 - \frac{|R_i - R_j|}{R_{\max}},\quad
F_{\text{Rawls}} = \frac{\min(R_i, R_j)}{R_{\max}}.
]

Plan:

1. After computing Rᵢ and Rⱼ over the action grid, compute:

   ```python
   R_max = max(R_i.max(), R_j.max())
   F_max_sum = (R_i + R_j) / R_max
   F_equal_split = 1.0 - np.abs(R_i - R_j) / R_max
   F_rawls = np.minimum(R_i, R_j) / R_max
   ```

2. Store in the `Game` object as:

   ```python
   F = {
       "max_sum": F_max_sum,
       "equal_split": F_equal_split,
       "rawls": F_rawls,
   }
   ```

These already lie in [0,1] by construction if Rᵢ,Rⱼ ∈ [0,1].

### 1.5 Sampling a game

Implement a `MoralGameGenerator` class:

```python
class MoralGameGenerator:
    def __init__(self, archetype_probs, action_grid, rng, config):
        self.archetype_probs = archetype_probs  # e.g. uniform over 4 archetypes
        self.action_grid = action_grid
        self.rng = rng
        self.config = config

    def sample_game(self) -> Game:
        # 1. Sample archetype
        archetype = self.rng.choice(list(Archetype), p=self.archetype_probs)

        # 2. Sample payoff function closure
        payoff_fn = sample_payoff_function(archetype, self.rng, self.config)

        # 3. Construct payoff matrices
        A = self.action_grid
        n = len(A)
        R_i = np.zeros((n, n))
        R_j = np.zeros((n, n))
        for idx_i, ai in enumerate(A):
            for idx_j, aj in enumerate(A):
                Ri_ij, Rj_ij = payoff_fn(ai, aj)
                R_i[idx_i, idx_j] = Ri_ij
                R_j[idx_i, idx_j] = Rj_ij

        # 4. Normalize payoffs to [0,1]
        R_min = min(R_i.min(), R_j.min())
        R_max_raw = max(R_i.max(), R_j.max())
        # shift & scale to [0,1]
        R_i = (R_i - R_min) / (R_max_raw - R_min)
        R_j = (R_j - R_min) / (R_max_raw - R_min)
        R_max = max(R_i.max(), R_j.max())

        # 5. Compute fairness functions as above
        F_max_sum = (R_i + R_j) / R_max
        F_equal_split = 1.0 - np.abs(R_i - R_j) / R_max
        F_rawls = np.minimum(R_i, R_j) / R_max

        return Game(
            archetype=archetype,
            action_space_i=A,
            action_space_j=A,
            R_i=R_i,
            R_j=R_j,
            F={"max_sum": F_max_sum,
               "equal_split": F_equal_split,
               "rawls": F_rawls},
            R_max=R_max,
        )
```

This matches the description: each sampled game specifies a payoff surface and fairness environment; all payoffs and fairness values are rescaled to [0,1].

### 1.6 Moral priors for agents

At *agent creation* (not per game), assign:

[
B_i = (w_{\text{Max-Sum}}, w_{\text{Equal-Split}}, w_{\text{Rawls}})
\sim \text{Dirichlet}(1,1,1).
]

Implementation:

```python
def sample_moral_prior(rng, num_principles=3):
    alpha = np.ones(num_principles)
    return rng.dirichlet(alpha)  # returns np.array of length 3
```

---

### 1.7 Tests for MGG

Before touching TGMI:

1. **Unit tests**

* Sample 100 games; assert:

  * `0 ≤ R_i, R_j ≤ 1` and `0 ≤ F[φ] ≤ 1`.
  * `F["equal_split"]` is highest when `R_i == R_j`.
  * `F["max_sum"]` is monotonic in `(R_i + R_j)`.

2. **Sanity plots (optional)**

* Heatmaps of Rᵢ and each Fϕ for one sampled game per archetype.

Once that is solid, move on to TGMI.

---

## 2. Implement the TGMI agent

We follow Algorithm 1 from the paper.

### 2.1 Data structures

#### PartnerState

Per partner j, agent i keeps:

```python
@dataclass
class PartnerState:
    B_hat: np.ndarray   # belief over partner's moral norms, shape (|F|,)
    tau: float          # trust τ_i ∈ [0,1]
    c: float            # confidence c_i ∈ [0,1]
    kappa: float        # κ_i = τ_i * c_i
    d: float            # reservation utility d_i
```

#### TGMI_Agent

```python
class TGMI_Agent:
    def __init__(self, agent_id, moral_prior, hyperparams, rng):
        self.id = agent_id
        self.B = moral_prior            # intrinsic moral prior Bi(ϕ)
        self.hyper = hyperparams        # η, λ_dev, α, β, γ, τ0, ...
        self.rng = rng
        self.partners: Dict[int, PartnerState] = {}
        # for logging
        self.log = []
```

### 2.2 Partner initialization

When i first meets j:

```python
def _init_partner_state(self, partner_id):
    num_phi = len(self.B)
    B_hat = self.rng.dirichlet(np.ones(num_phi))  # Dirichlet(1,...,1)
    H = -np.sum(B_hat * np.log(B_hat + 1e-12))    # Shannon entropy
    c = 1.0 - H / np.log(num_phi)                 # confidence ∈ [0,1]
    tau = self.hyper.tau0
    kappa = tau * c
    d = 0.0
    self.partners[partner_id] = PartnerState(B_hat=B_hat, tau=tau, c=c, kappa=kappa, d=d)
```

### 2.3 Trust–confidence–gated moral utility

From Eq. 2:

[
U_i(a) = \sum_{\phi\in F}
\big[(1-\kappa_i)B_i(\phi) + \kappa_i \hat B_{i\to j}(\phi)\big] F_\phi(a)
]

Implement as:

```python
def moral_utility(self, partner_id, F_at_a: np.ndarray) -> float:
    """
    F_at_a: np.array of length |F| with [F_max_sum(a), F_equal_split(a), F_rawls(a)]
    """
    ps = self.partners[partner_id]
    w = (1.0 - ps.kappa) * self.B + ps.kappa * ps.B_hat
    return float(np.dot(w, F_at_a))
```

### 2.4 Virtual bargaining (joint step)

This is easiest to implement **outside** the agent as a function that sees both agents’ states. From Algorithm 1 and Eq. 3:

[
(a_i^{VB}, a_j^{VB}) = \arg\max_{a_i,a_j}
\big(U_i(a_i,a_j) - d_i\big)*+^\gamma
\big(U_j(a_j,a_i) - d_j\big)*+^{1-\gamma}
]
with ((x)_+ = \max(x,0)).

Implementation plan:

```python
def virtual_bargain(game: Game, agent_i: TGMI_Agent, agent_j: TGMI_Agent,
                    gamma: float) -> Tuple[int, int, float, float]:
    """
    Returns:
      idx_ai, idx_aj: indices in action_space_i, action_space_j
      Ui_vb, Uj_vb: utilities at VB joint action
    """
    A_i = game.action_space_i
    A_j = game.action_space_j
    ps_i = agent_i.partners[agent_j.id]
    ps_j = agent_j.partners[agent_i.id]
    best_value = -np.inf
    best = None

    for idx_i, ai in enumerate(A_i):
        for idx_j, aj in enumerate(A_j):
            # fairness vector at this joint action
            F_vec = np.array([
                game.F["max_sum"][idx_i, idx_j],
                game.F["equal_split"][idx_i, idx_j],
                game.F["rawls"][idx_i, idx_j],
            ])

            Ui = agent_i.moral_utility(agent_j.id, F_vec)
            Uj = agent_j.moral_utility(agent_i.id, F_vec)  # note symmetry in Fϕ(a)

            gain_i = max(Ui - ps_i.d, 0.0)
            gain_j = max(Uj - ps_j.d, 0.0)

            value = (gain_i ** gamma) * (gain_j ** (1.0 - gamma))
            if value > best_value:
                best_value = value
                best = (idx_i, idx_j, Ui, Uj)

    return best  # (idx_ai, idx_aj, Ui_vb, Uj_vb)
```

This directly implements the Nash-product-style VB step described in the paper.

### 2.5 Action noise εₐ (optional at first)

From the MGG description: each game includes independent action error εₐ.

Implementation:

* After choosing `(idx_ai, idx_aj)` as VB actions, with probability εₐ, replace the action index for each agent independently by a random index.

In early debugging, set εₐ = 0 to keep things deterministic.

### 2.6 Fairness-only utility Uᶠᵢ and deviation

From Algorithm 1:

[
U^F_i(a_i, a_j) = \sum_\phi B_i(\phi) F_\phi(a_i, a_j)
]

Deviations:

[
d_i^{(t)} = \max_{a_j'} U^F_i(a_i^{VB}, a_j') - U^F_i(a_i^{VB}, a_j^{VB})
]

Implementation:

```python
def fairness_utility(agent_i: TGMI_Agent, game: Game, idx_ai: int, idx_aj: int) -> float:
    F_vec = np.array([
        game.F["max_sum"][idx_ai, idx_aj],
        game.F["equal_split"][idx_ai, idx_aj],
        game.F["rawls"][idx_ai, idx_aj],
    ])
    return float(np.dot(agent_i.B, F_vec))

def fairness_deviation(agent_i: TGMI_Agent, agent_j: TGMI_Agent,
                       game: Game, idx_ai_vb: int, idx_aj_vb: int) -> Tuple[float, float]:
    """Returns (d_i_t, U_F_i_vb)"""
    A_j = game.action_space_j
    B_i = agent_i.B

    # fairness under realized (VB) joint action
    UF_vb = fairness_utility(agent_i, game, idx_ai_vb, idx_aj_vb)

    # find best possible fairness i could have gotten from j
    best_UF = -np.inf
    for idx_aj_prime, aj_prime in enumerate(A_j):
        UF_candidate = fairness_utility(agent_i, game, idx_ai_vb, idx_aj_prime)
        if UF_candidate > best_UF:
            best_UF = UF_candidate

    d_i_t = best_UF - UF_vb  # fairness shortfall caused by j
    return d_i_t, UF_vb
```

This matches the description: `d_i` measures fairness shortfall caused by the partner’s action given i’s VB action.

### 2.7 Trust update

From the text: trust evolves via a leaky integrator of compliance.

1. Compliance:

[
s_i^{(t)} = \exp(-\lambda_{\text{dev}} d_i^{(t)})
]

2. Trust update:

[
\tau_i^{(t+1)} = (1-\eta)\tau_i^{(t)} + \eta s_i^{(t)}
]

Implementation:

```python
def update_trust(agent_i: TGMI_Agent, partner_id: int, d_i_t: float):
    ps = agent_i.partners[partner_id]
    lam = agent_i.hyper.lambda_dev
    eta = agent_i.hyper.eta

    s_i_t = np.exp(-lam * d_i_t)
    ps.tau = (1.0 - eta) * ps.tau + eta * s_i_t
```

Clamp to [0,1] if needed for numerical stability.

### 2.8 CK-ToM belief update

From Eq. 4:

[
\hat B_{i\to j}^{(t+1)}(\phi) \propto
\hat B_{i\to j}^{(t)}(\phi)
\left[\exp(\beta F_\phi(a^{(t)}))\right]^{1-\alpha\tau_i^{(t)}}
\left[B_i(\phi)\right]^{\alpha\tau_i^{(t)}}
]

Implementation:

```python
def update_belief(agent_i: TGMI_Agent, partner_id: int,
                  game: Game, idx_ai_vb: int, idx_aj_vb: int):
    ps = agent_i.partners[partner_id]
    beta = agent_i.hyper.beta
    alpha = agent_i.hyper.alpha

    F_vec = np.array([
        game.F["max_sum"][idx_ai_vb, idx_aj_vb],
        game.F["equal_split"][idx_ai_vb, idx_aj_vb],
        game.F["rawls"][idx_ai_vb, idx_aj_vb],
    ])

    old_B_hat = ps.B_hat
    tau = ps.tau

    # exponent for fairness evidence
    w_evidence = 1.0 - alpha * tau
    # exponent for self-anchoring
    w_prior = alpha * tau

    unnorm = old_B_hat * np.exp(beta * F_vec) ** w_evidence * (agent_i.B ** w_prior)
    new_B_hat = unnorm / (unnorm.sum() + 1e-12)

    ps.B_hat = new_B_hat
```

This matches the CK-ToM update in the paper: high trust ⇒ lean more on fairness evidence; low trust ⇒ revert toward self prior.

### 2.9 Confidence and κ update

From the text:

[
c_i^{(t+1)} = 1 - \frac{H(\hat B_{i\to j}^{(t+1)})}{\log |F|},\quad
\kappa_i^{(t+1)} = \tau_i^{(t+1)} c_i^{(t+1)}
]

Implementation:

```python
def update_confidence_and_kappa(agent_i: TGMI_Agent, partner_id: int):
    ps = agent_i.partners[partner_id]
    B_hat = ps.B_hat
    num_phi = len(B_hat)

    H = -np.sum(B_hat * np.log(B_hat + 1e-12))
    ps.c = 1.0 - H / np.log(num_phi)
    ps.kappa = ps.tau * ps.c
```

### 2.10 Reservation utility update

From Algorithm 1: reservation utility is updated from fairness-only utility at VB joint action.

[
d_i^{(t+1)} = U^F_i(a_i^{VB}, a_j^{VB})
]

Implementation:

```python
def update_reservation_utility(agent_i: TGMI_Agent, partner_id: int,
                               UF_i_vb: float):
    ps = agent_i.partners[partner_id]
    ps.d = UF_i_vb
```

### 2.11 One round of interaction between two TGMI agents

Glue all the above into a `step` function:

```python
def tgmi_round(game: Game, agent_i: TGMI_Agent, agent_j: TGMI_Agent,
               hyperparams):
    # 1. Ensure partner states exist
    if agent_j.id not in agent_i.partners:
        agent_i._init_partner_state(agent_j.id)
    if agent_i.id not in agent_j.partners:
        agent_j._init_partner_state(agent_i.id)

    # 2. Virtual bargaining joint action
    idx_ai_vb, idx_aj_vb, Ui_vb, Uj_vb = virtual_bargain(game, agent_i, agent_j, hyperparams.gamma)

    # 3. (Optional) Apply action noise ε_a to get realized actions
    idx_ai_real, idx_aj_real = maybe_apply_action_noise(idx_ai_vb, idx_aj_vb, game, hyperparams.epsilon_a)

    # 4. Get realized payoffs
    Ri = game.R_i[idx_ai_real, idx_aj_real]
    Rj = game.R_j[idx_ai_real, idx_aj_real]

    # 5. Fairness utilities and deviations
    d_i_t, UF_i_vb = fairness_deviation(agent_i, agent_j, game, idx_ai_vb, idx_aj_vb)
    d_j_t, UF_j_vb = fairness_deviation(agent_j, agent_i, game, idx_aj_vb, idx_ai_vb)

    # 6. Update trust
    update_trust(agent_i, agent_j.id, d_i_t)
    update_trust(agent_j, agent_i.id, d_j_t)

    # 7. Update beliefs
    update_belief(agent_i, agent_j.id, game, idx_ai_vb, idx_aj_vb)
    update_belief(agent_j, agent_i.id, game, idx_aj_vb, idx_ai_vb)

    # 8. Update confidence & κ
    update_confidence_and_kappa(agent_i, agent_j.id)
    update_confidence_and_kappa(agent_j, agent_i.id)

    # 9. Update reservation utilities
    update_reservation_utility(agent_i, agent_j.id, UF_i_vb)
    update_reservation_utility(agent_j, agent_i.id, UF_j_vb)

    # 10. Logging (next section)
    log_round(agent_i, agent_j, game, ...)  # we'll define below

    return Ri, Rj, UF_i_vb, UF_j_vb
```

At this point you have a working TGMI loop as described in Algorithm 1.

---

## 3. Logging

From the Methods: they log, per iteration:

{τᵢᵗ, cᵢᵗ, κᵢᵗ, dᵢᵗ, Uᵢ(aᵛᵇ), Uᶠᵢ(aᵛᵇ), Rᵢ(aᵛᵇ), B̂ᵢ→ⱼᵗ}.

Implement:

```python
def log_round(agent_i: TGMI_Agent, agent_j: TGMI_Agent, game: Game,
              t: int,
              idx_ai_vb: int, idx_aj_vb: int,
              Ri: float, Rj: float,
              UF_i_vb: float, UF_j_vb: float,
              Ui_vb: float, Uj_vb: float,
              d_i_t: float, d_j_t: float):
    ps_i = agent_i.partners[agent_j.id]
    ps_j = agent_j.partners[agent_i.id]

    agent_i.log.append({
        "round": t,
        "partner": agent_j.id,
        "tau": ps_i.tau,
        "c": ps_i.c,
        "kappa": ps_i.kappa,
        "d": ps_i.d,
        "Ri": Ri,
        "UF": UF_i_vb,
        "U": Ui_vb,
        "d_t": d_i_t,
        "B_hat": ps_i.B_hat.copy(),
        "archetype": game.archetype.name,
        "ai_idx": idx_ai_vb,
        "aj_idx": idx_aj_vb,
    })

    agent_j.log.append({
        "round": t,
        "partner": agent_i.id,
        "tau": ps_j.tau,
        "c": ps_j.c,
        "kappa": ps_j.kappa,
        "d": ps_j.d,
        "Rj": Rj,
        "UF": UF_j_vb,
        "U": Uj_vb,
        "d_t": d_j_t,
        "B_hat": ps_j.B_hat.copy(),
        "archetype": game.archetype.name,
        "ai_idx": idx_aj_vb,
        "aj_idx": idx_ai_vb,
    })
```

Later you can dump `agent.log` to pandas DataFrame for analysis.

**Sanity checks**:

* Plot τ over time for different partner types (fair vs selfish). Should look like Fig. 3A: trust rises with fair partner, decays with exploitative one.
* Plot entropy of B̂ over time → confidence increasing.

---

## 4. Moran evolutionary process

Once MGG + TGMI are stable, wrap them in a population-level process. Follow the paper’s Moran setup.

### 4.1 Represent population

```python
class Population:
    def __init__(self, agents: List[BaseAgent], types: List[str]):
        self.agents = agents    # list of TGMI, Selfish, etc.
        self.types = types      # parallel list of type labels
```

You’ll have multiple agent classes:

* `TGMI_Agent` (you already built)
* `SelfishAgent`, `AltruisticAgent`, `TFTAgent`, `WSLSAgent`, etc., with simple policies.

### 4.2 Intragenerational rollout

Per generation:

```python
def play_generation(pop: Population, mgg: MoralGameGenerator, T: int,
                    omega: float, rng) -> Dict[int, Dict[str, float]]:
    """
    Returns per-agent stats {
      agent_id: {"R_total": ..., "UF_total": ..., "Phi": ...}
    }
    """
    # initialize per-agent accumulators
    stats = {agent.id: {"R_total": 0.0, "UF_total": 0.0} for agent in pop.agents}

    for t in range(T):
        # choose pairs (i,j); simplest: random pairing
        pairs = sample_pairs(pop.agents, rng)

        for (agent_i, agent_j) in pairs:
            game = mgg.sample_game()

            if isinstance(agent_i, TGMI_Agent) and isinstance(agent_j, TGMI_Agent):
                Ri, Rj, UF_i, UF_j = tgmi_round(game, agent_i, agent_j, agent_i.hyper)
            else:
                # handle mixed interactions: TGMI vs baseline, baseline vs baseline
                Ri, Rj, UF_i, UF_j = play_round_mixed(game, agent_i, agent_j, ...)

            stats[agent_i.id]["R_total"] += Ri
            stats[agent_i.id]["UF_total"] += UF_i
            stats[agent_j.id]["R_total"] += Rj
            stats[agent_j.id]["UF_total"] += UF_j

    # compute Φ_i for each agent (fairness-weighted return):contentReference[oaicite:27]{index=27}
    for agent in pop.agents:
        R_avg = stats[agent.id]["R_total"] / T
        UF_avg = stats[agent.id]["UF_total"] / T
        Phi = (1.0 - omega) * R_avg + omega * UF_avg
        stats[agent.id]["Phi"] = Phi

    return stats
```

### 4.3 Expected returns per type and selection

From Methods: they average over S replicates to estimate Φ̅ₖ(n) and use a softmax with selection strength s.

Implementation:

```python
def estimate_returns_by_type(pop, mgg, T, omega, S, rng):
    # collect Φ for each type
    type_to_phis = defaultdict(list)
    for s in range(S):
        # you may need to deep-copy the population, or reset each agent’s internal state
        stats = play_generation(pop, mgg, T, omega, rng)
        for agent, t_label in zip(pop.agents, pop.types):
            type_to_phis[t_label].append(stats[agent.id]["Phi"])

    Phi_bar = {t: np.mean(vals) for t, vals in type_to_phis.items()}
    return Phi_bar
```

Then selection probabilities:

[
\pi_k(n) = \frac{\exp(s \bar{\Phi}_k(n))}{\sum_m \exp(s \bar{\Phi}_m(n))}.
]

```python
def selection_probabilities(Phi_bar, selection_strength_s):
    types = list(Phi_bar.keys())
    phis = np.array([Phi_bar[t] for t in types])
    logits = selection_strength_s * phis
    probs = np.exp(logits - logits.max())
    probs = probs / probs.sum()
    return dict(zip(types, probs))
```

### 4.4 Moran update

They define a Moran process over population compositions n with mutation μ.

For implementation simplicity, you can do:

```python
def moran_step(pop: Population, pi_type: Dict[str, float], rng, mu: float):
    """
    One Moran update:
    - pick a random learner (to be replaced)
    - pick type to copy according to pi_type
    - with prob mu, mutate to a random type instead of copying
    - rebuild the population object
    """
    # 1. Choose learner index
    learner_idx = rng.integers(len(pop.agents))

    # 2. Choose type to copy or mutate
    if rng.random() < mu:
        new_type = rng.choice(list(pi_type.keys()))
    else:
        types = list(pi_type.keys())
        probs = np.array([pi_type[t] for t in types])
        new_type = rng.choice(types, p=probs)

    # 3. Replace learner with new agent of type new_type (re-initialize its moral prior etc.)
    pop.types[learner_idx] = new_type
    pop.agents[learner_idx] = make_new_agent_of_type(new_type, rng)
```

Then run this for many generations and track frequency of each type.

If you want the *exact* stationary distribution x*, you’d need to build the full transition matrix over compositions and compute its dominant left eigenvector as described in the paper—but for a first implementation, long-run Monte Carlo is fine.

---

## 5. Concrete build order (checklist)

**Phase 1 – MGG**

* [ ] Implement payoff archetype functions (even if as placeholders).
* [ ] Implement `MoralGameGenerator.sample_game`.
* [ ] Write tests for payoff and fairness ranges.
* [ ] Visualize 1–2 sampled games per archetype.

**Phase 2 – TGMI intragenerational loop**

* [ ] Implement `TGMI_Agent` and `PartnerState`.
* [ ] Implement moral utility, fairness-only utility.
* [ ] Implement VB (`virtual_bargain`).
* [ ] Implement fairness deviation + trust update.
* [ ] Implement belief update + confidence + κ update.
* [ ] Write a 2-agent script that runs T=30 rounds with fixed priors and prints τ, B̂ evolution.

**Phase 3 – Logging and debugging**

* [ ] Add logging of τ, c, κ, d, U, Uᶠ, R, B̂ each round.
* [ ] Inspect logs / plots: trust should go up with fair partner, down with selfish.
* [ ] Introduce action noise εₐ and check robustness.

**Phase 4 – Baselines and Moran**

* [ ] Implement simple baseline agents (Selfish, Altruistic, TFT, WSLS, etc.).
* [ ] Implement `play_generation` and `estimate_returns_by_type`.
* [ ] Implement Moran step and run many generations, track type frequencies.
* [ ] Experiment with ω=0 (payoff-only selection) vs ω>0 (fairness-weighted).
