# TGMI — Implementation Change Instructions (Equations Only)
*(for updating the implementation to match the **updated likelihood** + **restless mixture** belief update, before editing the PDF)*

This note assumes your current implementation matches the **current paper** (old CK-ToM belief update). Below are **equation-level changes only**—no code.

---

## 1) What stays the same (no change)

All steps **before** the belief update remain as in the current implementation/paper:

### (a) Moral evaluation
Compute trust-gated moral utility \(U_i(a_i,a_j)\) (per your Eq. 2 / existing implementation).

### (b) Virtual Bargaining (VB)
Compute the VB agreement \((a_i^{\mathrm{VB}}, a_j^{\mathrm{VB}})\) (per your Eq. 3 / existing implementation).

### (c) Fairness deviation + trust update
Keep the fairness utility and deviation computation (Fairness Eq.) and keep trust dynamics, e.g.
\[
\tau_i^{(t+1)} = (1-\eta)\,\tau_i^{(t)} + \eta\,\exp(-\lambda_{\mathrm{dev}}\, d_i^{(t)}).
\]
*(Exact symbols follow your current paper/implementation.)*

✅ **Only the belief update block (Step (d)) changes.**

---

## 2) Replace the belief-update evidence with an explicit likelihood of the **realized** action

### 2.1 New likelihood (replaces the old CK-ToM “evidence term”)
For each moral principle hypothesis \(\phi \in \mathcal{F}\), compute the likelihood of the **observed** partner action \(a_j^{(t)}\):

\[
\mathcal{L}(a_j^{(t)}\mid \phi)
=
\tau_i^{(t)}\,\frac{\exp\big(\beta\,F_{\phi}(a_j^{(t)},a_i^{(t)})\big)}{Z_{\phi}(a_i^{(t)})}
+
\big(1-\tau_i^{(t)}\big)\,\frac{1}{\lvert A_j\rvert}.
\tag{L}
\]

**Interpretation:**
- With probability \(\tau_i^{(t)}\), agent \(i\) treats partner \(j\) as informative and uses a softmax model of \(j\)’s action under principle \(\phi\).
- With probability \(1-\tau_i^{(t)}\), agent \(i\) treats the observation as uninformative and uses a uniform distribution.

### 2.2 Critical correction (the bug you found)
In the exponential term, the fairness utility must use the **realized** outcome:
\[
F_{\phi}(a_j^{(t)},a_i^{(t)})\quad \textbf{NOT}\quad F_{\phi}(a_j^{\mathrm{VB}},a_i^{\mathrm{VB}}).
\]
This is required because \(\mathcal{L}(a_j^{(t)}\mid\phi)\) is explicitly a likelihood of the **observed** action at time \(t\).

---

## 3) Fix the softmax normalizer \(Z_{\phi}\)

To ensure the likelihood is meaningful, the normalizer must sum over the **candidate partner actions**:

\[
Z_{\phi}(a_i^{(t)})
=
\sum_{a_j\in A_j}
\exp\big(\beta\,F_{\phi}(a_j,a_i^{(t)})\big).
\tag{Z}
\]

**Note:** This choice is consistent with defining the likelihood over \(a_j^{(t)}\).  
If you instead normalize over a joint action grid \(A_i\times A_j\), then the likelihood must be defined over the realized **joint** action \((a_i^{(t)},a_j^{(t)})\). The updated screenshots describe likelihood of \(a_j^{(t)}\), so the consistent implementation is the conditional form above.

---

## 4) Replace the belief update with the **restless mixture** operator

Given current belief \(\hat B_{i\to j}^{(t)}(\phi)\) and likelihood \(\mathcal{L}(a_j^{(t)}\mid\phi)\), compute the Bayes-updated belief:

\[
B^{\mathrm{Bayes}}_{i\to j}(\phi)
=
\frac{\hat B_{i\to j}^{(t)}(\phi)\,\mathcal{L}(a_j^{(t)}\mid\phi)}
{\sum_{\psi\in\mathcal{F}} \hat B_{i\to j}^{(t)}(\psi)\,\mathcal{L}(a_j^{(t)}\mid\psi)}.
\tag{B}
\]

Then apply the **restless mixture** with epistemic vigilance \(\gamma\in(0,1)\):

\[
\hat B_{i\to j}^{(t+1)}(\phi)
=
(1-\gamma)\,B^{\mathrm{Bayes}}_{i\to j}(\phi)
+
\gamma\,\frac{1}{\lvert\mathcal{F}\rvert}.
\tag{RM}
\]

**What this changes vs current implementation:**
- You are explicitly blending **Bayesian evidence** with a **uniform exploration prior**.
- This guarantees beliefs stay strictly interior to the simplex (no collapse to 0/1), improving sensitivity to later regime shifts.

---

## 5) Confidence update should be applied to the **new** belief

If your algorithm tracks confidence as normalized neg-entropy (as in the updated Algorithm screenshot), compute:

\[
c_i^{(t+1)}
=
1
-
\frac{H\big(\hat B_{i\to j}^{(t+1)}\big)}{\log\lvert\mathcal{F}\rvert},
\quad
H(p) = -\sum_{\phi\in\mathcal{F}} p(\phi)\log p(\phi).
\tag{C}
\]

**Instruction:** apply entropy to the **post-mixture** belief \(\hat B^{(t+1)}\), not the intermediate Bayes-only distribution.

---

## 6) Mapping to “Algorithm 1” steps (what to edit conceptually)

### Only Step (d) changes:

**Old Step (d):** CK-ToM belief update using an evidence term that (in the current paper/implementation) is tied to \(a^{\mathrm{VB}}\) and does not use the restless mixture.

**New Step (d):**
1. Compute \(\mathcal{L}(a_j^{(t)}\mid\phi)\) using Eqs. (L)–(Z), with **realized** \(a^{(t)}\).
2. Update beliefs using Eqs. (B)–(RM).
3. Update confidence using Eq. (C) (if you track it).

Everything else (VB computation, deviation/trust update, etc.) remains unchanged.

---

## 7) Consistency checklist (equation-level)

Before you move on to editing the PDF/LaTeX, ensure the implementation matches these consistency conditions:

1. **Realized outcome in likelihood:** \(F_{\phi}(a_j^{(t)},a_i^{(t)})\) appears in the exponent.
2. **Normalizer sums over candidate actions:** \(Z_{\phi}(a_i^{(t)}) = \sum_{a_j\in A_j} \exp(\beta F_{\phi}(a_j,a_i^{(t)}))\).
3. **Uniform term matches action space:** \(1/|A_j|\) if likelihood is over \(a_j\).
4. **Belief update is Bayes + mixture:** \((1-\gamma)\times\text{Bayes} + \gamma/|\mathcal{F}|\).
5. **Confidence uses updated belief:** entropy computed on \(\hat B^{(t+1)}\).

---

## 8) Notation reminder (to avoid accidental mismatches)

- \(a^{\mathrm{VB}}\): counterfactual coordination outcome (used for VB selection and deviation computations).
- \(a^{(t)}\): realized/observed outcome at time \(t\) (must be used in the likelihood).
- \(A_j\): discretized partner action set used for the likelihood normalization.
- \(\mathcal{F}\): finite hypothesis set of moral principles.

---

*End of instructions.*
