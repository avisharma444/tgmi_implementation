# TGMI Implementation

Implementation of the **Trust-Gated Moral Inference (TGMI)** algorithm from *"A Computational Theory of Cooperation under Moral Uncertainty"*.

## Project Structure

```
pnas_paper_implementation/
├── README.md
├── requirements.txt
├── tgmi/                      # TGMI Agent Implementation
│   ├── agent.py               # Core TGMI agent (Algorithm 1)
│   └── experiments.py         # Experiments and plot generation
├── mgg/                       # Moral Game Generator
│   ├── generator.py           # Generates games with fairness matrices
│   └── config.py              # Game configuration
├── outputs/
│   └── plots/                 # Generated plots saved here
└── docs/                      # Reference papers
```

### Key Files

| File | Description |
|------|-------------|
| `tgmi/agent.py` | TGMI agent with trust-gated belief updates, virtual bargaining (Nash), 3 fairness principles |
| `tgmi/experiments.py` | Batch experiments, BR agent comparison, trust dynamics visualization |
| `mgg/generator.py` | Moral Game Generator - samples games across different archetypes |

## How to Run

### 1. Create and activate virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run experiments

```bash
cd tgmi
python experiments.py
```

### Output

The script generates 3 plots in `outputs/plots/`:

- **tgmi_comparison.png** - Compares TGMI vs TGMI (same/different priors) and TGMI vs BR agent
- **trust_vs_defector.png** - Trust dynamics when facing an always-defect partner
- **trust_vs_deceptive.png** - Trust dynamics when facing a deceptive agent (cooperates then defects)
