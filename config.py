"""
Configuration and hyperparameters for TGMI implementation.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class Hyperparameters:
    """TGMI hyperparameters from the paper."""
    
    # Trust and belief update parameters
    eta: float = 0.1              # Trust learning rate (leaky integrator)
    lambda_dev: float = 5.0       # Deviation sensitivity parameter
    alpha: float = 0.5            # Self-anchoring weight in belief update
    beta: float = 3.0             # Fairness evidence strength
    
    # Virtual bargaining
    gamma: float = 0.5            # Nash bargaining power (symmetric by default)
    
    # Initial values
    tau0: float = 0.5             # Initial trust
    
    # Action noise
    epsilon_a: float = 0.0        # Action error probability (start with 0 for debugging)
    
    # Evolutionary parameters
    omega: float = 0.5            # Fairness weight in fitness (0=payoff-only, 1=fairness-only)
    selection_strength: float = 1.0  # Selection strength s in Moran process
    mutation_rate: float = 0.01   # Mutation rate μ
    
    # Simulation parameters
    T: int = 30                   # Rounds per generation
    S: int = 10                   # Replicates for return estimation
    population_size: int = 50     # Population size for Moran process
    
    # MGG parameters
    num_actions: int = 21         # Discretization of action space
    num_principles: int = 3       # Number of fairness principles


# Default hyperparameters
DEFAULT_HYPERPARAMS = Hyperparameters()
