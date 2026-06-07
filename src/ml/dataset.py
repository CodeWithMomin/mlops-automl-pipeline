import numpy as np
import pandas as pd

def generate_base_dataset(n_samples: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    Generates a synthetic credit card default dataset with continuous and categorical features.
    """
    np.random.seed(seed)
    
    # Continuous features
    age = np.random.normal(loc=38, scale=10, size=n_samples).clip(18, 80)
    income = np.random.normal(loc=60000, scale=20000, size=n_samples).clip(15000, 250000)
    credit_score = np.random.normal(loc=680, scale=70, size=n_samples).clip(300, 850)
    debt_to_income = np.random.beta(a=2, b=5, size=n_samples) * 100  # Percentage
    employment_years = np.random.exponential(scale=5, size=n_samples).clip(0, 45)
    
    # Categorical/binary features
    prior_defaults = np.random.binomial(n=1, p=0.12, size=n_samples)
    
    # Target definition (non-linear combo + noise)
    # Score formula where lower credit score, higher debt, prior default, lower income -> higher probability of default (1)
    logit = (
        -0.03 * (credit_score - 600) 
        + 0.05 * debt_to_income 
        - 0.00002 * (income - 50000) 
        + 1.5 * prior_defaults
        - 0.02 * age
        - 0.8
    )
    prob = 1 / (1 + np.exp(-logit))
    target = np.random.binomial(n=1, p=prob)
    
    df = pd.DataFrame({
        "age": age,
        "income": income,
        "credit_score": credit_score,
        "debt_to_income": debt_to_income,
        "employment_years": employment_years,
        "prior_defaults": prior_defaults,
        "target": target
    })
    
    return df

def generate_drifted_dataset(n_samples: int = 1000, shift_type: str = "covariate", seed: int = 100) -> pd.DataFrame:
    """
    Generates a drifted version of the dataset by shifting feature distributions.
    
    Types of shift:
    - 'covariate': Shifts input features (e.g. average credit score drops, debt-to-income spikes).
    - 'concept': Changes target relationship (e.g. higher default probability for same credit score).
    """
    np.random.seed(seed)
    
    # Run the base generation first
    df = generate_base_dataset(n_samples=n_samples, seed=seed)
    
    if shift_type == "covariate":
        # Simulate credit score drop (e.g., during economic downturn)
        df["credit_score"] = np.random.normal(loc=610, scale=80, size=n_samples).clip(300, 850)
        
        # Simulate income drop
        df["income"] = np.random.normal(loc=48000, scale=18000, size=n_samples).clip(10000, 200000)
        
        # Simulate debt-to-income spike
        df["debt_to_income"] = np.random.beta(a=3, b=4, size=n_samples) * 100 + 5
        
    elif shift_type == "concept":
        # Logit equation shifts, prior defaults become highly predictive, or base default rate increases
        logit = (
            -0.03 * (df["credit_score"] - 600) 
            + 0.08 * df["debt_to_income"] 
            - 0.00002 * (df["income"] - 50000) 
            + 2.8 * df["prior_defaults"]  # Increased impact of defaults
            - 0.02 * df["age"]
            + 0.5  # Increased base intercept
        )
        prob = 1 / (1 + np.exp(-logit))
        df["target"] = np.random.binomial(n=1, p=prob)
        
    return df
