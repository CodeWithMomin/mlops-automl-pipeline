import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, List
from src import config

def calculate_psi(expected: np.ndarray, actual: np.ndarray, num_bins: int = 10) -> float:
    """
    Calculates the Population Stability Index (PSI) between two distributions.
    
    PSI = sum((Actual_i - Expected_i) * ln(Actual_i / Expected_i))
    """
    # Remove NaNs
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
        
    # Get bin edges based on the expected dataset (percentiles)
    percentiles = np.linspace(0, 100, num_bins + 1)
    # Ensure bin edges are unique to prevent pandas.cut errors
    bin_edges = np.percentile(expected, percentiles)
    bin_edges = np.unique(bin_edges)
    
    # If we have only 1 unique value, return 0.0 (no distribution to compare)
    if len(bin_edges) <= 1:
        return 0.0
        
    # Adjust boundaries slightly to include min/max
    bin_edges[0] -= 1e-5
    bin_edges[-1] += 1e-5
    
    # Calculate counts in each bin
    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)
    
    # Convert to proportions (fractions)
    expected_pcts = expected_counts / len(expected)
    actual_pcts = actual_counts / len(actual)
    
    # Handle zero counts using Laplace-style smoothing
    eps = 1e-4
    expected_pcts = np.where(expected_pcts == 0, eps, expected_pcts)
    actual_pcts = np.where(actual_pcts == 0, eps, actual_pcts)
    
    # Normalize again to ensure they sum to 1
    expected_pcts /= np.sum(expected_pcts)
    actual_pcts /= np.sum(actual_pcts)
    
    # Calculate PSI
    psi_value = np.sum((actual_pcts - expected_pcts) * np.log(actual_pcts / expected_pcts))
    return float(psi_value)

class DriftDetector:
    def __init__(self, reference_data: pd.DataFrame):
        """
        Initializes with reference (training) data.
        """
        self.reference_data = reference_data.copy()
        # Exclude target if present
        if "target" in self.reference_data.columns:
            self.reference_data = self.reference_data.drop(columns=["target"])
        
        self.features = list(self.reference_data.columns)
        self.categorical_features = [
            col for col in self.features 
            if self.reference_data[col].nunique() <= 5 or self.reference_data[col].dtype in ['object', 'category']
        ]
        self.continuous_features = [col for col in self.features if col not in self.categorical_features]

    def check_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Compares current (inference) dataset against reference dataset.
        Computes KS-Test for continuous features and PSI for all features.
        
        Returns a detailed report.
        """
        current_df = current_data.copy()
        if "target" in current_df.columns:
            current_df = current_df.drop(columns=["target"])
            
        # Ensure we are comparing matching columns
        cols_to_check = [col for col in self.features if col in current_df.columns]
        
        feature_reports = {}
        drifted_features = []
        
        for col in cols_to_check:
            ref_vals = self.reference_data[col].values
            curr_vals = current_df[col].values
            
            # 1. Compute PSI (for all columns)
            # Use fewer bins if categorical or binary
            num_bins = 2 if col in self.categorical_features else 10
            psi = calculate_psi(ref_vals, curr_vals, num_bins=num_bins)
            
            # 2. Compute KS-Test (only for continuous columns)
            ks_p_value = None
            ks_drift_flag = False
            
            if col in self.continuous_features:
                # Kolmogorov-Smirnov test
                ks_stat, p_val = stats.ks_2samp(ref_vals, curr_vals)
                ks_p_value = float(p_val)
                # If p-value is lower than threshold, distribution is significantly different
                if ks_p_value < config.DRIFT_THRESHOLD_KS:
                    ks_drift_flag = True
            
            # 3. Determine if this feature has drifted
            # Drift is flagged if PSI is high OR KS test detects significant shift
            psi_drift_flag = psi >= config.DRIFT_THRESHOLD_PSI
            feature_drift_flag = psi_drift_flag or ks_drift_flag
            
            if feature_drift_flag:
                drifted_features.append(col)
                
            feature_reports[col] = {
                "psi": psi,
                "psi_drift": bool(psi_drift_flag),
                "ks_p_value": ks_p_value,
                "ks_drift": bool(ks_drift_flag),
                "drift_detected": bool(feature_drift_flag)
            }
            
        # System-level drift status
        # If > 30% of features drift, or average PSI is high, flag overall drift
        total_features = len(cols_to_check)
        drift_fraction = len(drifted_features) / total_features if total_features > 0 else 0.0
        
        avg_psi = np.mean([report["psi"] for report in feature_reports.values()]) if feature_reports else 0.0
        
        overall_drift = drift_fraction >= 0.3 or avg_psi >= config.DRIFT_THRESHOLD_PSI
        
        return {
            "overall_drift": bool(overall_drift),
            "drift_fraction": float(drift_fraction),
            "average_psi": float(avg_psi),
            "drifted_features": drifted_features,
            "feature_reports": feature_reports,
            "reference_sample_count": len(self.reference_data),
            "current_sample_count": len(current_df)
        }
