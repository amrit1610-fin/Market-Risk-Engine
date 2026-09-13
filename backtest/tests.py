import numpy as np
import pandas as pd
import logging
from scipy.stats import chi2

logger = logging.getLogger(__name__)

class Backtester:
    """
    Evaluates VaR models using the Kupiec POF test, Christoffersen Independence test,
    and the Basel Committee Traffic Light framework.
    """
    def __init__(self, confidence_level: float = 0.99):
        self.alpha = confidence_level
        self.expected_failure_rate = 1 - self.alpha

    def evaluate(self, actual_losses: np.ndarray, var_predictions: np.ndarray) -> dict:
        """
        Runs all backtests on a time series of losses and VaR thresholds.
        """
        logger.info("Running backtests (Kupiec, Christoffersen, Traffic Light)...")
        
        # 1. Identify Breaches (Exceptions)
        # Boolean array where True means actual loss exceeded predicted VaR
        breaches = actual_losses > var_predictions
        
        N = len(breaches)
        x = breaches.sum()
        actual_failure_rate = x / N if N > 0 else 0
        
        logger.info(f"Analyzed {N} days. Found {x} breaches (Expected: {N * self.expected_failure_rate:.2f}).")
        
        # 2. Basel Traffic Light System (Standardized for a 250-day window at 99% VaR)
        # Note: If N != 250, these thresholds theoretically need scaling, but Basel uses 250 hardcoded.
        if x <= 4:
            traffic_light = "Green"
        elif 5 <= x <= 9:
            traffic_light = "Amber"
        else:
            traffic_light = "Red"
            
        # 3. Kupiec Proportion of Failures (POF) Test
        # Null Hypothesis: Actual failure rate == Expected failure rate
        p = self.expected_failure_rate
        p_hat = actual_failure_rate
        
        if x == 0:
            # Handle edge case where there are zero breaches
            lr_kupiec = -2 * np.log(((1 - p)**N))
        else:
            # Log-Likelihood Ratio
            numerator = ((1 - p)**(N - x)) * (p**x)
            denominator = ((1 - p_hat)**(N - x)) * (p_hat**x)
            lr_kupiec = -2 * np.log(numerator / denominator)
            
        # p-value from Chi-square distribution with 1 degree of freedom
        p_val_kupiec = 1 - chi2.cdf(lr_kupiec, 1)
        
        # 4. Christoffersen Interval Forecast (Independence) Test
        # Null Hypothesis: Breaches are independent (no volatility clustering)
        # T00: No breach followed by no breach. T01: No breach -> Breach.
        # T10: Breach -> No breach. T11: Breach -> Breach.
        T00 = T01 = T10 = T11 = 0
        for i in range(1, N):
            if not breaches[i-1] and not breaches[i]: T00 += 1
            elif not breaches[i-1] and breaches[i]: T01 += 1
            elif breaches[i-1] and not breaches[i]: T10 += 1
            elif breaches[i-1] and breaches[i]: T11 += 1

        pi_0 = T01 / (T00 + T01) if (T00 + T01) > 0 else 0
        pi_1 = T11 / (T10 + T11) if (T10 + T11) > 0 else 0
        pi = (T01 + T11) / (T00 + T01 + T10 + T11)
        
        # Likelihood for independence
        ln_L_ind = (T00 + T10) * np.log(1 - pi + 1e-10) + (T01 + T11) * np.log(pi + 1e-10)
        # Likelihood for dependence (Markov chain)
        ln_L_dep = (T00 * np.log(1 - pi_0 + 1e-10) + T01 * np.log(pi_0 + 1e-10) + 
                    T10 * np.log(1 - pi_1 + 1e-10) + T11 * np.log(pi_1 + 1e-10))
                    
        lr_christ = -2 * (ln_L_ind - ln_L_dep)
        p_val_christ = 1 - chi2.cdf(lr_christ, 1)
        
        logger.info(f"Traffic Light: {traffic_light}")
        logger.info(f"Kupiec p-value: {p_val_kupiec:.4f}")
        logger.info(f"Christoffersen p-value: {p_val_christ:.4f}")
        
        return {
            "Total_Days": N,
            "Expected_Breaches": round(N * self.expected_failure_rate, 2),
            "Actual_Breaches": x,
            "Traffic_Light": traffic_light,
            "Kupiec_p_value": p_val_kupiec,
            "Christoffersen_p_value": p_val_christ
        }