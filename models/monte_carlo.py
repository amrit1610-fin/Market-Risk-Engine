import numpy as np
import pandas as pd
import logging
from .base import BaseVaRModel

logger = logging.getLogger(__name__)

class MonteCarloVaR(BaseVaRModel):
    """
    Calculates VaR and ES using Monte Carlo Simulation with Geometric Brownian Motion (GBM)
    and Cholesky decomposition for asset correlation.
    """
    def __init__(self, confidence_level: float = 0.99, portfolio_value: float = 1.0, num_simulations: int = 10000):
        super().__init__(confidence_level, portfolio_value)
        self.num_simulations = num_simulations
        # Seed for reproducibility - critical when regulators audit your model
        np.random.seed(42) 

    def calculate(self, returns: pd.DataFrame, weights: np.ndarray) -> dict:
        logger.info(f"Running Monte Carlo VaR with {self.num_simulations} simulations...")
        
        num_assets = len(weights)
        
        # Step 1: Calculate parameters for GBM
        mean_returns = returns.mean().values
        variances = returns.var().values
        
        # Ito's Lemma correction: The drift of log returns is (mu - 0.5 * sigma^2)
        drift = mean_returns - (0.5 * variances)
        
        # Step 2: Calculate Covariance and perform Cholesky Decomposition
        cov_matrix = returns.cov().values
        
        try:
            # L is the lower triangular matrix such that L * L^T = Covariance Matrix
            L = np.linalg.cholesky(cov_matrix)
        except np.linalg.LinAlgError:
            logger.warning("Covariance matrix is not positive definite. Falling back to Ledoit-Wolf shrinkage.")
            from sklearn.covariance import LedoitWolf
            cov_matrix = LedoitWolf().fit(returns).covariance_
            L = np.linalg.cholesky(cov_matrix)
            
        # Step 3: Generate Independent Random Shocks
        # Z ~ N(0, 1) matrix of shape (num_simulations, num_assets)
        Z = np.random.standard_normal((self.num_simulations, num_assets))
        
        # Step 4: Correlate the Shocks
        # Matrix multiplication of independent shocks (Z) by the Cholesky matrix (L)
        # We use L.T because Z is (sims, assets) and L is (assets, assets)
        correlated_shocks = Z @ L.T
        
        # Step 5: Simulate 1-Day Log Returns
        # Formula: drift + correlated_shock
        simulated_asset_returns = drift + correlated_shocks
        
        # Step 6: Calculate Simulated Portfolio Returns
        simulated_port_returns = simulated_asset_returns @ weights
        
        # Step 7: Sort and extract VaR / ES (Identical to Historical Simulation)
        sorted_returns = np.sort(simulated_port_returns)
        percentile = (1 - self.alpha) * 100
        var_pct = np.percentile(sorted_returns, percentile, method='linear')
        
        tail_losses = sorted_returns[sorted_returns <= var_pct]
        es_pct = np.mean(tail_losses)
        
        var_value = -var_pct * self.portfolio_value
        es_value = -es_pct * self.portfolio_value
        
        logger.info(f"Monte Carlo complete. VaR: {var_value:.4f}, ES: {es_value:.4f}")
        
        return {
            "VaR": var_value,
            "ES": es_value
        }