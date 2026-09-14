import numpy as np
import pandas as pd
import logging
from scipy.stats import t
from .base import BaseVaRModel

logger = logging.getLogger(__name__)

class MonteCarloVaR(BaseVaRModel):
    """
    Calculates VaR and ES using Monte Carlo Simulation with Geometric Brownian Motion (GBM)
    and Cholesky decomposition for asset correlation.
    """
    def __init__(self, confidence_level: float = 0.99, portfolio_value: float = 1.0,
                 num_simulations: int = 10000, dof: int = 5):
        super().__init__(confidence_level, portfolio_value)
        self.num_simulations = num_simulations
        self.dof = dof
        np.random.seed(42) 

    def calculate(self, returns: pd.DataFrame, weights: np.ndarray) -> dict:
        logger.info(f"Running Monte Carlo VaR with {self.num_simulations} simulations...")
        
        num_assets = len(weights)

        # parameters for GBM
        mean_returns = returns.mean().values
        variances = returns.var().values
        drift = mean_returns - (0.5 * variances)                         # Ito's Lemma correction: The drift of log returns is (mu - 0.5 * sigma^2)
        cov_matrix = returns.cov().values                                # Covariance and performing Cholesky Decomposition
        
        try:
            L = np.linalg.cholesky(cov_matrix)                            # L is the lower triangular matrix such that L * L^T = Covariance Matrix
        except np.linalg.LinAlgError:
            logger.warning("Covariance matrix is not positive definite. Falling back to Ledoit-Wolf shrinkage.")
            from sklearn.covariance import LedoitWolf
            cov_matrix = LedoitWolf().fit(returns).covariance_
            L = np.linalg.cholesky(cov_matrix)
            
        # Generating the underlying process using Student's t-statistics
        raw_t_shocks = t.rvs(df=self.dof, size=(self.num_simulations, num_assets))
        scaling_factor = np.sqrt((self.dof - 2) / self.dof)
        standardized_t_shocks = raw_t_shocks * scaling_factor               # Standardize Variance
        correlated_shocks = standardized_t_shocks @ L.T                                         # Correlating the Shocks
        simulated_asset_returns = drift + correlated_shocks                 # Simulating 1-Day Log Returns
        simulated_port_returns = simulated_asset_returns @ weights          # Simulated Portfolio Returns
        
        # Sort and extract VaR / ES 
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