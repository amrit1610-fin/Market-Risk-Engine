import numpy as np
import pandas as pd
import logging
from scipy.stats import norm
from sklearn.covariance import LedoitWolf
from .base import BaseVaRModel

logger = logging.getLogger(__name__)

class ParametricVaR(BaseVaRModel):
    """
    Calculates VaR and ES analytically using the Variance-Covariance method,
    enhanced with Ledoit-Wolf shrinkage for the covariance matrix.
    """
    
    def calculate(self, returns: pd.DataFrame, weights: np.ndarray) -> dict:
        logger.info("Calculating Parametric VaR/ES with Ledoit-Wolf Shrinkage...")
        
        # Step 1: Calculate the mean return of each asset and the portfolio
        mean_returns = returns.mean()
        port_mean = np.dot(weights, mean_returns)
        
        # Step 2: Calculate the Covariance Matrix using Ledoit-Wolf Shrinkage
        # We use scikit-learn to automatically find the optimal shrinkage intensity (delta)
        lw = LedoitWolf()
        fitted_lw = lw.fit(returns)
        cov_matrix = fitted_lw.covariance_
        shrinkage_penalty = fitted_lw.shrinkage_
        
        logger.info(f"Ledoit-Wolf optimal shrinkage coefficient applied: {shrinkage_penalty:.4f}")
        
        # Step 3: Calculate Portfolio Variance and Volatility (Standard Deviation)
        # Formula: Variance = weights^T * CovarianceMatrix * weights
        port_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        port_vol = np.sqrt(port_variance)
        
        # Step 4: Calculate Parametric VaR
        # Find the z-score for the lower tail (e.g., -2.326 for 99%)
        z_score = norm.ppf(1 - self.alpha)
        
        # VaR cutoff return
        var_pct = port_mean + (z_score * port_vol)
        
        # Step 5: Calculate Parametric Expected Shortfall (ES)
        # For a normal distribution, ES has an exact closed-form analytical solution:
        # ES_pct = mu - sigma * (PDF(z) / (1 - alpha))
        pdf_at_z = norm.pdf(z_score)
        es_pct = port_mean - port_vol * (pdf_at_z / (1 - self.alpha))
        
        # Step 6: Convert to dollar values (positive numbers for loss reporting)
        var_value = -var_pct * self.portfolio_value
        es_value = -es_pct * self.portfolio_value
        
        logger.info(f"Parametric (LW) complete. VaR: {var_value:.4f}, ES: {es_value:.4f}")
        
        return {
            "VaR": var_value,
            "ES": es_value
        }