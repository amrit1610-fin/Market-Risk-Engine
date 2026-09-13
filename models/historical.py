import numpy as np
import pandas as pd
import logging
from .base import BaseVaRModel

logger = logging.getLogger(__name__)

class HistoricalVaR(BaseVaRModel):
    """
    Calculates Value at Risk and Expected Shortfall using empirical historical simulation.
    """
    
    def calculate(self, returns: pd.DataFrame, weights: np.ndarray) -> dict:
        logger.info("Calculating Historical Simulation VaR/ES...")
        
        # Step 1: Calculate historical portfolio returns
        # Matrix multiplication of daily asset returns by current portfolio weights
        portfolio_returns = returns.dot(weights)
        
        # Step 2: Sort the returns from worst loss to largest gain
        sorted_returns = np.sort(portfolio_returns)
        
        # Step 3: Find the VaR threshold (Quantile)
        # If alpha is 0.99, we want the 1st percentile (1 - 0.99 = 0.01)
        percentile = (1 - self.alpha) * 100
        
        # We use interpolation in case the percentile falls between two discrete days
        var_pct = np.percentile(sorted_returns, percentile, method='linear')
        
        # Step 4: Calculate Expected Shortfall (CVaR)
        # Average of all returns that are worse than or equal to the VaR threshold
        tail_losses = sorted_returns[sorted_returns <= var_pct]
        es_pct = np.mean(tail_losses)
        
        # Step 5: Convert percentage returns to dollar/value losses
        # VaR and ES are typically reported as positive numbers representing the loss amount
        var_value = -var_pct * self.portfolio_value
        es_value = -es_pct * self.portfolio_value
        
        logger.info(f"Historical Simulation complete. VaR: {var_value:.4f}, ES: {es_value:.4f}")
        
        return {
            "VaR": var_value,
            "ES": es_value
        }