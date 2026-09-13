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
        
        # Calculate historical portfolio returns
        portfolio_returns = returns.dot(weights)
        sorted_returns = np.sort(portfolio_returns)
    
        percentile = (1 - self.alpha) * 100                                           # VaR threshold (Quantile)
        var_pct = np.percentile(sorted_returns, percentile, method='linear')          # interpolation in case the percentile falls between two discrete days
        
        # Calculate Expected Shortfall (CVaR)
        tail_losses = sorted_returns[sorted_returns <= var_pct]
        es_pct = np.mean(tail_losses)
        
        # Convert percentage returns to dollar/value losses
        var_value = -var_pct * self.portfolio_value
        es_value = -es_pct * self.portfolio_value
        
        logger.info(f"Historical Simulation complete. VaR: {var_value:.4f}, ES: {es_value:.4f}")
        
        return {
            "VaR": var_value,
            "ES": es_value
        }