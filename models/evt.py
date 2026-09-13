import numpy as np
import pandas as pd
import logging
from scipy.stats import genpareto
from .base import BaseVaRModel

logger = logging.getLogger(__name__)

class EVTVaR(BaseVaRModel):
    """
    Calculates VaR and ES using Extreme Value Theory (EVT).
    Implements the Peak Over Threshold (POT) method and fits a 
    Generalized Pareto Distribution (GPD) to the tail.
    """
    def __init__(self, confidence_level: float = 0.99, portfolio_value: float = 1.0, threshold_percentile: float = 0.95):
        super().__init__(confidence_level, portfolio_value)
        # 95th percentile is a standard threshold to isolate the top 5% of losses
        self.threshold_percentile = threshold_percentile

    def calculate(self, returns: pd.DataFrame, weights: np.ndarray) -> dict:
        logger.info("Calculating EVT (POT) VaR/ES...")
        
        # Step 1: Calculate Portfolio Returns and convert to positive Loss domain
        # EVT mathematics are standardly calculated on positive variables representing losses
        port_returns = returns.dot(weights)
        losses = -port_returns 
        
        # Step 2: Determine Threshold (u) and Extract Exceedances
        u = np.percentile(losses, self.threshold_percentile * 100)
        excess_losses = losses[losses > u] - u
        
        N = len(losses)
        Nu = len(excess_losses)
        pu = Nu / N  # Probability of exceeding the threshold
        
        logger.info(f"Threshold (u) set at {u:.4f}. Found {Nu} tail exceedances.")
        
        # Step 3: Fit the GPD to the excess losses
        # floc=0 forces the location parameter to be 0 since our excesses start exactly at 0
        # Scipy returns: shape (c), location (loc), scale (beta)
        shape, loc, scale = genpareto.fit(excess_losses, floc=0)
        
        logger.info(f"GPD Fit - Shape (xi): {shape:.4f}, Scale (beta): {scale:.4f}")
        
        # Step 4: Calculate EVT VaR
        # p is the probability mapped to the conditional tail distribution
        p = (1 - self.alpha) / pu
        
        if shape != 0:
            var_loss = u + (scale / shape) * ((p ** -shape) - 1)
        else:
            # Fallback to exponential distribution formula if shape is exactly 0
            var_loss = u - scale * np.log(p)
            
        # Step 5: Calculate EVT Expected Shortfall
        # The ES of a GPD is the VaR plus the mean excess loss beyond VaR.
        # Formula: ES = VaR + (scale + shape * (VaR - u)) / (1 - shape)
        if shape < 1:
            es_loss = var_loss + (scale + shape * (var_loss - u)) / (1 - shape)
        else:
            logger.warning("Fat Tail Warning: Shape parameter >= 1. Theoretical Expected Shortfall is infinite.")
            es_loss = np.inf
            
        # Step 6: Convert to final portfolio values
        var_value = var_loss * self.portfolio_value
        es_value = es_loss * self.portfolio_value
        
        logger.info(f"EVT complete. VaR: {var_value:.4f}, ES: {es_value:.4f}")
        
        return {
            "VaR": var_value,
            "ES": es_value
        }