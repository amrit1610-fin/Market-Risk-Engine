import numpy as np
import pandas as pd
import logging
from arch import arch_model
from .base import BaseVaRModel

logger = logging.getLogger(__name__)

class GarchHistoricalVaR(BaseVaRModel):
    """
    Calculates VaR and ES using Filtered Historical Simulation (FHS).
    Uses a GARCH(1,1) model to dynamically scale historical returns based on current market volatility.
    """
    
    def calculate(self, returns: pd.DataFrame, weights: np.ndarray) -> dict:
        logger.info("Calculating GARCH(1,1) Filtered Historical VaR/ES...")

        portfolio_returns = returns.dot(weights)                          # historical portfolio returns
        port_returns_scaled = portfolio_returns * 100
        
        # Define and fit the GARCH(1,1) model
        am = arch_model(port_returns_scaled, mean='Zero', vol='GARCH', p=1, q=1)
        res = am.fit(disp='off') 
        
        historical_volatility = res.conditional_volatility / 100          # Historical conditional volatility
        standardized_returns = portfolio_returns / historical_volatility  # Standardize the historical returns (De-volatilize)
        
        # Forecast tomorrow's volatility
        forecast = res.forecast(horizon=1)
        forecasted_volatility = np.sqrt(forecast.variance.iloc[-1, 0]) / 100
        
        logger.info(f"Forecasted tomorrow's portfolio volatility: {forecasted_volatility:.4f}")
        
        # Re-scale returns using tomorrow's forecasted volatility
        filtered_returns = standardized_returns * forecasted_volatility
        
        # Sort and extract VaR / ES (Standard Historical Simulation math)
        sorted_returns = np.sort(filtered_returns)
        percentile = (1 - self.alpha) * 100
        
        var_pct = np.percentile(sorted_returns, percentile, method='linear')
        
        tail_losses = sorted_returns[sorted_returns <= var_pct]
        es_pct = np.mean(tail_losses)
        
        var_value = -var_pct * self.portfolio_value
        es_value = -es_pct * self.portfolio_value
        
        logger.info(f"GARCH FHS complete. VaR: {var_value:.4f}, ES: {es_value:.4f}")
        
        return {
            "VaR": var_value,
            "ES": es_value
        }