import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Cleans raw price data and transforms it into log returns.
    """
    @staticmethod
    def clean_and_calculate_returns(raw_prices: pd.DataFrame) -> pd.DataFrame:
        logger.info("Starting data preprocessing...")
        
        # fill missing data 
        # forward-fill because we assume the last traded price remains the valid valuation.
        prices_clean = raw_prices.ffill()
        
        # Backward-fill any remaining NaNs at the very beginning of the series
        prices_clean = prices_clean.bfill()
        
        # Calculate Log Returns: ln(P_t / P_{t-1})
        log_returns = np.log(prices_clean / prices_clean.shift(1))
        log_returns = log_returns.dropna(how='all')
        
        logger.info(f"Preprocessing complete. Final returns matrix shape: {log_returns.shape}")
        return log_returns