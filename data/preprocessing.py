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
        
        # forward-fill because we assume the last traded price remains the valid valuation.
        clean_prices = raw_prices.ffill()
        
        # Backward-fill any remaining NaNs at the very beginning of the series
        clean_prices = clean_prices.bfill()
        
        # Calculate Log Returns
        log_returns = np.log(clean_prices / clean_prices.shift(1))
        log_returns = log_returns.dropna(how='all')
        logger.info("Calculated Log returns...")
        
        logger.info(f"**Preprocessing complete. Final returns matrix shape: {log_returns.shape}**\n")
        return log_returns