import yfinance as yf
import pandas as pd
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MarketDataLoader:
    """
    Handles fetching raw market data from Yahoo Finance.
    """
    def __init__(self, tickers: list, start_date: str, end_date: str):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date

    def fetch_data(self) -> pd.DataFrame:
        logger.info(f"Fetching data for {len(self.tickers)} tickers from {self.start_date} to {self.end_date}...")
        
        try:
            raw_data = yf.download(self.tickers, start=self.start_date, end=self.end_date)
            
            # Get only closing price for portfolio
            if 'Close' in raw_data:
                adj_close = raw_data['Close']
            else:
                raise ValueError("Could not find 'Close' column in downloaded data.")
            
            # If only one ticker is provided, yfinance returns a Series. Convert to DataFrame.
            if isinstance(adj_close, pd.Series):
                adj_close = adj_close.to_frame(name=self.tickers[0])
                
            logger.info("**Data fetching successful.**\n")
            return adj_close
            
        except Exception as e:
            logger.error(f"!! Failed to fetch market data: {e} !!")
            raise