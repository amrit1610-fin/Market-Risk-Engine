import logging
import argparse
import numpy as np
import pandas as pd

from data.data_loader import MarketDataLoader
from data.preprocessing import DataPreprocessor
from models.historical import BaseHistoricalVaR, GarchHistoricalVaR
from models.parametric import ParametricVaR
from models.monte_carlo import MonteCarloVaR
from models.evt import BaseEVTVaR, GarchEVTVaR
from backtest.tests import Backtester

def setup_logging(debug_mode: bool):
    """
    Sets the logging level based on the command line argument.
    If debug_mode is False, only WARNINGs and ERRORs will print.
    """
    log_level = logging.INFO if debug_mode else logging.WARNING
    
    # Remove any existing handlers so we don't get duplicate logs
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
    return logging.getLogger(__name__)


def main():
    # ========== LOGGING DESCRIPTION ============
    # Parse Command Line Arguments
    parser = argparse.ArgumentParser(description="Market Risk VaR Engine")
    parser.add_argument('--debug', action='store_true', help="Enable detailed INFO logging")
    args = parser.parse_args()

    # Configure Logging
    logger = setup_logging(args.debug)
    if not args.debug:
        print("Running in clean mode (Logs muted). Use --debug to see detailed execution logs.\n")

    # ========== ENGINE DESCRIPTION ==============
    # 1. Define Portfolio and Parameters
    tickers = ['AAPL', 'MSFT', 'JPM', 'XOM']
    weights = np.array([0.40, 0.30, 0.20, 0.10])
    portfolio_value = 1_000_000  # $1M Portfolio
    
    # We pull 2 years of data. 
    # Year 1 (250 days) builds the first window. Year 2 (250 days) is the backtest period.
    start_date = '2024-01-01'
    end_date = '2026-01-01'
    window_size = 250 

    # 2. Ingest and Clean Data
    loader = MarketDataLoader(tickers, start_date, end_date)
    raw_prices = loader.fetch_data()
    print(f"* Data Loaded for {tickers} from {start_date} to {end_date}\n")
    
    preprocessor = DataPreprocessor()
    returns = preprocessor.clean_and_calculate_returns(raw_prices)
    print(f"* Log returns calculated for loaded data\n")
    
    # 3. Initialize Models and Backtester
    models = {
        "Historical Simulation": BaseHistoricalVaR(portfolio_value=portfolio_value),
        "GARCH(1,1) Historical Simulation": GarchHistoricalVaR(portfolio_value=portfolio_value),
        "Parametric (Ledoit-Wolf)": ParametricVaR(portfolio_value=portfolio_value),
        "Monte Carlo (Cholesky)": MonteCarloVaR(portfolio_value=portfolio_value, num_simulations=2000),      # Reduced simulations for speed in daily rolling loop
        "Extreme Value Theory (POT)": BaseEVTVaR(portfolio_value=portfolio_value),
        "Conditional EVT ": GarchEVTVaR(portfolio_value=portfolio_value)
    }
    
    backtester = Backtester(confidence_level=0.99)
    
    # Storage arrays for our rolling predictions
    var_predictions = {name: [] for name in models.keys()}
    actual_losses = []
    
    logger.info(f"Starting rolling backtest over {len(returns) - window_size} days...")

    print(f"* Starting rolling backtest over {len(returns) - window_size} days...")
    # 4. The Rolling Window Backtest Loop
    for i in range(window_size, len(returns)):
        # Isolate the trailing 250-day window
        window_returns = returns.iloc[i - window_size : i]
        
        # What actually happened on day T+1?
        actual_daily_return = returns.iloc[i].dot(weights)
        actual_loss = -actual_daily_return * portfolio_value
        actual_losses.append(actual_loss)
        
        # Have each model predict VaR for day T+1 based on the window
        for name, model in models.items():
            # Suppress individual model logs in the loop to avoid terminal spam
            logging.getLogger(model.__module__).setLevel(logging.WARNING)
            
            result = model.calculate(window_returns, weights)
            var_predictions[name].append(result["VaR"])
            
    # 5. Evaluate Results
    actual_losses_arr = np.array(actual_losses)
    expected_breaks = round(len(actual_losses_arr) * backtester.expected_failure_rate, 2)
    
    print("\n" + "="*85)
    print(f" MARKET RISK BACKTESTING REPORT | Window: {window_size} Days | Confidence: {backtester.alpha * 100}%")
    print(f" Total Days Analyzed: {len(actual_losses_arr)} | Expected Breaches: {expected_breaks}")
    print("="*85)
    
    summary_data = []
    
    for name in models.keys():
        preds = np.array(var_predictions[name])
        
        # Temporarily mute the backtester logger to keep the console clean
        logging.getLogger(backtester.__module__).setLevel(logging.WARNING)
        results = backtester.evaluate(actual_losses_arr, preds)
        
        summary_data.append({
            "Model": name,
            "Breaches": results['Actual_Breaches'],
            "Traffic Light": results['Traffic_Light'],
            "Kupiec (p-val)": f"{results['Kupiec_p_value']:.4f}",
            "Christoff (p-val)": f"{results['Christoffersen_p_value']:.4f}"
        })
        
    report_df = pd.DataFrame(summary_data).set_index("Model")
    print(report_df.to_string())
    print("="*85 + "\n")

    print(f"\n!!SUCCESS: VAR/ES engine for Market Risk executed !!")

if __name__ == "__main__":
    main()