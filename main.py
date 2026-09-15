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
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
    return logging.getLogger(__name__)


def main():
    # ========== LOGGING SETUP ============
    parser = argparse.ArgumentParser(description="Market Risk VaR Engine")
    parser.add_argument('--debug', action='store_true', help="Enable detailed INFO logging")
    args = parser.parse_args()

    logger = setup_logging(args.debug)
    if not args.debug:
        print("Running in clean mode (Logs muted). Use --debug to see detailed execution logs.\n")

    # ========== ENGINE SETUP ==============
    # 1. Define Portfolio and Parameters
    tickers = ['AAPL', 'MSFT', 'JPM', 'XOM']
    weights = np.array([0.40, 0.30, 0.20, 0.10])
    portfolio_value = 1_000_000  # $1M Portfolio
    
    start_date = '2024-01-01'
    end_date = '2026-01-01'
    window_size = 250 

    # 2. Ingest and Clean Data
    loader = MarketDataLoader(tickers, start_date, end_date)
    raw_prices = loader.fetch_data()
    print(f"* Data Loaded for {tickers} from {start_date} to {end_date}")
    
    preprocessor = DataPreprocessor()
    returns = preprocessor.clean_and_calculate_returns(raw_prices)
    print(f"* Log returns calculated for loaded data\n")
    
    # 3. Initialize Models and Backtester
    models = {
        "Historical Simulation": BaseHistoricalVaR(portfolio_value=portfolio_value),
        "GARCH(1,1) Historical": GarchHistoricalVaR(portfolio_value=portfolio_value),
        "Parametric (Ledoit-Wolf)": ParametricVaR(portfolio_value=portfolio_value),
        "Monte Carlo (Cholesky)": MonteCarloVaR(portfolio_value=portfolio_value, num_simulations=2000),
        "Extreme Value Theory (POT)": BaseEVTVaR(portfolio_value=portfolio_value),
        "Conditional EVT (GARCH)": GarchEVTVaR(portfolio_value=portfolio_value)
    }
    
    backtester = Backtester(confidence_level=0.99)
    
    var_predictions = {name: [] for name in models.keys()}
    actual_losses = []
    
    print(f"* Starting rolling backtest over {len(returns) - window_size} days...")
    
    # 4. The Rolling Window Backtest Loop
    for i in range(window_size, len(returns)):
        window_returns = returns.iloc[i - window_size : i]
        
        actual_daily_return = returns.iloc[i].dot(weights)
        actual_loss = -actual_daily_return * portfolio_value
        actual_losses.append(actual_loss)
        
        for name, model in models.items():
            logging.getLogger(model.__module__).setLevel(logging.WARNING)
            result = model.calculate(window_returns, weights)
            var_predictions[name].append(result["VaR"])
            
    # 5. Calculate T+1 Risk Forecast (By Desk and Total Book)
    print("\n" + "="*85)
    print(f" T+1 DAILY VaR/ES REPORT | Confidence: 99.0%")
    print("="*85)
    
    current_window = returns.iloc[-window_size:]
    
    # Define Desk Allocations (Indices map to: AAPL=0, MSFT=1, JPM=2, XOM=3)
    desks = {
        "Tech Desk": {"cols": [0, 1], "value": 700_000},
        "Macro Desk": {"cols": [2, 3], "value": 300_000},
        "Total Book": {"cols": [0, 1, 2, 3], "value": 1_000_000}
    }
    
    forecast_data = []
    
    for desk_name, desk_info in desks.items():
        # Isolate the data and weights for this specific desk
        desk_cols = desk_info["cols"]
        desk_value = desk_info["value"]
        
        desk_returns = current_window.iloc[:, desk_cols]
        
        # Extract original weights and re-normalize them so they sum to 1.0 for the desk
        raw_weights = weights[desk_cols]
        desk_weights = raw_weights / raw_weights.sum() 
        
        # We'll use the Conditional EVT (GARCH) model as the official desk reporting metric
        model = models["Conditional EVT (GARCH)"]
        
        # Temporarily update the model's portfolio value for this calculation
        model.portfolio_value = desk_value
        logging.getLogger(model.__module__).setLevel(logging.WARNING)
        
        res = model.calculate(desk_returns, desk_weights)
        
        forecast_data.append({
            "Portfolio": desk_name,
            "Allocation": f"${desk_value:,.0f}",
            "VaR (99%)": f"${res['VaR']:,.2f}",
            "Expected Shortfall": f"${res['ES']:,.2f}"
        })
        
    # Reset the model's portfolio value back to 1M for the SVaR calculations later
    models["Conditional EVT (GARCH)"].portfolio_value = portfolio_value 

    desk_df = pd.DataFrame(forecast_data).set_index("Portfolio")
    print(desk_df.to_string())

    # 6. Evaluate Backtest Results
    actual_losses_arr = np.array(actual_losses)
    expected_breaks = round(len(actual_losses_arr) * backtester.expected_failure_rate, 2)
    
    print("\n" + "="*85)
    print(f" MARKET RISK BACKTESTING REPORT | Window: {window_size} Days | Confidence: 99.0%")
    print(f" Total Days Analyzed: {len(actual_losses_arr)} | Expected Breaches: {expected_breaks}")
    print("="*85)
    
    summary_data = []
    logging.getLogger(backtester.__module__).setLevel(logging.WARNING)
    
    for name in models.keys():
        preds = np.array(var_predictions[name])
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

    # 7. Stressed VaR (SVaR) - 2008 Lehman Crisis
    print("* Loading 2008 Lehman Crisis data for Stressed VaR (SVaR)...")
    
    # Fetch exact 2008 data
    loader_2008 = MarketDataLoader(tickers, start_date='2008-01-01', end_date='2008-12-31')
    raw_prices_2008 = loader_2008.fetch_data()
    crisis_returns = preprocessor.clean_and_calculate_returns(raw_prices_2008)

    print("\n" + "="*85)
    print(f" STRESSED VaR (SVaR) | 2008 Lehman Crisis | Confidence: 99.0%")
    print("="*85)

    svar_data = []
    for name, model in models.items():
        # Mute logs for clean terminal output
        logging.getLogger(model.__module__).setLevel(logging.WARNING)
        
        # Calculate risk using 2008 data, but CURRENT portfolio weights
        res = model.calculate(crisis_returns, weights)
        svar_data.append({
            "Model": name,
            "SVaR (99%)": f"${res['VaR']:,.2f}",
            "Stressed ES": f"${res['ES']:,.2f}"
        })

    svar_df = pd.DataFrame(svar_data).set_index("Model")
    print(svar_df.to_string())
    print("="*85 + "\n")

    print(f"!!SUCCESS: VAR/ES engine for Market Risk executed !!\n")

if __name__ == "__main__":
    main()