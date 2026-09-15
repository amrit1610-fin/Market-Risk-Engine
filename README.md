<div align="center">

# Enterprise Market Risk Engine 

</div>

A professional-grade, object-oriented Market Risk engine built in Python. This project calculates Value at Risk (VaR) and Expected Shortfall (ES) for a multi-asset portfolio, strictly adhering to Basel Committee regulatory standards including backtesting and historical stress testing (SVaR).

## 📊 Core Capabilities

* **Model Progression Benchmarking:** Evaluates portfolios across multiple mathematical frameworks simultaneously to isolate the value of different statistical assumptions.
* **Dynamic Volatility Filtering:** Implements GARCH(1,1) Filtered Historical Simulation (FHS) to adapt instantly to volatility clustering and market shocks.
* **Extreme Value Theory (EVT):** Uses Peak Over Threshold (POT) with a Generalized Pareto Distribution (GPD) to accurately price fat-tailed black swan events. 
* **Dynamic Tail Optimization:** Eliminates hardcoded POT thresholds by utilizing a Kolmogorov-Smirnov (KS) Goodness-of-Fit grid search with a 21-day caching system for computational efficiency.
* **Advanced Stochastic Simulation:** Upgraded standard Monte Carlo Geometric Brownian Motion (GBM) by injecting Multivariate Student-t shocks (5 DoF) applied over a Cholesky-decomposed correlation matrix (with Ledoit-Wolf shrinkage fallback).

## 🏛️ Regulatory Compliance (Basel III/IV)

* **Traffic Light System:** Automatically grades models into Green, Amber, or Red zones based on Basel penalty zones.
* **Statistical Backtesting:** Runs a rolling 250-day window utilizing Kupiec's POF (Proportion of Failures) and Christoffersen's Independence tests to penalize Markov-chain breach clustering.
* **Stressed VaR (SVaR):** Seamlessly executes data-override stress tests, re-valuing the current portfolio against the 2008 Lehman Brothers collapse.
* **Desk-Level Granularity:** Disaggregates risk limits by trading desk, accurately capturing cross-asset diversification benefits.

## 🚀 Execution & Output

```bash
python main.py

```

*Sample T+1 Risk Output (Breach Analysis):*

```text
=====================================================================================
 T+1 DAILY VaR/ES REPORT | Confidence: 99.0%
=====================================================================================
            Allocation   VaR (99%) Expected Shortfall
Portfolio                                            
Tech Desk     $700,000  $18,051.58         $37,829.11
Macro Desk    $300,000   $7,065.05          $7,978.57
Total Book  $1,000,000  $21,408.45         $32,949.00

```

## 🏗️ Architecture

The codebase relies on a modular OOP design, making it highly extensible for new asset classes or models:

* `BaseVaRModel`: Abstract base class handling confidence intervals and portfolio normalization.
* `historical.py`: Unweighted and GARCH-filtered empirical simulations.
* `evt.py`: Unconditional EVT and Conditional FHS-EVT hybrid models.
* `monte_carlo.py`: Fat-tailed stochastic path generation.
* `parametric.py`: Variance-Covariance matrix analytics.

## 🛠️ Tech Stack

* **Python 3.x**
* **Pandas / NumPy:** Vectorized matrix operations and data wrangling.
* **SciPy:** Statistical distributions (Student-t, GenPareto) and KS-testing.
* **Arch:** GARCH(1,1) conditional volatility forecasting.
* **Scikit-Learn:** Ledoit-Wolf matrix shrinkage for illiquid covariance handling.

