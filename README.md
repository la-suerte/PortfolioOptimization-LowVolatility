SVD Alpha Generation (Work in Progress)
⚠️ Status: Experimental / Under Construction

This repository explores the use of Singular Value Decomposition (SVD) as a method for generating alpha in equity markets. The core objective is to isolate latent factors in stock price movements to predict future returns.

While the results in the research phase show promise (high Sharpe ratios relative to benchmarks), this project is currently a Proof of Concept. It is intended to test the mathematical viability of SVD in this context rather than serve as a deployment-ready trading engine.

📂 Project Structure
The repository is divided into two main environments:

1. testCode/ (The Lab)
Content: Research scripts, grid search implementations, and rapid prototyping.

State: The code here is "messy" and optimized for speed of iteration rather than readability.

Performance: This is where the core breakthroughs are located. Through extensive grid search and parameter tuning, strategies in this folder have achieved Sharpe ratios up to +0.4 above the benchmark.

Methodology: rigorous testing using non-overlapping train/test windows to ensure statistical validity during the research phase.

2. productionCode/ (The Clean Build)
Content: A structured, step-by-step Jupyter Notebook architecture.

State: Cleaner, modular, and commented.

Goal: This represents the effort to refactor the chaotic findings of testCode into a reproducible and readable pipeline.

⚙️ Methodology & Implementation
The strategy utilizes SVD to decompose price return matrices into singular vectors (eigenportfolios) to identify noise vs. signal.

Optimization: We utilize Grid Search to determine the optimal hyperparameters for the model.

Validation: Strict adherence to time-series cross-validation rules, ensuring distinct training and testing windows to prevent look-ahead bias.

🛑 Limitations & Simplifications
Please be aware of the following constraints when interpreting the results of this backtest:

1. Survivorship Bias (Data Source)
This project currently scrapes data using yfinance (Yahoo Finance).

The Problem: Yahoo Finance generally does not maintain historical data for delisted companies (e.g., Lehman Brothers, Enron).

The Impact: The backtest results are likely artificially inflated because the universe only consists of companies that "survived" to the present day. This introduces a significant positive bias.

2. SVD Matrix Truncation
Due to computational constraints and the noise-heavy nature of financial data, the SVD matrix size was arbitrarily capped.

Current Setting: The rank is limited to 150.

Implication: We may be discarding useful information or retaining too much noise. This parameter requires further theoretical optimization.

🚀 Future Improvements (Ouverture)
To move this from a Proof of Concept to a robust trading strategy, the following steps are necessary:

Professional Data Integration: Replacing yfinance with a paid data provider (e.g., Bloomberg Terminal or FactSet) to access the "Graveyard" of delisted stocks. This is the only way to accurately remove survivorship bias and stress-test the strategy against market failures.

Dynamic Rank Selection: Moving away from the fixed 150 matrix size to a dynamic selection method (e.g., analyzing the cumulative energy of singular values) to adapt to changing market volatility.

Transaction Costs: Integrating realistic slippage and commission fees into the productionCode backtest.

Disclaimer
This code is for educational and research purposes only. Do not trade real capital based on these signals without rigorous out-of-sample testing and risk management.
