# SVD Alpha Generation Project

> **⚠️ Status: Experimental / Work in Progress**
> This project is a Proof of Concept (PoC) designed to test Singular Value Decomposition (SVD) as a method for generating alpha in equity markets.

---

## 📖 Overview
This repository explores the mathematical viability of using SVD to isolate latent factors in stock price movements. The goal is to separate signal from noise to predict future returns.

While the results in the research phase show promise, this code is intended for **research and testing purposes** rather than immediate deployment in a live trading environment.

---

## 📂 Project Structure

The repository is divided into two distinct environments:

### 1. `testCode/` (The Lab)
* **State:** Experimental, messy, optimized for speed.
* **Purpose:** Rapid prototyping and heavy parameter tuning.
* **Performance:** This folder contains the strongest research results. Through extensive **Grid Search** and testing, strategies here have achieved **Sharpe ratios up to +0.4** above the benchmark.
* **Methodology:** Utilizes rigorous non-overlapping train/test windows to ensure statistical validity.

### 2. `productionCode/` (The Clean Build)
* **State:** Structured, modular, and documented.
* **Purpose:** Refactoring the chaotic findings of the `testCode` folder into a proper, cell-by-cell Jupyter Notebook architecture.
* **Goal:** To create a reproducible pipeline based on the best parameters found in the lab.

---

## ⚙️ Methodology
The core strategy involves decomposing price return matrices to identify eigenportfolios.

* **Grid Search:** Used extensively to find the best hyperparameters.
* **Backtesting Protocols:** We strictly observe standard good practices, such as using distinct, non-overlapping training and testing windows to prevent data leakage.

---

## 🛑 Limitations & Simplifications

**Please read this section carefully.** As this is a research project, several simplifications were made that affect the realism of the results.

### 1. Survivorship Bias (Data Source)
* **Source:** The project currently uses `yfinance` (Yahoo Finance).
* **Issue:** Yahoo Finance generally removes delisted companies. Consequently, this dataset does not include failed companies (e.g., Lehman Brothers, Enron).
* **Impact:** The backtest results are likely positively biased, as the algorithm is selecting from a universe of stocks that we know "survived" to the present day.

### 2. SVD Matrix Truncation
* **Constraint:** Due to the computational cost of SVD and the noise inherent in large financial datasets, the matrix size is limited.
* **Simplification:** The rank was arbitrarily set to **150**.
* **Impact:** This fixed number acts as a hard cap on the dimensionality of the model.

---

## 🚀 Future Improvements (Ouverture)

To advance this project from a PoC to a deployable strategy, the following steps are required:

1.  **Professional Data Integration:** Connect to institutional data sources (e.g., **Bloomberg** or **FactSet**) to include delisted equities. This is critical to removing survivorship bias and validating the strategy's true robustness.
2.  **Dynamic Rank Selection:** Replace the fixed matrix size (150) with a dynamic threshold based on the cumulative energy of singular values.
3.  **Transaction Costs:** Implement realistic fee structures and slippage modeling in the `productionCode` backtest.

---

## ⚖️ Disclaimer
*This code is provided for educational and research purposes only. The high Sharpe ratios observed in the test environment may not translate to live markets due to the limitations listed above.*
