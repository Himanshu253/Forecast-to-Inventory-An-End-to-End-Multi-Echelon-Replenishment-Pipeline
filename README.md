# Forecast-to-Inventory-An-End-to-End-Multi-Echelon-Replenishment-Pipeline
This project bridges the gap between predictive machine learning and prescriptive operations research. It integrates a leak-free XGBoost multi-quantile demand forecaster with a multi-echelon Mixed-Integer Linear Programming (MILP) replenishment strategy. Instead of standard ML error metrics (RMSE), the pipeline is evaluated through a custom rolling-origin simulator to measure true business impact: service-level fill rates and total inventory costs.

---

## 📑 Table of Contents
- [The Core Philosophy](#-the-core-philosophy)
- [Pipeline Architecture](#-pipeline-architecture)
- [Key Features](#-key-features)
- [Business Outcomes](#-business-outcomes)
- [Visual Insights](#-visual-insights)
- [Installation & Usage](#-installation--usage)

---

##  The Core Philosophy: Why Not RMSE?
Most public data science portfolios stop at building a forecaster and reporting statistical errors like `RMSE` or `MAPE`. This project takes the next step into **Prescriptive Analytics**. 

In retail, uncalibrated high-quantile forecasts can silently break a replenishment policy, leading to severe under-stocking. Here, every candidate model feeds the *same* replenishment policy, the plan is executed against *actual* demand in a custom simulator, and models are compared strictly on **service level (fill rate) and financial cost**.

---

##  Pipeline Architecture

The project executes a decision-consistent rolling-origin backtest simulating a realistic monthly model-refresh production cadence:

 Raw Retail Panel
* Feature Engineering (Leak-free, target = forward L+R day demand sum).
* Quantile Forecaster (XGBoost GPU, multi-quantile reg:quantileerror) 
* Optimization Policy (Newsvendor order-up-to OR Multi-echelon MILP)
* Custom Simulator (Real demand, real lead times, DC allocation under scarcity)
* Business Metrics (Fill rate %, Total Cost INR, Inventory value)


---

## ✨ Key Features

* **Aggregated Forward-Sum Target:** Instead of forecasting daily demand and summing it, the model forecasts aggregated demand directly over the *protection interval* (Lead Time + Review Period).
* **Multi-Quantile XGBoost:** Fits all quantiles (0.50 to 0.99) simultaneously in a single GPU-accelerated model, preventing crossing quantile curves.
* **Multi-Echelon MILP Optimization:** A rolling-horizon Mixed-Integer Linear Programming formulation (solved via `PuLP`/`HiGHS`) that balances distribution center (DC) and store echelons.
* **Realistic Simulation Engine:** Simulates lost sales, physical transit lead times, and proportional fair-share DC allocation when network inventory runs dry.

---

## 📊 Business Outcomes

The pipeline was backtested over a **336-day simulation window** covering a scope of **40 SKUs across 6 stores**.

By evaluating the optimization layer against a baseline buffer-stock policy on the exact same forecaster, the integration of advanced ML and operations research yielded the following:

| Forecaster | Policy | Avg Fill Rate | Total Cost (INR) | Avg Inventory Value (INR) |
| --- | --- | --- | --- | --- |
| **XGB Quantile** | **Newsvendor** | **99.77%** | **₹1,737,005** | ₹5,596,691 |
| XGB Quantile | Buffer (Baseline) | 99.44% | ₹1,775,594 | ₹5,288,508 |
| XGB Quantile | MILP (Constrained) | 97.51% | ₹2,673,510 | ₹5,442,510 |

> **🏆 The Headline:** The Newsvendor optimization reduced total network costs by **₹38,589** over 336 days while improving fill rates by 0.32 percentage points compared to baseline buffer rules.

---

## 📈 Visual Insights



* **Left:** The Service-Inventory frontier demonstrating the tradeoff between capital tied up in inventory and the resulting fill rate. Up and to the left is optimal.
* **Right:** Cost decomposition breaking down lost margin, holding costs, and ordering costs per policy.

---

## 🛠️ Installation & Usage

### 1. Dependencies

Ensure you have the following libraries installed:

```bash
pip install pandas numpy xgboost pulp highspy pyarrow matplotlib

```

### 2. Running the Project

The entire pipeline is contained within `S1_forecast_to_inventory_Final.ipynb`.

**Configuration:**
All hyperparameters and scale settings live in the `Config` dataclass.

* Run a quick 60-second smoke test: Set `FAST_MODE = True`.
* Execute the full, rigorous backtest: Set `FAST_MODE = False`.

### 3. Data Source

The notebook dynamically generates an INR-denominated synthetic panel dataset replicating hard retail failure modes (intermittency, promo spikes, price elasticity). It also includes built-in support to seamlessly swap to the **M5 Kaggle Dataset** (Walmart US data) via API token.
