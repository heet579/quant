# Quantitative Finance Analysis

End-to-end equity research: exploratory analysis, cross-sectional factors, XGBoost return prediction, long/short backtesting, and risk metrics (VaR, drawdown, stress tests).

**Universe:** 35 large-cap U.S. equities (7 GICS sectors) · **Benchmark:** SPY · **Period:** 2019–2024  
**Model split:** train 2019–2022 · validate 2023 · test 2024 (no lookahead)

---

## Contents

| Section | Topics |
|---------|--------|
| 1. Data | `yfinance` prices, daily/monthly returns |
| 2. EDA | Cumulative returns, distributions, rolling vol, correlation, sector heatmap |
| 3. Factors | 12-1 / 6-1 momentum, reversal, low vol · IC / ICIR · quintile L/S |
| 4. ML | Feature panel · XGBoost · time-series split · SHAP (optional) |
| 5. Portfolio | ML quintile long/short · 10 bps transaction costs · performance vs SPY/EW |
| 6. Risk | Historical / parametric / MC VaR & CVaR · drawdown · macro stress scenarios |
| 7. Summary | Executive dashboard chart |

---

## Prerequisites

- Python **3.10+**
- Internet access (live data via `yfinance`)

---

## Setup

```bash
cd quant
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

python -m pip install -r requirements.txt
```

---

## Run the notebook

1. Open `quant_analysis.ipynb` in Jupyter Lab or VS Code.
2. Run all cells (first run downloads prices; allow ~1–2 minutes).
3. Charts are saved as `output_01_*.png` … `output_14_*.png`.

```bash
jupyter notebook quant_analysis.ipynb
```

---

## Interactive dashboard

```bash
python -m streamlit run app.py
```

Opens at `http://localhost:8501` — tabs for overview, EDA, factors, ML model, portfolio, and risk.

### Deploy on Streamlit Community Cloud

1. Sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub.
2. **Create app** → **From existing repo** → `heet579/quant`
3. Branch: `main` · Main file: `app.py`

---

## Project layout

```
quant/
├── quant_analysis.ipynb
├── analysis.py
├── app.py
├── requirements.txt
├── packages.txt
└── README.md
```

---

## Notes

Research use only — not investment advice. Results depend on `yfinance` data, a fixed universe, and simplified transaction costs (10 bps one-way).

---

## Stack

Python · yfinance · XGBoost · SHAP · NumPy · Pandas · SciPy · scikit-learn · Matplotlib · Seaborn · Streamlit · Plotly
