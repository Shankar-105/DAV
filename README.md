# Orbital Engine — Data Analytics & Visualization

A small data analytics lab project built with **Python, Streamlit, Pandas, Plotly, and Scikit-learn**.

## Dataset

The app uses a reproducible **synthetic space-mission dataset with 10,000 records**. It contains mission details such as agency, payload, orbit altitude, budget, fuel efficiency, risk mitigation, testing hours, crewed status, launch-window alignment, final mission cost, and mission status.

A few fields intentionally contain missing values so the data-cleaning step can be demonstrated.

## Models

The app predicts **final mission cost** using two regression models with the same 80/20 train-test split:

| Model | MAE | RMSE | R² |
| --- | ---: | ---: | ---: |
| Linear Regression | 10.96 | 13.91 | 0.976 |
| Random Forest | 14.25 | 17.77 | 0.961 |

For this dataset, **Linear Regression performs better**. This is expected because the synthetic target was generated mostly from additive relationships between the input variables.

## What the app shows

- Dataset overview and interactive visualizations
- Data cleaning and missing-value analysis
- Correlation and cost analysis
- Model comparison
- A small mission-cost evaluator for new inputs

No login or authentication is required.
