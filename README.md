# Orbital Engine — Data Analytics & Visualization

A small data analytics mini-project built with **Python, Streamlit, Pandas, Plotly, and Scikit-learn**.

## 1. Dataset

The project uses a reproducible **synthetic space-mission dataset with 10,000 records**. The domain is space-mission planning and cost analysis.

The dataset contains:
- Agency type
- Payload mass
- Orbit altitude
- Mission budget
- Fuel efficiency score
- Risk mitigation index
- Testing hours
- Crewed status
- Launch-window alignment
- Final mission cost
- Mission status

The data is generated with a fixed seed, so the same dataset can be recreated every time the app runs.

## 2. Data Cleaning

The raw dataset intentionally contains missing values so the preprocessing step can be demonstrated.

### Missing values

| Column | Missing values | Cleaning method |
| --- | ---: | --- |
| fuel_efficiency_score | 300 | Filled with the median of the available values |
| window_alignment_pct | 200 | Filled with the median of the available values |

So, **500 missing cells** are handled without deleting those rows.

The complete dataset is also checked for duplicate rows. No duplicate rows were found, so no valid records were removed.

### Model preprocessing

Before training:
- Numeric features are median-imputed and standardized.
- Categorical features are filled with the most frequent value.
- Categorical values are converted using one-hot encoding.
- The same preprocessing pipeline is used for both models.

## 3. Exploratory Data Analysis

After cleaning, the app shows:
- Final mission cost distribution
- Mission outcome distribution
- Budget vs final mission cost
- Average mission cost by agency
- Numeric correlation matrix

This helps identify patterns and relationships before model training.

## 4. Machine Learning Models

The project predicts **final mission cost** using two regression models:

1. **Linear Regression**
2. **Random Forest Regressor**

Both models use the same **80/20 train-test split** and the same preprocessing pipeline.

### Model performance

| Model | MAE | RMSE | R² |
| --- | ---: | ---: | ---: |
| Linear Regression | 10.96 | 13.91 | 0.976 |
| Random Forest | 14.25 | 17.77 | 0.961 |

### Which model is better?

**Linear Regression is the better model for this dataset.**

The reason is:

- Lower MAE: **10.96** vs **14.25**
- Lower RMSE: **13.91** vs **17.77**
- Higher R²: **0.976** vs **0.961**

MAE and RMSE measure prediction error, so lower is better. R² measures how much variation in the target is explained by the model, so higher is better.

The synthetic final-cost target was created mainly from additive relationships between the input variables. Because of that, Linear Regression fits the underlying pattern better than the Random Forest in this project.

## 5. Web Application Flow

The Streamlit app is intentionally arranged in the same order as the mini-project workflow:

1. **Actual Dataset** — view the raw records before cleaning.
2. **Data Cleaning** — see missing values, cleaning operations, and the cleaned dataset.
3. **Exploratory Analysis** — understand distributions and relationships.
4. **Model Comparison** — compare both ML models using MAE, RMSE, and R².
5. **Predict Performance** — enter a new mission profile and get cost predictions from both models.

No login or authentication is required.
