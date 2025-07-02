Machine learning-powered web app to forecast hourly electricity demand across multiple cities. Integrates time series forecasting with clustering analysis to reveal demand-weather patterns and provides an interactive dashboard for visual insights.

---

## Features

- Hourly electricity demand forecasting using XGBoost
- Weather-demand pattern clustering with KMeans
- Time series feature engineering (lag, rolling, cyclical encoding)
- Interactive dashboard for input selection, forecasts, clustering results, and model performance

---

## Dataset

The dataset, sourced from Kaggle, includes:

- Datetime: Timestamp of the observation
- City: Identifier for the U.S. city
- Demand (MW): Electricity demand in megawatts
- Weather variables: Temperature, humidity, wind speed, pressure, etc.

---

## Preprocessing

- Missing Values:
  - Numerical: Imputed using the median
  - Categorical: Imputed using the most frequent value
- Feature Engineering:
  - Extracted hour, day of week, and month
  - Applied sine/cosine transformations for cyclic features (hour, month)
  - Created lag features (e.g., previous hour demand)
  - Calculated rolling statistics (mean and standard deviation)

---

## Forecasting Model

- Model: `XGBoostRegressor`
- Baseline: Naive model using previous hour demand
- Validation: `TimeSeriesSplit` (5 splits)
- Hyperparameter Tuning: `GridSearchCV` over:
  - n_estimators: [100, 200, 300]
  - max_depth: [3, 5, 7]
  - learning_rate: [0.01, 0.1, 0.2]
- Evaluation Metrics:
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)
  - MAPE (Mean Absolute Percentage Error)

---

## Clustering Component

- Normalization: `StandardScaler` applied to demand and weather features
- Algorithm: `KMeans`
- Optimal cluster count selected using:
  - Silhouette Score
  - Calinski-Harabasz Index
  - Davies-Bouldin Index
- Dimensionality Reduction: PCA for 2D visualization

---

## Web Application Architecture

- Backend: Flask
- Frontend: HTML, CSS, Bootstrap
- Data Handling: Pandas, NumPy
- Visualization: Matplotlib, Seaborn, Base64 for plot transfer
- Machine Learning: Scikit-learn, XGBoost

---

## Dashboard Visualizations

- Actual vs. Predicted demand (XGBoost and Naive)
- Clustering (PCA 2D plot)
- Feature distribution across clusters
- Metric summaries for both forecasting and clustering
- Interactive set-up with filter adjustments
