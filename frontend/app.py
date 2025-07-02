from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.impute import SimpleImputer
from joblib import dump, load
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from datetime import datetime, timedelta
import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import warnings

app = Flask(__name__)
warnings.filterwarnings('ignore')

# Configuration
DATA_PATH = os.path.join('data', 'processed (1).csv')
N_JOBS = -1  # Use all cores

# Global variables
df = None
best_xgb = None
numeric_features = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos',
                   'temperature', 'humidity', 'windSpeed', 'pressure',
                   'demand_lag1', 'demand_rolling_24h']
categorical_features = ['is_weekend', 'season', 'weather_severity']
target = 'demand'
cluster_features = ['demand', 'temperature', 'humidity', 'windSpeed', 'pressure']

def load_data():
    global df
    df = pd.read_csv(DATA_PATH)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    return df

def create_features(df):
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    return df

def temporal_train_test_split(df, test_date):
    train = df[df.index < test_date]
    test = df[df.index >= test_date].copy()
    return train, test

def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'model': model_name}

def plot_to_base64(plt):
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf8')

def get_preprocessor():
    return ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

def train_xgb_model(X_train, y_train):
    preprocessor = get_preprocessor()

    xgb = Pipeline([
        ('preprocessor', preprocessor),
        ('xgbregressor', XGBRegressor(tree_method='hist', n_jobs=N_JOBS))
    ])

    param_grid = {
        'xgbregressor__n_estimators': [50, 100],
        'xgbregressor__max_depth': [3, 6],
        'xgbregressor__learning_rate': [0.01, 0.1]
    }

    grid_search = GridSearchCV(
        xgb,
        param_grid,
        cv=TimeSeriesSplit(n_splits=3),
        scoring='neg_mean_absolute_error',
        n_jobs=N_JOBS,
        verbose=1
    )

    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def perform_pca_visualization(data):
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[cluster_features])
    
    # Perform PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)
    
    # Create visualization colored by city
    plt.figure(figsize=(12, 6))
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], 
                         c=data['city'].astype('category').cat.codes, 
                         cmap='viridis', alpha=0.6)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA Projection (Colored by City)')
    plt.colorbar(scatter, label='City Code')
    plt.grid(True)
    
    pca_plot = plot_to_base64(plt)
    plt.close()
    
    return pca_plot


def perform_clustering(data, n_clusters=5):
    # Normalize data
    normalized_data = normalize(data[cluster_features])
    
    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(normalized_data)
    
    # Calculate evaluation metrics
    metrics = {}
    if len(np.unique(cluster_labels)) > 1:
        metrics['silhouette'] = silhouette_score(normalized_data, cluster_labels, metric='cosine')
    metrics['calinski_harabasz'] = calinski_harabasz_score(normalized_data, cluster_labels)
    metrics['davies_bouldin'] = davies_bouldin_score(normalized_data, cluster_labels)
    
    # Perform PCA for visualization
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(normalized_data)
    
    # Create cluster visualization
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(f'Cluster Visualization (k={n_clusters})')
    plt.colorbar(scatter, label='Cluster')
    plt.grid(True)
    
    cluster_plot = plot_to_base64(plt)
    plt.close()
    
    return {
        'plot': cluster_plot,
        'metrics': metrics,
        'cluster_sizes': pd.Series(cluster_labels).value_counts().to_dict()
    }

def initialize():
    global df, best_xgb
    df = load_data()
    df = create_features(df)

    # Train initial model with default parameters
    TEST_DATE = str(df.index.max() - pd.Timedelta(days=30))
    train, test = temporal_train_test_split(df, TEST_DATE)
    X_train, y_train = train[numeric_features + categorical_features], train[target]

    best_xgb = train_xgb_model(X_train, y_train)

# Initialize the app when it starts up
with app.app_context():
    initialize()

@app.route('/', methods=['GET', 'POST'])
def index():
    cities = df['city'].unique().tolist() if df is not None else []

    if request.method == 'POST':
        city = request.form.get('city', 'all')
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')
        k_clusters = int(request.form.get('k_clusters', 5))
        
        # Filter data based on selections
        filtered_df = df.copy()
        if city != 'all':
            filtered_df = filtered_df[filtered_df['city'] == city]
        
        if start_date and end_date:
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
            filtered_df = filtered_df[(filtered_df.index >= start_date) & (filtered_df.index <= end_date)]
        else:
            # Default to last 30 days if no dates selected
            end_date = filtered_df.index.max()
            start_date = end_date - pd.Timedelta(days=30)
            filtered_df = filtered_df[(filtered_df.index >= start_date) & (filtered_df.index <= end_date)]
        
        # Prepare data for models
        X = filtered_df[numeric_features + categorical_features]
        y = filtered_df[target]
        
        # Naive model predictions
        naive_pred = filtered_df['demand_lag1'].values
        
        # XGBoost predictions
        if best_xgb is not None:
            xgb_pred = best_xgb.predict(X)
        else:
            xgb_pred = naive_pred  # Fallback if model not trained
        
        # Create forecast visualization
        plt.figure(figsize=(15, 7))
        plt.plot(filtered_df.index, y, label='Actual', color='black', alpha=0.7)
        plt.plot(filtered_df.index, naive_pred, label='Naive', linestyle='--')
        plt.plot(filtered_df.index, xgb_pred, label='XGBoost', alpha=0.7)
        plt.title(f'Hourly Demand Forecast Comparison for {city.capitalize() if city != "all" else "All Cities"}')
        plt.xlabel('Date')
        plt.ylabel('Demand')
        plt.legend()
        plt.grid()
        plot_url = plot_to_base64(plt)
        plt.close()
        
        # Feature importance plot
        feature_plot_url = None
        if best_xgb is not None:
            try:
                preprocessor = get_preprocessor()
                preprocessor.fit(X)
                
                # Get feature names after preprocessing
                numeric_feature_names = numeric_features
                categorical_feature_names = []
                if 'cat' in preprocessor.named_transformers_:
                    if hasattr(preprocessor.named_transformers_['cat'], 'named_steps'):
                        if 'onehot' in preprocessor.named_transformers_['cat'].named_steps:
                            categorical_feature_names = list(
                                preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
                            )
                
                feature_names = numeric_feature_names + categorical_feature_names
                
                # Get feature importances
                importances = best_xgb.named_steps['xgbregressor'].feature_importances_
                
                # Ensure lengths match
                if len(importances) == len(feature_names):
                    plt.figure(figsize=(12, 6))
                    feat_importances = pd.Series(importances, index=feature_names)
                    feat_importances.nlargest(15).plot(kind='barh')
                    plt.title('Top 15 Feature Importances from XGBoost')
                    plt.xlabel('Importance Score')
                    plt.tight_layout()
                    feature_plot_url = plot_to_base64(plt)
                    plt.close()
            except Exception as e:
                print(f"Error creating feature importance plot: {e}")
        
        # PCA visualization
        pca_plot_url = perform_pca_visualization(filtered_df)
        
        # Cluster visualization
        clustering_results = perform_clustering(filtered_df, k_clusters)
        
        # Calculate metrics
        naive_metrics = evaluate_model(y, naive_pred, "Naive")
        xgb_metrics = evaluate_model(y, xgb_pred, "XGBoost")
        
        return render_template('index.html', 
                            cities=cities,
                            selected_city=city,
                            start_date=start_date.strftime('%Y-%m-%d'),
                            end_date=end_date.strftime('%Y-%m-%d'),
                            k_clusters=k_clusters,
                            plot_url=plot_url,
                            feature_plot_url=feature_plot_url,
                            pca_plot_url=pca_plot_url,
                            cluster_plot_url=clustering_results['plot'],
                            cluster_metrics=clustering_results['metrics'],
                            cluster_sizes=clustering_results['cluster_sizes'],
                            naive_metrics=naive_metrics,
                            xgb_metrics=xgb_metrics)

    # Default view (no form submission)
    end_date = df.index.max() if df is not None else datetime.now()
    start_date = end_date - pd.Timedelta(days=30)

    return render_template('index.html',
                         cities=cities,
                         selected_city='all',
                         start_date=start_date.strftime('%Y-%m-%d'),
                         end_date=end_date.strftime('%Y-%m-%d'),
                         k_clusters=5)

if __name__ == '__main__':
    app.run(debug=True)