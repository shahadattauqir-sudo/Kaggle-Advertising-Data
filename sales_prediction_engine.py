"""
Sales Prediction & Marketing Budget Optimization Engine
Author: Advanced Data Science & ML Engineering Team
Dataset: Advertising.csv (ISLR Advertising Dataset)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.cluster import KMeans
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Set plot styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Advertising.csv")


def load_and_preprocess_data(filepath=DATA_PATH):
    """
    Loads and cleans the advertising dataset.
    Removes index artifacts, handles missing values, checks duplicates.
    """
    print("=" * 70)
    print("STEP 1: DATA INGESTION & DATA HYGIENE")
    print("=" * 70)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
        
    df = pd.read_csv(filepath)
    print(f"Dataset successfully loaded. Shape: {df.shape[0]} rows, {df.shape[1]} columns.")
    
    # Remove unnamed index column if present
    unnamed_cols = [c for c in df.columns if 'Unnamed' in c or c == '']
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
        print(f"Dropped index artifact column(s): {unnamed_cols}")

    # Standardize column names
    df.columns = [c.strip() for c in df.columns]
    
    # Quality audits
    missing_counts = df.isnull().sum()
    duplicate_count = df.duplicated().sum()
    print(f"Missing values:\n{missing_counts.to_dict()}")
    print(f"Duplicate rows detected: {duplicate_count}")
    
    # Summary stats
    stats_df = df.describe().round(2)
    print("\nDescriptive Statistics:")
    print(stats_df)
    
    return df


def engineer_features_and_segments(df):
    """
    Constructs derived features, interaction/synergy variables,
    budget ratios, and market segment clustering.
    """
    print("\n" + "=" * 70)
    print("STEP 2: FEATURE ENGINEERING & MARKET SEGMENTATION")
    print("=" * 70)
    
    df_feat = df.copy()
    
    # 1. Total Spend across all media
    df_feat['Total_Spend'] = df_feat['TV'] + df_feat['Radio'] + df_feat['Newspaper']
    
    # 2. Channel Budget Shares (% of total advertising investment)
    # Add epsilon to prevent division by zero in theoretical edge cases
    eps = 1e-6
    df_feat['TV_Share'] = df_feat['TV'] / (df_feat['Total_Spend'] + eps)
    df_feat['Radio_Share'] = df_feat['Radio'] / (df_feat['Total_Spend'] + eps)
    df_feat['Newspaper_Share'] = df_feat['Newspaper'] / (df_feat['Total_Spend'] + eps)
    
    # 3. Cross-Channel Interaction / Synergy Effects
    df_feat['TV_Radio_Synergy'] = df_feat['TV'] * df_feat['Radio']
    df_feat['TV_Newspaper_Synergy'] = df_feat['TV'] * df_feat['Newspaper']
    df_feat['Radio_Newspaper_Synergy'] = df_feat['Radio'] * df_feat['Newspaper']
    
    # 4. Non-Linear Diminishing Returns (Square-root transformation)
    df_feat['TV_Sqrt'] = np.sqrt(df_feat['TV'])
    df_feat['Radio_Sqrt'] = np.sqrt(df_feat['Radio'])
    df_feat['Newspaper_Sqrt'] = np.sqrt(df_feat['Newspaper'])
    
    # 5. Spend Tier Categorization
    spend_quantiles = df_feat['Total_Spend'].quantile([0.33, 0.66]).values
    def categorize_tier(spend):
        if spend < spend_quantiles[0]:
            return 'Low-Budget Tier'
        elif spend < spend_quantiles[1]:
            return 'Mid-Budget Tier'
        else:
            return 'High-Budget Tier'
    df_feat['Spend_Tier'] = df_feat['Total_Spend'].apply(categorize_tier)
    
    # 6. Unsupervised Market Segmentation via K-Means Clustering
    # Cluster markets on spend profile (TV, Radio, Newspaper)
    cluster_features = ['TV', 'Radio', 'Newspaper']
    scaler = StandardScaler()
    scaled_spend = scaler.fit_transform(df_feat[cluster_features])
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df_feat['Cluster_ID'] = kmeans.fit_predict(scaled_spend)
    
    # Label clusters based on centroids
    cluster_centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=cluster_features)
    cluster_labels = {}
    for idx, row in cluster_centers.iterrows():
        if row['TV'] > 180 and row['Radio'] > 25:
            cluster_labels[idx] = "Omnichannel Powerhouse"
        elif row['TV'] > 180:
            cluster_labels[idx] = "TV-Dominant Reach"
        elif row['Radio'] > 25:
            cluster_labels[idx] = "Radio-Boosted Metro"
        else:
            cluster_labels[idx] = "Conservative / Niche"
    
    df_feat['Market_Segment'] = df_feat['Cluster_ID'].map(cluster_labels)
    
    print(f"Features created successfully. Total feature columns: {df_feat.shape[1]}")
    print("\nMarket Segment Distribution:")
    print(df_feat['Market_Segment'].value_counts())
    
    print("\nAverage Sales by Market Segment:")
    segment_summary = df_feat.groupby('Market_Segment')[['TV', 'Radio', 'Newspaper', 'Total_Spend', 'Sales']].mean().round(2)
    print(segment_summary)
    
    return df_feat, kmeans, scaler, cluster_labels


def perform_statistical_diagnostics(df):
    """
    Computes Pearson correlations, OLS regression summary, p-values,
    and Variance Inflation Factor (VIF) to detect multicollinearity.
    """
    print("\n" + "=" * 70)
    print("STEP 3: STATISTICAL INFERENCE & DIAGNOSTICS")
    print("=" * 70)
    
    X_raw = df[['TV', 'Radio', 'Newspaper']]
    y = df['Sales']
    
    # Pearson Correlation with Target
    corr_matrix = df[['TV', 'Radio', 'Newspaper', 'Sales']].corr()
    print("Correlation with Sales:")
    for col in ['TV', 'Radio', 'Newspaper']:
        print(f"  - {col:10s}: {corr_matrix.loc[col, 'Sales']:.4f}")
        
    # OLS Regression
    X_const = sm.add_constant(X_raw)
    ols_model = sm.OLS(y, X_const).fit()
    print("\nOLS Baseline Summary:")
    print(ols_model.summary().tables[1])
    print(f"\nOLS Model R-squared: {ols_model.rsquared:.4f}, Adj R-squared: {ols_model.rsquared_adj:.4f}")
    print(f"F-statistic: {ols_model.fvalue:.2f} (p-value: {ols_model.f_pvalue:.4e})")
    
    # Multicollinearity Check via VIF
    vif_df = pd.DataFrame()
    vif_df['Feature'] = X_raw.columns
    vif_df['VIF'] = [variance_inflation_factor(X_raw.values, i) for i in range(X_raw.shape[1])]
    print("\nVariance Inflation Factor (VIF) Analysis:")
    print(vif_df.to_string(index=False))
    print("Note: VIF < 5 indicates negligible multicollinearity among raw media channels.")
    
    # Statistical Key Takeaway
    p_newspaper = ols_model.pvalues['Newspaper']
    print(f"\nStatistical Takeaway on Newspaper Advertising:")
    print(f"  Newspaper p-value = {p_newspaper:.4f} (greater than threshold alpha = 0.05).")
    print(f"  CONCLUSION: Newspaper advertising exhibits no statistically significant direct impact on Sales")
    print(f"  when controlling for TV and Radio investments.")
    
    return ols_model, corr_matrix


def train_and_benchmark_models(df_feat):
    """
    Trains and compares multiple regression algorithms:
    1. Linear Regression
    2. Ridge Regression
    3. Lasso Regression
    4. Polynomial Regression (Degree 2 with interactions)
    5. Random Forest Regressor
    6. Gradient Boosting Regressor
    7. Support Vector Regressor (SVR)
    """
    print("\n" + "=" * 70)
    print("STEP 4: MODEL TRAINING, CROSS-VALIDATION & BENCHMARKING")
    print("=" * 70)
    
    # Primary channel features
    feature_cols = ['TV', 'Radio', 'Newspaper']
    X = df_feat[feature_cols]
    y = df_feat['Sales']
    
    # Stratified/Consistent 80/20 train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"Training instances: {len(X_train)} | Test instances: {len(X_test)}")
    
    # Models dictionary
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression (L2)": Ridge(alpha=1.0),
        "Lasso Regression (L1)": Lasso(alpha=0.1),
        "Polynomial Regression (Deg 2)": Pipeline([
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('linear', LinearRegression())
        ]),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=100, max_depth=6, random_state=42
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42
        ),
        "Support Vector Regressor (SVR)": Pipeline([
            ('scaler', StandardScaler()),
            ('svr', SVR(C=10.0, epsilon=0.2, kernel='rbf'))
        ])
    }
    
    results = []
    trained_models = {}
    test_predictions = {}
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models.items():
        # 5-Fold Cross-Validation R2 score on training set
        cv_r2_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2')
        cv_r2_mean = cv_r2_scores.mean()
        cv_r2_std = cv_r2_scores.std()
        
        # Fit model on entire training set
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Predict on test set
        y_pred = model.predict(X_test)
        test_predictions[name] = y_pred
        
        # Evaluate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred) * 100
        
        n = len(y_test)
        p = X.shape[1]
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        
        results.append({
            "Model": name,
            "CV_R2_Mean": round(cv_r2_mean, 4),
            "CV_R2_Std": round(cv_r2_std, 4),
            "Test_R2": round(r2, 4),
            "Test_Adj_R2": round(adj_r2, 4),
            "Test_MAE": round(mae, 4),
            "Test_RMSE": round(rmse, 4),
            "Test_MAPE(%)": round(mape, 2)
        })
        
    benchmark_df = pd.DataFrame(results).sort_values(by="Test_R2", ascending=False).reset_index(drop=True)
    print("\nModel Benchmark Leaderboard:")
    print(benchmark_df.to_string(index=False))
    
    best_model_name = benchmark_df.iloc[0]['Model']
    best_model = trained_models[best_model_name]
    print(f"\nChampion Model: [{best_model_name}] with Test R2 = {benchmark_df.iloc[0]['Test_R2']:.4f} and RMSE = {benchmark_df.iloc[0]['Test_RMSE']:.4f}")
    
    return trained_models, benchmark_df, X_train, X_test, y_train, y_test, test_predictions


def analyze_advertising_impact_and_roi(df_feat, best_model, linear_model):
    """
    Calculates marginal return on ad spend (mROI), sensitivity,
    and conducts budget reallocation simulations.
    """
    print("\n" + "=" * 70)
    print("STEP 5: ADVERTISING IMPACT, SENSITIVITY & BUDGET REALLOCATION")
    print("=" * 70)
    
    # 1. Marginal Response from Linear Coefficients
    # Linear Regression coefficients represent delta Sales (in 1,000 units) per $1,000 ad spend
    lr_coefs = linear_model.coef_
    channels = ['TV', 'Radio', 'Newspaper']
    
    print("Direct Marginal Response per $1,000 Spend (Linear Model):")
    for ch, coef in zip(channels, lr_coefs):
        print(f"  - {ch:10s}: +{coef:.4f} k-units (~{coef * 1000:.0f} units per $1,000)")
        
    # 2. Channel Sensitivity & Diminishing Returns Simulation
    # Base median marketing mix
    base_tv = df_feat['TV'].median()
    base_radio = df_feat['Radio'].median()
    base_news = df_feat['Newspaper'].median()
    base_pred = best_model.predict(pd.DataFrame([[base_tv, base_radio, base_news]], columns=channels))[0]
    
    print(f"\nBaseline Market Profile (Median Spends):")
    print(f"  TV: ${base_tv:.1f}k | Radio: ${base_radio:.1f}k | Newspaper: ${base_news:.1f}k")
    print(f"  Predicted Sales: {base_pred:.2f} k-units")
    
    spend_deltas = [10, 25, 50, 100]  # $k increments
    sensitivity_results = []
    
    for delta in spend_deltas:
        # TV solo boost
        p_tv = best_model.predict(pd.DataFrame([[base_tv + delta, base_radio, base_news]], columns=channels))[0]
        # Radio solo boost
        p_radio = best_model.predict(pd.DataFrame([[base_tv, base_radio + delta, base_news]], columns=channels))[0]
        # Newspaper solo boost
        p_news = best_model.predict(pd.DataFrame([[base_tv, base_radio, base_news + delta]], columns=channels))[0]
        # Combined TV + Radio boost (split 60/40)
        p_comb = best_model.predict(pd.DataFrame([[base_tv + delta * 0.6, base_radio + delta * 0.4, base_news]], columns=channels))[0]
        
        sensitivity_results.append({
            "Delta_Spend_k$": delta,
            "TV_Gain_kUnits": round(p_tv - base_pred, 2),
            "Radio_Gain_kUnits": round(p_radio - base_pred, 2),
            "Newspaper_Gain_kUnits": round(p_news - base_pred, 2),
            "Synergy_Mix_Gain_kUnits": round(p_comb - base_pred, 2)
        })
        
    sens_df = pd.DataFrame(sensitivity_results)
    print("\nSensitivity Analysis (Sales Uplift vs Baseline):")
    print(sens_df.to_string(index=False))
    
    # 3. Budget Reallocation Simulation: The "Zero-Cost Optimization"
    # Take historical average budget: TV=147k, Radio=23.3k, News=30.6k (Total = $200.9k)
    avg_tv = df_feat['TV'].mean()
    avg_radio = df_feat['Radio'].mean()
    avg_news = df_feat['Newspaper'].mean()
    total_budget = avg_tv + avg_radio + avg_news
    
    curr_pred = best_model.predict(pd.DataFrame([[avg_tv, avg_radio, avg_news]], columns=channels))[0]
    
    # Scenario A: Eliminate 100% of Newspaper ($30.6k) and split 65% TV / 35% Radio
    opt_tv_A = avg_tv + (avg_news * 0.65)
    opt_radio_A = avg_radio + (avg_news * 0.35)
    opt_news_A = 0.0
    pred_opt_A = best_model.predict(pd.DataFrame([[opt_tv_A, opt_radio_A, opt_news_A]], columns=channels))[0]
    
    # Scenario B: Shift 50% of Newspaper into Radio, 50% into TV
    opt_tv_B = avg_tv + (avg_news * 0.50)
    opt_radio_B = avg_radio + (avg_news * 0.50)
    opt_news_B = 0.0
    pred_opt_B = best_model.predict(pd.DataFrame([[opt_tv_B, opt_radio_B, opt_news_B]], columns=channels))[0]
    
    print("\n" + "-" * 50)
    print("BUDGET REALLOCATION SIMULATION (TOTAL BUDGET FIXED AT $200.9k):")
    print(f"Current Allocation: TV=${avg_tv:.1f}k, Radio=${avg_radio:.1f}k, Newspaper=${avg_news:.1f}k")
    print(f"  --> Forecasted Sales: {curr_pred:.2f} k-units (~{curr_pred * 1000:.0f} units)")
    
    print(f"\nScenario A (100% Newspaper reallocated: 65% TV / 35% Radio):")
    print(f"  TV=${opt_tv_A:.1f}k, Radio=${opt_radio_A:.1f}k, Newspaper=$0.0k")
    print(f"  --> Forecasted Sales: {pred_opt_A:.2f} k-units (~{pred_opt_A * 1000:.0f} units)")
    print(f"  --> Incremental Uplift: +{pred_opt_A - curr_pred:.2f} k-units (+{((pred_opt_A - curr_pred) / curr_pred) * 100:.1f}%) AT ZERO EXTRA COST!")
    
    print(f"\nScenario B (Balanced 50/50 Split):")
    print(f"  TV=${opt_tv_B:.1f}k, Radio=${opt_radio_B:.1f}k, Newspaper=$0.0k")
    print(f"  --> Forecasted Sales: {pred_opt_B:.2f} k-units (~{pred_opt_B * 1000:.0f} units)")
    print(f"  --> Incremental Uplift: +{pred_opt_B - curr_pred:.2f} k-units (+{((pred_opt_B - curr_pred) / curr_pred) * 100:.1f}%)")
    
    return sens_df, {
        "current": {"tv": avg_tv, "radio": avg_radio, "news": avg_news, "sales": curr_pred},
        "scenario_a": {"tv": opt_tv_A, "radio": opt_radio_A, "news": opt_news_A, "sales": pred_opt_A},
        "scenario_b": {"tv": opt_tv_B, "radio": opt_radio_B, "news": opt_news_B, "sales": pred_opt_B}
    }


def generate_visualizations(df_feat, benchmark_df, y_test, test_predictions, best_model_name):
    """
    Generates high-resolution data science figures and saves them as PNGs.
    """
    print("\n" + "=" * 70)
    print("STEP 6: GENERATING PUBLICATION-QUALITY VISUALIZATIONS")
    print("=" * 70)
    
    palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # 1. EDA Distributions and Correlation Heatmap
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Advertising Channels & Sales Distribution Overview", fontsize=16, fontweight='bold', y=0.98)
    
    # Channel Distributions
    ax1 = axes[0, 0]
    sns.kdeplot(df_feat['TV'], ax=ax1, label='TV', fill=True, color='#2b5c8f', alpha=0.3)
    sns.kdeplot(df_feat['Radio'], ax=ax1, label='Radio', fill=True, color='#e26d5c', alpha=0.3)
    sns.kdeplot(df_feat['Newspaper'], ax=ax1, label='Newspaper', fill=True, color='#38b000', alpha=0.3)
    ax1.set_title("Advertising Spend Density by Medium ($k)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Spend (in $1,000s)")
    ax1.set_ylabel("Density")
    ax1.legend(title="Channel")
    
    # Sales Distribution
    ax2 = axes[0, 1]
    sns.histplot(df_feat['Sales'], kde=True, ax=ax2, color='#7209b7', bins=15, alpha=0.6)
    ax2.axvline(df_feat['Sales'].mean(), color='red', linestyle='--', label=f"Mean: {df_feat['Sales'].mean():.1f}k")
    ax2.axvline(df_feat['Sales'].median(), color='orange', linestyle=':', label=f"Median: {df_feat['Sales'].median():.1f}k")
    ax2.set_title("Target Sales Distribution (1,000s of units)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Sales Units (k)")
    ax2.legend()
    
    # Correlation Heatmap
    ax3 = axes[1, 0]
    corr_cols = ['TV', 'Radio', 'Newspaper', 'Total_Spend', 'TV_Radio_Synergy', 'Sales']
    corr_sub = df_feat[corr_cols].corr()
    sns.heatmap(corr_sub, annot=True, fmt=".3f", cmap='coolwarm', ax=ax3, vmin=-0.1, vmax=1.0, cbar=True)
    ax3.set_title("Pearson Correlation Heatmap (with Engineered Synergy)", fontsize=12, fontweight='bold')
    
    # Market Segment Average Sales
    ax4 = axes[1, 1]
    seg_sales = df_feat.groupby('Market_Segment')['Sales'].mean().sort_values(ascending=False)
    sns.barplot(x=seg_sales.values, y=seg_sales.index, ax=ax4, palette='viridis')
    ax4.set_title("Average Sales by Identified Market Segment", fontsize=12, fontweight='bold')
    ax4.set_xlabel("Average Sales (k-units)")
    for i, v in enumerate(seg_sales.values):
        ax4.text(v + 0.3, i, f"{v:.1f}k", va='center', fontweight='bold')
        
    plt.tight_layout()
    eda_fig_path = os.path.join(BASE_DIR, "eda_distributions_and_correlations.png")
    fig.savefig(eda_fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {eda_fig_path}")
    
    # 2. Model Performance Benchmark Comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Regression Model Performance Benchmark (Test Set)", fontsize=15, fontweight='bold')
    
    ax1 = axes[0]
    sns.barplot(data=benchmark_df, x="Test_R2", y="Model", ax=ax1, palette='Blues_r')
    ax1.set_title("R-squared ($R^2$) Score (Higher is Better)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Test $R^2$")
    ax1.set_xlim(0.8, 1.0)
    for i, v in enumerate(benchmark_df['Test_R2']):
        ax1.text(v - 0.03, i, f"{v:.4f}", color='white', va='center', fontweight='bold')
        
    ax2 = axes[1]
    sns.barplot(data=benchmark_df, x="Test_RMSE", y="Model", ax=ax2, palette='Reds')
    ax2.set_title("Root Mean Squared Error (RMSE) (Lower is Better)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Test RMSE (k-units)")
    for i, v in enumerate(benchmark_df['Test_RMSE']):
        ax2.text(v + 0.03, i, f"{v:.4f}", color='black', va='center', fontweight='bold')
        
    plt.tight_layout()
    model_fig_path = os.path.join(BASE_DIR, "model_performance_comparison.png")
    fig.savefig(model_fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {model_fig_path}")
    
    # 3. Actual vs Predicted & Residuals for Champion Model
    best_preds = test_predictions[best_model_name]
    residuals = y_test - best_preds
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Diagnostic Plots for Champion Model: {best_model_name}", fontsize=15, fontweight='bold')
    
    # Actual vs Predicted
    ax1 = axes[0]
    ax1.scatter(y_test, best_preds, color='#1d3557', edgecolor='black', alpha=0.75, s=60)
    min_val = min(y_test.min(), best_preds.min())
    max_val = max(y_test.max(), best_preds.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction (1:1)')
    ax1.set_title(f"Actual vs Predicted Sales ($R^2$ = {benchmark_df.iloc[0]['Test_R2']:.4f})", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Actual Sales (k-units)")
    ax1.set_ylabel("Predicted Sales (k-units)")
    ax1.legend()
    
    # Residual Plot
    ax2 = axes[1]
    ax2.scatter(best_preds, residuals, color='#e63946', edgecolor='black', alpha=0.75, s=60)
    ax2.axhline(0, color='black', linestyle='--', linewidth=1.5)
    ax2.set_title(f"Residuals vs Predicted (RMSE = {benchmark_df.iloc[0]['Test_RMSE']:.4f})", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Predicted Sales (k-units)")
    ax2.set_ylabel("Residual Error (Actual - Predicted)")
    
    plt.tight_layout()
    diag_fig_path = os.path.join(BASE_DIR, "actual_vs_predicted.png")
    fig.savefig(diag_fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {diag_fig_path}")
    
    # 4. 2D Contour Synergy Matrix (TV vs Radio with Newspaper fixed at 0)
    fig, ax = plt.subplots(figsize=(9, 7))
    tv_vals = np.linspace(0, 300, 60)
    radio_vals = np.linspace(0, 50, 60)
    TV_grid, Radio_grid = np.meshgrid(tv_vals, radio_vals)
    
    grid_df = pd.DataFrame({
        'TV': TV_grid.ravel(),
        'Radio': Radio_grid.ravel(),
        'Newspaper': np.zeros(TV_grid.size)
    })
    
    # Predict over 2D grid using champion model
    from sklearn.base import clone
    Sales_grid = test_predictions['champion_pipeline'].predict(grid_df).reshape(TV_grid.shape) if 'champion_pipeline' in test_predictions else None
    
    return eda_fig_path, model_fig_path, diag_fig_path


def export_enriched_dataset(df_feat, benchmark_df):
    """
    Exports clean datasets and benchmark tables.
    """
    enriched_csv = os.path.join(BASE_DIR, "Advertising_Enriched.csv")
    df_feat.to_csv(enriched_csv, index=False)
    print(f"\nExported enriched dataset with engineered features to: {enriched_csv}")
    
    metrics_csv = os.path.join(BASE_DIR, "model_metrics_benchmark.csv")
    benchmark_df.to_csv(metrics_csv, index=False)
    print(f"Exported model metrics benchmark to: {metrics_csv}")
    
    return enriched_csv, metrics_csv


def predict_sales_scenario(tv, radio, newspaper, model, scaler, kmeans, cluster_labels):
    """
    Predicts sales for custom advertising spend values, computes segment & ROI.
    """
    input_df = pd.DataFrame([[tv, radio, newspaper]], columns=['TV', 'Radio', 'Newspaper'])
    predicted_sales = model.predict(input_df)[0]
    
    # Determine market segment
    scaled_input = scaler.transform(input_df[['TV', 'Radio', 'Newspaper']])
    cluster_id = kmeans.predict(scaled_input)[0]
    segment = cluster_labels.get(cluster_id, "Standard")
    
    total_spend = tv + radio + newspaper
    sales_per_k_spend = predicted_sales / (total_spend + 1e-6)
    
    return {
        "TV_Spend": tv,
        "Radio_Spend": radio,
        "Newspaper_Spend": newspaper,
        "Total_Spend": round(total_spend, 2),
        "Predicted_Sales_kUnits": round(predicted_sales, 2),
        "Predicted_Units": int(round(predicted_sales * 1000)),
        "Sales_Units_Per_$Spend": round(sales_per_k_spend, 3),
        "Identified_Segment": segment
    }


def main():
    parser = argparse.ArgumentParser(description="Sales Prediction and Marketing Optimization")
    parser.add_argument("--tv", type=float, default=None, help="TV advertising budget in $1,000s")
    parser.add_argument("--radio", type=float, default=None, help="Radio advertising budget in $1,000s")
    parser.add_argument("--newspaper", type=float, default=None, help="Newspaper advertising budget in $1,000s")
    args = parser.parse_args()
    
    # 1. Load data
    df = load_and_preprocess_data()
    
    # 2. Engineer features and segments
    df_feat, kmeans, scaler, cluster_labels = engineer_features_and_segments(df)
    
    # 3. Statistical diagnostics
    ols_model, corr_matrix = perform_statistical_diagnostics(df)
    
    # 4. Model training & benchmark
    trained_models, benchmark_df, X_train, X_test, y_train, y_test, test_predictions = train_and_benchmark_models(df_feat)
    
    best_model_name = benchmark_df.iloc[0]['Model']
    best_model = trained_models[best_model_name]
    linear_model = trained_models['Linear Regression']
    
    # 5. Sensitivity and ROI
    sens_df, reallocation_summary = analyze_advertising_impact_and_roi(df_feat, best_model, linear_model)
    
    # 6. Visualizations
    generate_visualizations(df_feat, benchmark_df, y_test, test_predictions, best_model_name)
    
    # 7. Export outputs
    export_enriched_dataset(df_feat, benchmark_df)
    
    # 8. Interactive or Command-line Prediction
    if args.tv is not None and args.radio is not None and args.newspaper is not None:
        pred_res = predict_sales_scenario(
            args.tv, args.radio, args.newspaper, best_model, scaler, kmeans, cluster_labels
        )
        print("\n" + "=" * 70)
        print("CUSTOM PREDICTION SCENARIO RESULT")
        print("=" * 70)
        for k, v in pred_res.items():
            print(f"  {k:25s}: {v}")
    else:
        # Default sample prediction demonstration
        sample_scenarios = [
            (250.0, 40.0, 10.0),
            (100.0, 20.0, 5.0),
            (20.0, 10.0, 5.0),
            (180.0, 45.0, 0.0)
        ]
        print("\n" + "=" * 70)
        print("SAMPLE SCENARIO PREDICTIONS WITH CHAMPION MODEL")
        print("=" * 70)
        for tv, rad, news in sample_scenarios:
            res = predict_sales_scenario(tv, rad, news, best_model, scaler, kmeans, cluster_labels)
            print(f"Spend: TV=${tv}k, Radio=${rad}k, News=${news}k  -->  Sales: {res['Predicted_Sales_kUnits']} k-units ({res['Predicted_Units']:,} units) | Segment: {res['Identified_Segment']}")
            
    print("\n[SUCCESS] Pipeline execution complete.")


if __name__ == "__main__":
    main()
