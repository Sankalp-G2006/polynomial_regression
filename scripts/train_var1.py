#!/usr/bin/env python3
"""
Phase 1: Power Plant Steam Turbine Optimization (var1)
Roll Number: BT2024182
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.polynomial import generate_polynomial_features, standardize_features
from src.models import LinearRegression, RidgeRegression, LassoRegression, PostLassoRegression
from src.cross_val import cross_validate
from src.evaluate import compute_metrics, mean_squared_error, r2_score


def run_phase1_experiment():
    print("=" * 70)
    print("PHASE 1: STEAM TURBINE OPTIMIZATION (var1)")
    print("Roll Number: BT2024182")
    print("=" * 70)
    
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "BT2024182_train_var1.csv")
    df = pd.read_csv(data_path)
    feature_cols = [f"x{i}" for i in range(1, 7)]
    X_raw = df[feature_cols].values
    y = df["y"].values
    
    print(f"Dataset Loaded: {X_raw.shape[0]} samples, {X_raw.shape[1]} operational variables.")
    print(f"Target Summary: Mean={y.mean():.4f}, Std={y.std():.4f}, Min={y.min():.4f}, Max={y.max():.4f}")
    
    degrees = [1, 2, 3, 4, 5]
    results = []
    
    print("\n--- Systematic Cross-Validation across Polynomial Degrees ---")
    for d in degrees:
        X_poly, combos = generate_polynomial_features(X_raw, degree=d, include_bias=True)
        n_feats = X_poly.shape[1]
        
        # Test OLS
        if n_feats < 700:
            res_ols = cross_validate(LinearRegression, {}, X_poly, y, n_splits=5, standardize=False)
        else:
            res_ols = {"val_mse_mean": np.nan, "val_r2_mean": np.nan}
            
        # Test Ridge (tuned)
        best_ridge_alpha = 10.0 if d >= 4 else (2.0 if d == 3 else 1.0)
        res_ridge = cross_validate(RidgeRegression, {"alpha": best_ridge_alpha}, X_poly, y, n_splits=5, standardize=True)
        
        # Test Lasso
        best_lasso_alpha = 0.010 if d >= 4 else 0.005
        res_lasso = cross_validate(LassoRegression, {"alpha": best_lasso_alpha}, X_poly, y, n_splits=5, standardize=True)
        
        # Test Post-Lasso
        res_post = cross_validate(PostLassoRegression, {"alpha": best_lasso_alpha}, X_poly, y, n_splits=5, standardize=True)
        
        print(f"Degree {d:2d} ({n_feats:4d} terms):")
        print(f"  OLS:        Val MSE = {res_ols['val_mse_mean']:.5f}, Val R2 = {res_ols.get('val_r2_mean', 0.0):.5f}")
        print(f"  Ridge (L2): Val MSE = {res_ridge['val_mse_mean']:.5f}, Val R2 = {res_ridge['val_r2_mean']:.5f}")
        print(f"  Lasso (L1): Val MSE = {res_lasso['val_mse_mean']:.5f}, Val R2 = {res_lasso['val_r2_mean']:.5f}")
        print(f"  Post-Lasso: Val MSE = {res_post['val_mse_mean']:.5f}, Val R2 = {res_post['val_r2_mean']:.5f}")
        
        results.append({
            "degree": d,
            "terms": n_feats,
            "ols_val_mse": res_ols['val_mse_mean'],
            "ridge_val_mse": res_ridge['val_mse_mean'],
            "ridge_val_r2": res_ridge['val_r2_mean'],
            "lasso_val_mse": res_lasso['val_mse_mean'],
            "lasso_val_r2": res_lasso['val_r2_mean'],
            "post_lasso_val_mse": res_post['val_mse_mean'],
            "post_lasso_val_r2": res_post['val_r2_mean'],
        })
        
    res_df = pd.DataFrame(results)
    print("\nSummary Table:")
    print(res_df.to_string(index=False))
    
    # Generate Visualizations
    fig_dir = os.path.join(os.path.dirname(__file__), "..", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    # 1. Degree vs MSE / R2 Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(res_df["degree"], res_df["ridge_val_mse"], marker="s", color="#1f77b4", label="Ridge (L2)")
    ax1.plot(res_df["degree"], res_df["lasso_val_mse"], marker="o", color="#ff7f0e", label="Lasso (L1)")
    ax1.plot(res_df["degree"], res_df["post_lasso_val_mse"], marker="^", color="#2ca02c", label="Post-Lasso OLS")
    ax1.set_xlabel("Polynomial Degree", fontsize=11, fontweight="bold")
    ax1.set_ylabel("5-Fold Validation MSE", fontsize=11, fontweight="bold")
    ax1.set_title("Phase 1: Validation MSE vs Degree", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(fontsize=10)
    
    ax2.plot(res_df["degree"], res_df["ridge_val_r2"], marker="s", color="#1f77b4", label="Ridge (L2)")
    ax2.plot(res_df["degree"], res_df["lasso_val_r2"], marker="o", color="#ff7f0e", label="Lasso (L1)")
    ax2.plot(res_df["degree"], res_df["post_lasso_val_r2"], marker="^", color="#2ca02c", label="Post-Lasso OLS")
    ax2.set_xlabel("Polynomial Degree", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Validation $R^2$ Score", fontsize=11, fontweight="bold")
    ax2.set_title("Phase 1: Validation $R^2$ vs Degree", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(fontsize=10)
    plt.tight_layout()
    plot_path1 = os.path.join(fig_dir, "var1_degree_vs_error.png")
    plt.savefig(plot_path1, dpi=300)
    plt.close()
    print(f"\nSaved degree comparison chart to: {plot_path1}")
    
    # 2. Fit Optimal Model on Full Training Set (Degree 5 Post-Lasso)
    optimal_degree = 5
    print(f"\nTraining Final Phase 1 Model on Full Training Set (Degree {optimal_degree} Post-Lasso)...")
    X_poly5, combos5 = generate_polynomial_features(X_raw, degree=optimal_degree, include_bias=True)
    X_poly5_scaled, means, stds = standardize_features(X_poly5, has_bias=True)
    
    final_model = PostLassoRegression(alpha=0.012, ridge_stability=1e-4)
    final_model.fit(X_poly5_scaled, y)
    
    y_pred = final_model.predict(X_poly5_scaled)
    train_metrics = compute_metrics(y, y_pred)
    active_terms = final_model.active_indices
    print(f"Final Model Trained: {len(active_terms)} active monomial terms out of {X_poly5.shape[1]}.")
    print(f"Full Train Metrics: MSE = {train_metrics['mse']:.5f}, R2 = {train_metrics['r2']:.5f}, MAE = {train_metrics['mae']:.5f}")
    
    # Residuals Plot
    residuals = y - y_pred
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.scatter(y_pred, residuals, alpha=0.4, color="#1f77b4", edgecolors="none")
    ax1.axhline(0, color="red", linestyle="--", linewidth=1.5)
    ax1.set_xlabel(r"Fitted Net Power Score ($\hat{y}$)", fontsize=11, fontweight="bold")
    ax1.set_ylabel(r"Residuals ($y - \hat{y}$)", fontsize=11, fontweight="bold")
    ax1.set_title("Phase 1: Residuals vs Fitted", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)
    
    ax2.hist(residuals, bins=30, color="#2ca02c", edgecolor="black", alpha=0.7, density=True)
    ax2.set_xlabel("Residual Error", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Density", fontsize=11, fontweight="bold")
    ax2.set_title("Phase 1: Residual Distribution", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plot_path2 = os.path.join(fig_dir, "var1_residuals.png")
    plt.savefig(plot_path2, dpi=300)
    plt.close()
    print(f"Saved residual diagnostics to: {plot_path2}")
    
    # Save Model Artifacts
    model_artifact_path = os.path.join(os.path.dirname(__file__), "..", "data", "model_var1.npz")
    np.savez(
        model_artifact_path,
        degree=optimal_degree,
        weights=final_model.weights,
        means=means,
        stds=stds,
        active_indices=final_model.active_indices,
    )
    print(f"Saved Phase 1 model weights to: {model_artifact_path}")
    return res_df, train_metrics


if __name__ == "__main__":
    run_phase1_experiment()
