#!/usr/bin/env python3
"""
Phase 2: Subterranean Thermal Reservoir Mapping (var2)
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
from src.models import LinearRegression, RidgeRegression, LassoRegression
from src.cross_val import cross_validate
from src.evaluate import compute_metrics, mean_squared_error, r2_score


def run_phase2_experiment():
    print("=" * 70)
    print("PHASE 2: SUBTERRANEAN THERMAL RESERVOIR MAPPING (var2)")
    print("Roll Number: BT2024182")
    print("=" * 70)
    
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "BT2024182_train_var2.csv")
    df = pd.read_csv(data_path)
    feature_cols = ["x1", "x2", "x3"]
    X_raw = df[feature_cols].values
    y = df["y"].values
    
    print(f"Dataset Loaded: {X_raw.shape[0]} spatial core samples, 3 coordinates (x1: East-West, x2: North-South, x3: Depth).")
    print(f"Target Summary: Mean={y.mean():.4f}, Std={y.std():.4f}, Min={y.min():.4f}, Max={y.max():.4f}")
    
    degrees = [1, 2, 4, 6, 8, 10, 12]
    results = []
    
    print("\n--- Systematic Cross-Validation across Polynomial Degrees ---")
    for d in degrees:
        X_poly, combos = generate_polynomial_features(X_raw, degree=d, include_bias=True)
        n_feats = X_poly.shape[1]
        
        # Test OLS (only up to degree 8 where n_feats <= 165)
        if n_feats <= 200:
            res_ols = cross_validate(LinearRegression, {}, X_poly, y, n_splits=5, standardize=False)
        else:
            res_ols = {"val_mse_mean": np.nan, "val_r2_mean": np.nan}
            
        # Test Ridge (tuned)
        ridge_alpha = 0.05 if d >= 10 else (0.005 if d >= 8 else (0.02 if d >= 6 else 0.01))
        res_ridge = cross_validate(RidgeRegression, {"alpha": ridge_alpha}, X_poly, y, n_splits=5, standardize=False)
        
        print(f"Degree {d:2d} ({n_feats:4d} terms):")
        print(f"  OLS:        Val MSE = {res_ols['val_mse_mean']:.5f}, Val R2 = {res_ols.get('val_r2_mean', 0.0):.5f}")
        print(f"  Ridge (L2): Val MSE = {res_ridge['val_mse_mean']:.5f}, Val R2 = {res_ridge['val_r2_mean']:.5f}")
        
        results.append({
            "degree": d,
            "terms": n_feats,
            "ols_val_mse": res_ols['val_mse_mean'],
            "ridge_val_mse": res_ridge['val_mse_mean'],
            "ridge_val_r2": res_ridge['val_r2_mean'],
        })
        
    res_df = pd.DataFrame(results)
    print("\nSummary Table:")
    print(res_df.to_string(index=False))
    
    # Generate Visualizations
    fig_dir = os.path.join(os.path.dirname(__file__), "..", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    # 1. Degree vs MSE / R2 Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(res_df["degree"], res_df["ridge_val_mse"], marker="s", color="#d62728", linewidth=2, label="Ridge (L2, Optimal $\\alpha$)")
    ax1.axhline(0.245, color="black", linestyle=":", label=r"Inherent Noise Floor ($\sigma^2 \approx 0.25$)")
    ax1.set_xlabel("Polynomial Degree", fontsize=11, fontweight="bold")
    ax1.set_ylabel("5-Fold Validation MSE", fontsize=11, fontweight="bold")
    ax1.set_title("Phase 2: Validation MSE vs Degree", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(fontsize=10)
    
    ax2.plot(res_df["degree"], res_df["ridge_val_r2"], marker="s", color="#2ca02c", linewidth=2, label="Ridge (L2)")
    ax2.set_xlabel("Polynomial Degree", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Validation $R^2$ Score", fontsize=11, fontweight="bold")
    ax2.set_title("Phase 2: Validation $R^2$ vs Degree", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(fontsize=10)
    plt.tight_layout()
    plot_path1 = os.path.join(fig_dir, "var2_degree_vs_error.png")
    plt.savefig(plot_path1, dpi=300)
    plt.close()
    print(f"\nSaved degree comparison chart to: {plot_path1}")
    
    # 2. Fit Optimal Model on Full Training Set (Degree 10 Ridge, alpha=0.05)
    optimal_degree = 10
    print(f"\nTraining Final Phase 2 Model on Full Training Set (Degree {optimal_degree} Ridge, alpha=0.05)...")
    X_poly10, combos10 = generate_polynomial_features(X_raw, degree=optimal_degree, include_bias=True)
    
    final_model = RidgeRegression(alpha=0.05)
    final_model.fit(X_poly10, y)
    
    y_pred = final_model.predict(X_poly10)
    train_metrics = compute_metrics(y, y_pred)
    print(f"Final Model Trained: {X_poly10.shape[1]} polynomial terms.")
    print(f"Full Train Metrics: MSE = {train_metrics['mse']:.5f}, R2 = {train_metrics['r2']:.5f}, MAE = {train_metrics['mae']:.5f}")
    
    # Residuals Plot
    residuals = y - y_pred
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.scatter(y_pred, residuals, alpha=0.4, color="#d62728", edgecolors="none")
    ax1.axhline(0, color="black", linestyle="--", linewidth=1.5)
    ax1.set_xlabel(r"Fitted Thermal Anomaly Score ($\hat{y}$)", fontsize=11, fontweight="bold")
    ax1.set_ylabel(r"Residuals ($y - \hat{y}$)", fontsize=11, fontweight="bold")
    ax1.set_title("Phase 2: Residuals vs Fitted", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)
    
    ax2.hist(residuals, bins=30, color="#1f77b4", edgecolor="black", alpha=0.7, density=True)
    ax2.set_xlabel("Residual Error", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Density", fontsize=11, fontweight="bold")
    ax2.set_title("Phase 2: Residual Distribution", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plot_path2 = os.path.join(fig_dir, "var2_residuals.png")
    plt.savefig(plot_path2, dpi=300)
    plt.close()
    print(f"Saved residual diagnostics to: {plot_path2}")
    
    # 3. Subterranean Thermal Map 2D Slice Visualizer (at depth x3 = 0)
    grid_size = 100
    x1_lin = np.linspace(-1, 1, grid_size)
    x2_lin = np.linspace(-1, 1, grid_size)
    X1_grid, X2_grid = np.meshgrid(x1_lin, x2_lin)
    grid_samples = np.column_stack([X1_grid.ravel(), X2_grid.ravel(), np.zeros(grid_size * grid_size)])
    
    grid_poly, _ = generate_polynomial_features(grid_samples, degree=optimal_degree, include_bias=True)
    pred_grid = final_model.predict(grid_poly).reshape((grid_size, grid_size))
    
    fig, ax = plt.subplots(figsize=(7, 6))
    contour = ax.contourf(X1_grid, X2_grid, pred_grid, levels=25, cmap="inferno")
    fig.colorbar(contour, ax=ax, label="Thermal Anomaly Score (y)")
    ax.scatter(X_raw[:, 0], X_raw[:, 1], c="cyan", s=10, alpha=0.4, label="Sample Core Locations")
    ax.set_xlabel("East-West Coordinate Offset ($x_1$)", fontsize=11, fontweight="bold")
    ax.set_ylabel("North-South Coordinate Offset ($x_2$)", fontsize=11, fontweight="bold")
    ax.set_title("Subterranean Thermal Reservoir Map ($x_3=0$ Depth Slice)", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plot_path3 = os.path.join(fig_dir, "var2_thermal_slice.png")
    plt.savefig(plot_path3, dpi=300)
    plt.close()
    print(f"Saved thermal slice visualization to: {plot_path3}")
    
    # Save Model Artifacts
    model_artifact_path = os.path.join(os.path.dirname(__file__), "..", "data", "model_var2.npz")
    np.savez(
        model_artifact_path,
        degree=optimal_degree,
        weights=final_model.weights,
    )
    print(f"Saved Phase 2 model weights to: {model_artifact_path}")
    return res_df, train_metrics


if __name__ == "__main__":
    run_phase2_experiment()
