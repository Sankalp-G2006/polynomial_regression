#!/usr/bin/env python3
"""
Inference Script for Generating Assignment Prediction CSVs
Roll Number: BT2024182
"""

import os
import sys
import shutil
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.polynomial import generate_polynomial_features, standardize_features


def generate_predictions():
    print("=" * 70)
    print("GENERATING PREDICTIONS FOR ROLL NUMBER: BT2024182")
    print("=" * 70)
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(base_dir, "data")
    pred_dir = os.path.join(base_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    
    # --- PHASE 1 (var1) ---
    print("\n--- Processing Phase 1 (var1: Steam Turbine Optimization) ---")
    test1_path = os.path.join(data_dir, "BT2024182_test_var1.csv")
    model1_path = os.path.join(data_dir, "model_var1.npz")
    
    df_test1 = pd.read_csv(test1_path)
    X_test1 = df_test1[[f"x{i}" for i in range(1, 7)]].values
    print(f"Test var1 loaded: {X_test1.shape[0]} samples.")
    
    m1 = np.load(model1_path)
    deg1 = int(m1["degree"])
    w1 = m1["weights"]
    means1 = m1["means"]
    stds1 = m1["stds"]
    
    X_poly1, _ = generate_polynomial_features(X_test1, degree=deg1, include_bias=True)
    X_poly1_scaled, _, _ = standardize_features(X_poly1, means=means1, stds=stds1, has_bias=True)
    pred_y1 = X_poly1_scaled @ w1
    
    # Validation checks
    assert len(pred_y1) == 1000, f"Expected 1000 predictions, got {len(pred_y1)}"
    assert not np.isnan(pred_y1).any(), "Found NaNs in var1 predictions!"
    assert not np.isinf(pred_y1).any(), "Found Infs in var1 predictions!"
    
    pred1_df = pd.DataFrame({"y": pred_y1})
    out1_path = os.path.join(pred_dir, "BT2024182_pred_var1.csv")
    pred1_df.to_csv(out1_path, index=False)
    print(f"Phase 1 predictions saved to: {out1_path}")
    print(f"  Summary: Mean={pred_y1.mean():.4f}, Std={pred_y1.std():.4f}, Min={pred_y1.min():.4f}, Max={pred_y1.max():.4f}")
    
    # --- PHASE 2 (var2) ---
    print("\n--- Processing Phase 2 (var2: Subterranean Thermal Mapping) ---")
    test2_path = os.path.join(data_dir, "BT2024182_test_var2.csv")
    model2_path = os.path.join(data_dir, "model_var2.npz")
    
    df_test2 = pd.read_csv(test2_path)
    X_test2 = df_test2[["x1", "x2", "x3"]].values
    print(f"Test var2 loaded: {X_test2.shape[0]} samples.")
    
    m2 = np.load(model2_path)
    deg2 = int(m2["degree"])
    w2 = m2["weights"]
    
    X_poly2, _ = generate_polynomial_features(X_test2, degree=deg2, include_bias=True)
    pred_y2 = X_poly2 @ w2
    
    # Validation checks
    assert len(pred_y2) == 1000, f"Expected 1000 predictions, got {len(pred_y2)}"
    assert not np.isnan(pred_y2).any(), "Found NaNs in var2 predictions!"
    assert not np.isinf(pred_y2).any(), "Found Infs in var2 predictions!"
    
    pred2_df = pd.DataFrame({"y": pred_y2})
    out2_path = os.path.join(pred_dir, "BT2024182_pred_var2.csv")
    pred2_df.to_csv(out2_path, index=False)
    print(f"Phase 2 predictions saved to: {out2_path}")
    print(f"  Summary: Mean={pred_y2.mean():.4f}, Std={pred_y2.std():.4f}, Min={pred_y2.min():.4f}, Max={pred_y2.max():.4f}")
    
    # Copy predictions also to ~/Downloads for convenient user submission access
    downloads_dir = os.path.expanduser("~/Downloads")
    if os.path.exists(downloads_dir):
        shutil.copy2(out1_path, os.path.join(downloads_dir, "BT2024182_pred_var1.csv"))
        shutil.copy2(out2_path, os.path.join(downloads_dir, "BT2024182_pred_var2.csv"))
        print(f"\nAlso mirrored prediction files to: {downloads_dir}")
        
    print("\nFormat Verification:")
    sample_sub_path = os.path.expanduser("~/Downloads/sample_submission.csv")
    if os.path.exists(sample_sub_path):
        sample_df = pd.read_csv(sample_sub_path)
        print(f"  sample_submission.csv shape: {sample_df.shape}, cols: {sample_df.columns.tolist()}")
        print(f"  BT2024182_pred_var1.csv shape: {pred1_df.shape}, cols: {pred1_df.columns.tolist()}")
        print(f"  BT2024182_pred_var2.csv shape: {pred2_df.shape}, cols: {pred2_df.columns.tolist()}")
        assert sample_df.columns.tolist() == pred1_df.columns.tolist() == pred2_df.columns.tolist(), "Column names do not match sample!"
        assert len(sample_df) == len(pred1_df) == len(pred2_df), "Line counts do not match sample submission!"
        print("  -> FORMAT MATCH CONFIRMED!")


if __name__ == "__main__":
    generate_predictions()
