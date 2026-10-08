"""
Cross Validation Engine
Roll Number: BT2024182
"""

import numpy as np
from .evaluate import mean_squared_error, r2_score
from .polynomial import standardize_features


def k_fold_split(n_samples, n_splits=5, shuffle=True, random_state=42):
    """
    Generate train and validation indices for K-fold cross-validation.
    """
    rng = np.random.RandomState(random_state)
    indices = np.arange(n_samples)
    if shuffle:
        rng.shuffle(indices)
    folds = np.array_split(indices, n_splits)
    
    splits = []
    for i in range(n_splits):
        val_idx = folds[i]
        train_idx = np.setdiff1d(indices, val_idx)
        splits.append((train_idx, val_idx))
    return splits


def cross_validate(model_cls, model_kwargs, X, y, n_splits=5, standardize=True, random_state=42):
    """
    Perform K-Fold cross-validation for a given model class.
    
    Parameters
    ----------
    model_cls : class
        Model class to instantiate for each fold.
    model_kwargs : dict
        Arguments passed to model constructor.
    X : np.ndarray
        Design matrix (including bias column at index 0).
    y : np.ndarray
        Target vector.
    n_splits : int
        Number of cross-validation folds.
    standardize : bool
        Whether to standardize non-bias features per fold.
        
    Returns
    -------
    cv_summary : dict
        Contains mean and standard deviation of MSE and R2 on train and validation folds.
    """
    splits = k_fold_split(len(y), n_splits=n_splits, shuffle=True, random_state=random_state)
    
    val_mses, tr_mses = [], []
    val_r2s, tr_r2s = [], []
    
    for tr_idx, val_idx in splits:
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_va, y_va = X[val_idx], y[val_idx]
        
        if standardize:
            X_tr, means, stds = standardize_features(X_tr, has_bias=True)
            X_va, _, _ = standardize_features(X_va, means=means, stds=stds, has_bias=True)
            
        model = model_cls(**model_kwargs)
        model.fit(X_tr, y_tr)
        
        pred_tr = model.predict(X_tr)
        pred_va = model.predict(X_va)
        
        tr_mses.append(mean_squared_error(y_tr, pred_tr))
        val_mses.append(mean_squared_error(y_va, pred_va))
        tr_r2s.append(r2_score(y_tr, pred_tr))
        val_r2s.append(r2_score(y_va, pred_va))
        
    return {
        "val_mse_mean": float(np.mean(val_mses)),
        "val_mse_std": float(np.std(val_mses)),
        "tr_mse_mean": float(np.mean(tr_mses)),
        "tr_mse_std": float(np.std(tr_mses)),
        "val_r2_mean": float(np.mean(val_r2s)),
        "val_r2_std": float(np.std(val_r2s)),
        "tr_r2_mean": float(np.mean(tr_r2s)),
        "tr_r2_std": float(np.std(tr_r2s)),
    }
