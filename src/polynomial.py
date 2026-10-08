"""
Polynomial Feature Generation and Scaling
Roll Number: BT2024182
"""

import itertools
import numpy as np


def generate_polynomial_features(X, degree, include_bias=True):
    """
    Generate polynomial features up to a given degree for multi-dimensional data.
    
    A polynomial of degree d means for each term, the sum of powers of features
    adds at most up to d.
    
    Parameters
    ----------
    X : np.ndarray of shape (N, D)
        Input design matrix.
    degree : int
        Maximum degree of the polynomial.
    include_bias : bool, default=True
        Whether to include the intercept term (column of 1s).
        
    Returns
    -------
    X_poly : np.ndarray of shape (N, P)
        Transformed feature matrix containing all monomial combinations.
    feature_combinations : list of tuples
        List of index tuples representing each monomial term.
    """
    X = np.asarray(X, dtype=np.float64)
    n_samples, n_features = X.shape
    
    combinations = []
    if include_bias:
        combinations.append(())
        
    for d in range(1, degree + 1):
        combinations.extend(list(itertools.combinations_with_replacement(range(n_features), d)))
        
    n_output_features = len(combinations)
    X_poly = np.empty((n_samples, n_output_features), dtype=np.float64)
    
    for i, comb in enumerate(combinations):
        if len(comb) == 0:
            X_poly[:, i] = 1.0
        elif len(comb) == 1:
            X_poly[:, i] = X[:, comb[0]]
        else:
            X_poly[:, i] = np.prod(X[:, comb], axis=1)
            
    return X_poly, combinations


def standardize_features(X, means=None, stds=None, has_bias=True):
    """
    Standardize features to zero mean and unit variance.
    If has_bias is True, column 0 (the intercept) is not standardized.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    means : np.ndarray, optional
        Precomputed column means.
    stds : np.ndarray, optional
        Precomputed column standard deviations.
    has_bias : bool, default=True
        Whether column 0 is the bias/intercept column.
        
    Returns
    -------
    X_scaled : np.ndarray
    means : np.ndarray
    stds : np.ndarray
    """
    X_scaled = np.array(X, dtype=np.float64, copy=True)
    start_idx = 1 if has_bias else 0
    
    if means is None:
        means = np.mean(X_scaled[:, start_idx:], axis=0)
    if stds is None:
        stds = np.std(X_scaled[:, start_idx:], axis=0)
        # Avoid division by zero
        stds[stds == 0.0] = 1.0
        
    X_scaled[:, start_idx:] = (X_scaled[:, start_idx:] - means) / stds
    return X_scaled, means, stds
