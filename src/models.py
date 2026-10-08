"""
Regression Models: OLS, Ridge, Lasso, and Post-Lasso
Roll Number: BT2024182
"""

import numpy as np


class LinearRegression:
    """
    Ordinary Least Squares (OLS) Linear Regression.
    Solves w* = argmin ||y - Xw||_2^2 via Moore-Penrose pseudoinverse / QR decomposition.
    """
    def __init__(self, fit_intercept=False):
        self.fit_intercept = fit_intercept
        self.weights = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        if self.fit_intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])
            
        # Solve least squares via SVD-based robust solver
        self.weights, residuals, rank, s = np.linalg.lstsq(X, y, rcond=1e-12)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        if self.fit_intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])
        return X @ self.weights


class RidgeRegression:
    """
    Ridge Regression (L2 Regularization).
    Solves w* = argmin ||y - Xw||_2^2 + alpha * ||w_{1:}||_2^2
    The intercept term (column 0) is left unpenalized.
    """
    def __init__(self, alpha=1.0, fit_intercept=False):
        self.alpha = float(alpha)
        self.fit_intercept = fit_intercept
        self.weights = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        if self.fit_intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])
            
        n_features = X.shape[1]
        I = np.eye(n_features, dtype=np.float64)
        # Never penalize the intercept (column 0)
        I[0, 0] = 0.0
        
        A = X.T @ X + self.alpha * I
        b = X.T @ y
        
        try:
            self.weights = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            self.weights = np.linalg.lstsq(A, b, rcond=1e-12)[0]
            
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        if self.fit_intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])
        return X @ self.weights


class LassoRegression:
    """
    Lasso Regression (L1 Regularization).
    Solves w* = argmin 1/(2N) * ||y - Xw||_2^2 + alpha * ||w_{1:}||_1
    Implemented using Cyclic Coordinate Descent with soft-thresholding.
    """
    def __init__(self, alpha=0.01, max_iter=2000, tol=1e-5, fit_intercept=False):
        self.alpha = float(alpha)
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.weights = None

    @staticmethod
    def _soft_threshold(rho, lam):
        if rho > lam:
            return rho - lam
        elif rho < -lam:
            return rho + lam
        else:
            return 0.0

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        if self.fit_intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])
            
        n_samples, n_features = X.shape
        w = np.zeros(n_features, dtype=np.float64)
        w[0] = np.mean(y) # Initial intercept
        
        # Precompute column squared norms
        col_norms = np.sum(X[:, 1:]**2, axis=0)
        col_norms[col_norms == 0.0] = 1.0
        
        lam = self.alpha * n_samples
        residuals = y - X @ w
        
        for iteration in range(self.max_iter):
            w_prev = w.copy()
            
            # Intercept update (unpenalized)
            w[0] = np.mean(y - X[:, 1:] @ w[1:])
            residuals = y - X @ w
            
            # Coordinate descent over feature coefficients
            for j in range(1, n_features):
                xj = X[:, j]
                norm_j = col_norms[j - 1]
                
                # Add back current variable's contribution
                residuals += xj * w[j]
                rho = np.dot(xj, residuals)
                
                # Soft-thresholding operator
                w[j] = self._soft_threshold(rho, lam) / norm_j
                
                # Update residuals
                residuals -= xj * w[j]
                
            max_change = np.max(np.abs(w - w_prev))
            if max_change < self.tol:
                break
                
        self.weights = w
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        if self.fit_intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])
        return X @ self.weights

    @property
    def active_features(self):
        """Indices of features with non-zero coefficients."""
        if self.weights is None:
            return []
        return np.where(np.abs(self.weights) > 1e-6)[0]


class PostLassoRegression:
    """
    Two-Stage Post-Lasso Estimator:
    1. Feature selection via Lasso (L1 regularization) to discover active support.
    2. Refitting via unpenalized OLS (or slight ridge stabilization) on the active subset,
       removing shrinkage bias while retaining sparsity.
    """
    def __init__(self, alpha=0.01, ridge_stability=1e-5, max_iter=2000, tol=1e-5):
        self.alpha = float(alpha)
        self.ridge_stability = ridge_stability
        self.lasso = LassoRegression(alpha=alpha, max_iter=max_iter, tol=tol)
        self.active_indices = None
        self.weights = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        # Stage 1: Run Lasso
        self.lasso.fit(X, y)
        active = self.lasso.active_features
        if 0 not in active:
            active = np.insert(active, 0, 0)
        self.active_indices = active
        
        # Stage 2: Fit OLS/Ridge on active features only
        X_sub = X[:, active]
        n_active = len(active)
        I = np.eye(n_active, dtype=np.float64)
        I[0, 0] = 0.0 # Don't penalize bias
        
        A = X_sub.T @ X_sub + self.ridge_stability * I
        b = X_sub.T @ y
        w_sub = np.linalg.solve(A, b)
        
        # Expand back to full feature space
        full_weights = np.zeros(X.shape[1], dtype=np.float64)
        full_weights[active] = w_sub
        self.weights = full_weights
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        return X @ self.weights
