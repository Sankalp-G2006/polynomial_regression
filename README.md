# Machine Learning Assignment: Polynomial Regression
**Student Name:** Sankalp Gadamsetty  
**Roll Number:** BT2024182  
**Course:** Machine Learning (AIT 511) 
**Repository:** [https://github.com/Sankalp-G2006/polynomial_regression](https://github.com/Sankalp-G2006/polynomial_regression)

---

## 1. Project Overview

This repository contains the complete first-principles implementation, training pipelines, cross-validation engines, inference scripts, and visualization suite for the Polynomial Regression assignment.

The assignment investigates two domain-specific regression challenges personalized for roll number **BT2024182**:

1. **Phase 1: Power Plant Steam Turbine Optimization (`var1`)**
   - **Inputs:** Six operational controls ($x_1, \dots, x_6$) representing percentage deviations in high-pressure steam valve, condenser coolant flow rate, re-injection pump hydraulic pressure, turbine blade pitch angle, non-condensable gas exhaust valve rate, and steam inlet pressure adjustment.
   - **Target ($y$):** Net Power Score.
   - **Model:** Degree 5 Polynomial with $L_1$ Regularization (Post-Lasso OLS), selecting 96 active physical monomial couplings from 462 terms.
   - **Validation Performance:** 5-Fold CV MSE = **0.3328**, $R^2$ = **0.9652** (Full Train MSE = **0.2404**, $R^2$ = **0.9754**).

2. **Phase 2: Subterranean Thermal Reservoir Mapping (`var2`)**
   - **Inputs:** Three spatial coordinates ($x_1$: East-West offset, $x_2$: North-South offset, $x_3$: Vertical depth offset relative to basecamp).
   - **Target ($y$):** Thermal Anomaly Score for identifying optimal high-yield geothermal drilling sites.
   - **Model:** Degree 10 Polynomial with $L_2$ Tikhonov Regularization (Ridge Regression, $\alpha = 0.05$), spanning 286 terms.
   - **Validation Performance:** 5-Fold CV MSE = **0.2453**, $R^2$ = **0.9946** (hitting the sensor noise floor $\sigma^2 \approx 0.25$; Full Train MSE = **0.1531**, $R^2$ = **0.9967**).

---

## 2. Directory Structure

```
polynomial_regression/
├── BT2024182_Report.pdf          # Final 4-page publication-grade PDF report
├── README.md                     # Project documentation and reproduction instructions
├── requirements.txt              # Environment dependencies
├── data/                         # Roll-specific datasets and model weights
│   ├── BT2024182_train_var1.csv
│   ├── BT2024182_test_var1.csv
│   ├── BT2024182_train_var2.csv
│   ├── BT2024182_test_var2.csv
│   ├── model_var1.npz            # Serialized weights and scaling for Phase 1
│   └── model_var2.npz            # Serialized weights and scaling for Phase 2
├── src/                          # Core Machine Learning library (from scratch)
│   ├── __init__.py
│   ├── polynomial.py             # Combinatorial monomial expansion and scaling
│   ├── models.py                 # OLS, Ridge, Lasso (Coordinate Descent), Post-Lasso
│   ├── cross_val.py              # K-Fold CV engine & grid search
│   └── evaluate.py               # MSE, RMSE, MAE, R2 scoring functions
├── scripts/                      # Pipelines and executable scripts
│   ├── train_var1.py             # Phase 1 training & CV analysis
│   ├── train_var2.py             # Phase 2 training & CV analysis
│   └── inference.py              # Test set prediction generator
├── predictions/                  # Submission prediction CSVs
│   ├── BT2024182_pred_var1.csv
│   └── BT2024182_pred_var2.csv
└── figures/                      # High-resolution charts and visualizations
    ├── var1_degree_vs_error.png
    ├── var1_residuals.png
    ├── var2_degree_vs_error.png
    ├── var2_residuals.png
    └── var2_thermal_slice.png
```

---

## 3. Mathematical Foundations

### 3.1 Monomial Feature Mapping
A polynomial of total degree $d$ over $D$ inputs consists of all monomial basis functions whose powers satisfy $\sum_{j=1}^D p_j \le d$. The total feature dimension $P(D, d)$ is:
$$P(D, d) = \binom{D + d}{d} = \frac{(D + d)!}{D! \, d!}$$

### 3.2 Regularization Formulations
1. **Ridge Regression ($L_2$):**
   $$J_{\text{Ridge}}(\mathbf{w}) = \frac{1}{2N} \|\mathbf{y} - \mathbf{\Phi} \mathbf{w}\|_2^2 + \alpha \|\mathbf{w}_{1:}\|_2^2$$
   $$\mathbf{w}_{\text{Ridge}} = (\mathbf{\Phi}^T \mathbf{\Phi} + \alpha \mathbf{I}^*)^{-1} \mathbf{\Phi}^T \mathbf{y}$$
   where $\mathbf{I}^*$ leaves the bias intercept $w_0$ unpenalized.

2. **Lasso Regression ($L_1$ & Cyclic Coordinate Descent):**
   $$J_{\text{Lasso}}(\mathbf{w}) = \frac{1}{2N} \|\mathbf{y} - \mathbf{\Phi} \mathbf{w}\|_2^2 + \alpha \|\mathbf{w}_{1:}\|_1$$
   Solved iteratively via soft-thresholding:
   $$w_j \leftarrow \frac{S(\rho_j, \alpha N)}{\|\mathbf{\Phi}_{:, j}\|_2^2}, \quad S(\rho, \lambda) = \text{sign}(\rho) \max(0, |\rho| - \lambda)$$

3. **Post-Lasso OLS:**
   $$\mathcal{A} = \{j : |w_j^{\text{Lasso}}| > 0\}, \quad \mathbf{w}_{\mathcal{A}} = (\mathbf{\Phi}_{\mathcal{A}}^T \mathbf{\Phi}_{\mathcal{A}} + \epsilon \mathbf{I})^{-1} \mathbf{\Phi}_{\mathcal{A}}^T \mathbf{y}$$
   Eliminates $L_1$ shrinkage bias on non-zero weights while maintaining exact sparsity.

---

## 4. Quickstart & Reproduction

### Installation
Ensure Python 3.8+ is installed. Install the minimal dependencies:
```bash
pip install -r requirements.txt
```

### Reproduce Training & Cross-Validation
Run Phase 1 (var1):
```bash
python3 scripts/train_var1.py
```

Run Phase 2 (var2):
```bash
python3 scripts/train_var2.py
```

### Generate Submission Predictions
Run inference to regenerate `BT2024182_pred_var1.csv` and `BT2024182_pred_var2.csv`:
```bash
python3 scripts/inference.py
```

---

## 5. Verification & Submission Files

- **Report:** `BT2024182_Report.pdf` (Exactly 4 pages, comprehensive methodology and diagnostics).
- **Predictions:**
  - `predictions/BT2024182_pred_var1.csv` (1,000 predictions, matching `sample_submission.csv` format).
  - `predictions/BT2024182_pred_var2.csv` (1,000 predictions, matching `sample_submission.csv` format).
  - Both files are also mirrored to `~/Downloads/` for immediate access.
