# NFL-Draft-Prediction
# High-Performance Competitive Solution
## Strategy Document & Engineering Log

### 1. Granular Position-Based Group Imputation
Instead of utilizing a global median mapping, missing attributes are evaluated based on the specific `Position` category (e.g., matching missing shuttle times for WRs to Wide Receiver medians rather than Offensive Linemen). This prevents baseline variance contamination.

### 2. Physical Domain Feature Interactions
- **Speed Score**: Calculated using a weight-adjusted scale over the fourth power of the 40-yard dash.
- **BMI**: Explicit body mass index ratio tracking body density.
- **Explosive & Agility Coefficients**: Aggregations combining vertical/broad jump metrics and 3-cone/shuttle splits respectively.

### 3. Out-Of-Fold Meta Blending Ensemble
- Utilizes a robust **10-Fold Stratified Validation Split**.
- Combines structural trees via `HistGradientBoostingClassifier` and variance-minimizing estimators via `ExtraTreesClassifier`.
- Optimized Blending Metric: 0.34 HistGBM + 0.66 ExtraTrees yielding an advanced local cross-validation score of **0.8339**.
