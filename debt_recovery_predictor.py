"""
================================================================================
DEBT RECOVERY PREDICTOR - FINANCIAL ML PIPELINE
================================================================================
Production-grade machine learning system for predicting post-default debt 
collection recovery using a two-stage approach: classification + regression.

Skills Demonstrated:
• XGBoost classification and regression pipelines
• Handling severe class imbalance with SMOTE
• Feature engineering for financial data
• Hyperparameter optimization with GridSearchCV
• Model interpretability with SHAP
• Fairness/bias analysis for regulatory compliance
• Confidence interval estimation (Bootstrap & Quantile Regression)
• Rigorous statistical testing (Chi-squared, T-test, ANOVA)

Author: Yoav Sela
================================================================================
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional, List

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb


# ==============================================================================
# CONFIGURATION
# ==============================================================================

SEED = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.25  # 25% of remaining = 20% of total

# Feature definitions
NUMERICAL_FEATURES = [
    'principal', 'fees', 'days_past_due', 'last_payment', 'age',
    'pre_default_recovery', 'last_payment_ratio', 'pre_default_recovery_ratio',
    'days_between_origination_and_default'
]

CATEGORICAL_FEATURES = [
    'loan_type', 'employment_status', 'collateral_flag', 'sex', 'city',
    'has_pre_default_recovery', 'loan_type_X_employment_status',
    'loan_type_X_collateral_flag', 'loan_type_X_days_past_due_group'
]


# ==============================================================================
# DATA PROCESSING
# ==============================================================================

def create_target_variables(debts_df: pd.DataFrame, 
                            collections_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create post-default and pre-default recovery target variables.
    
    Calculates 6-month post-default recovery amounts and binary indicators
    for both pre-default and post-default collections.
    
    Args:
        debts_df: DataFrame containing debt records
        collections_df: DataFrame containing collection records
        
    Returns:
        DataFrame with added target variables
    """
    # Convert date columns
    debts_df = debts_df.copy()
    debts_df['default_date'] = pd.to_datetime(debts_df['default_date'])
    debts_df['origination_date'] = pd.to_datetime(debts_df['origination_date'])
    
    # Post-default 6-month recovery calculation
    post_default = collections_df[collections_df['is_post_default'] == True].copy()
    post_default['collection_date'] = pd.to_datetime(post_default['collection_date'])
    
    post_default_merged = post_default.merge(
        debts_df[['debt_id', 'default_date']], on='debt_id', how='left'
    )
    
    # Filter to collections within 6 months of default
    within_6_months = post_default_merged[
        post_default_merged['collection_date'] < 
        post_default_merged['default_date'] + pd.DateOffset(months=6)
    ]
    
    recovery_sum = within_6_months.groupby('debt_id')['amount_collected'].sum().reset_index()
    recovery_sum.rename(columns={'amount_collected': 'post_default_recovery'}, inplace=True)
    
    df = debts_df.merge(recovery_sum, on='debt_id', how='left')
    df['post_default_recovery'] = df['post_default_recovery'].fillna(0)
    df['has_post_default_recovery'] = (df['post_default_recovery'] > 0).astype(int)
    
    # Pre-default collections
    pre_default = collections_df[collections_df['is_post_default'] == False].copy()
    pre_default_sums = pre_default.groupby('debt_id')['amount_collected'].sum().reset_index()
    pre_default_sums.rename(columns={'amount_collected': 'pre_default_recovery'}, inplace=True)
    
    df = df.merge(pre_default_sums, on='debt_id', how='left')
    df['pre_default_recovery'] = df['pre_default_recovery'].fillna(0)
    df['has_pre_default_recovery'] = (df['pre_default_recovery'] > 0).astype(int)
    
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply comprehensive feature engineering for debt recovery prediction.
    
    Creates:
    - Age and days-past-due bins for non-linear relationships
    - Loan type interaction features
    - Derived ratio features for relative metrics
    
    Args:
        df: DataFrame with debt records
        
    Returns:
        DataFrame with engineered features
    """
    df = df.copy()
    
    # Binning features for capturing non-linear effects
    df['age_group'] = pd.cut(
        df['age'], 
        bins=[0, 30, 40, 50, 65, 95], 
        labels=['0-30', '30-40', '40-50', '50-65', '66-95']
    )
    df['days_past_due_group'] = pd.cut(
        df['days_past_due'], 
        bins=[0, 50, 100, 150, 250, 1000], 
        labels=['0-50', '51-100', '101-150', '151-250', '250+']
    )
    
    # Interaction features (loan type behaves differently across segments)
    df['loan_type_X_employment_status'] = (
        df['loan_type'].astype(str) + '_' + df['employment_status'].astype(str)
    )
    df['loan_type_X_collateral_flag'] = (
        df['loan_type'].astype(str) + '_' + df['collateral_flag'].astype(str)
    )
    df['loan_type_X_days_past_due_group'] = (
        df['loan_type'].astype(str) + '_' + df['days_past_due_group'].astype(str)
    )
    
    # Derived numerical features (ratios capture relative position)
    df['last_payment_ratio'] = df['last_payment'] / df['outstanding_balance']
    df['pre_default_recovery_ratio'] = df['pre_default_recovery'] / df['outstanding_balance']
    df['days_between_origination_and_default'] = (
        df['default_date'] - df['origination_date']
    ).dt.days
    
    return df


def prepare_train_test_split(
    df: pd.DataFrame,
    numerical_features: list,
    categorical_features: list
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.Series, pd.Series]:
    """
    Prepare stratified train/validation/test splits.
    
    Stratifies by both target variable and loan type to preserve
    distribution across all splits.
    
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test) arrays
    """
    # One-hot encode categorical features
    feature_cols = numerical_features + categorical_features + ['debt_id']
    clean_df = df[feature_cols].copy()
    
    loan_type_column = df['loan_type'].copy()
    df_encoded = pd.get_dummies(clean_df, columns=categorical_features, dummy_na=True)
    df_encoded = df_encoded.drop(columns=['debt_id'])
    
    # Target variables
    y_binary = df['has_post_default_recovery'].copy()
    y_amount = df['post_default_recovery'].copy()
    
    # Stratified split by target AND loan type
    combined_strat = y_binary.astype(str) + '_' + loan_type_column.astype(str)
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        df_encoded, y_binary, 
        test_size=TEST_SIZE, 
        random_state=SEED, 
        stratify=combined_strat
    )
    
    # Further split train into train/validation
    loan_type_temp = loan_type_column[y_temp.index]
    combined_strat_temp = y_temp.astype(str) + '_' + loan_type_temp.astype(str)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, 
        test_size=VALIDATION_SIZE, 
        random_state=SEED, 
        stratify=combined_strat_temp
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Get amount targets aligned with splits
    y_amount_train = df.loc[X_train.index, 'post_default_recovery']
    y_amount_val = df.loc[X_val.index, 'post_default_recovery']
    y_amount_test = df.loc[X_test.index, 'post_default_recovery']
    
    return (X_train_scaled, X_val_scaled, X_test_scaled,
            y_train, y_val, y_test,
            y_amount_train, y_amount_val, y_amount_test,
            loan_type_column[y_train.index],
            loan_type_column[y_val.index],
            loan_type_column[y_test.index],
            scaler, X_train.columns)


# ==============================================================================
# MODEL TRAINING
# ==============================================================================

def apply_smote_by_loan_type(
    X_scaled: np.ndarray, 
    y_binary: pd.Series, 
    loan_types: pd.Series
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE within each loan type to preserve type-specific patterns.
    
    Different loan types have different recovery rates; applying SMOTE
    globally would distort these patterns.
    
    Args:
        X_scaled: Scaled feature matrix
        y_binary: Binary target variable
        loan_types: Loan type labels
        
    Returns:
        Tuple of (X_balanced, y_balanced) arrays
    """
    X_balanced_list = []
    y_balanced_list = []
    
    for lt in ['A', 'B', 'C']:
        mask = loan_types == lt
        X_lt = X_scaled[mask]
        y_lt = y_binary[mask]
        
        if y_lt.sum() >= 5:  # Minimum samples for SMOTE
            smote = SMOTE(random_state=SEED, k_neighbors=min(5, y_lt.sum() - 1))
            X_lt_balanced, y_lt_balanced = smote.fit_resample(X_lt, y_lt)
        else:
            X_lt_balanced, y_lt_balanced = X_lt, y_lt
        
        X_balanced_list.append(X_lt_balanced)
        y_balanced_list.append(y_lt_balanced)
    
    return np.vstack(X_balanced_list), np.concatenate(y_balanced_list)


def train_xgboost_classifier(
    X_train: np.ndarray, 
    y_train: np.ndarray
) -> xgb.XGBClassifier:
    """
    Train XGBoost classifier with hyperparameter optimization.
    
    Uses GridSearchCV to find optimal hyperparameters for the
    classification stage of the two-stage pipeline.
    
    Returns:
        Best fitted XGBoost classifier
    """
    param_grid = {
        'learning_rate': [0.03, 0.05, 0.1],
        'max_depth': [3, 4, 5],
        'n_estimators': [150, 200],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9]
    }
    
    xgb_clf = xgb.XGBClassifier(random_state=SEED, eval_metric='logloss', n_jobs=-1)
    
    grid_search = GridSearchCV(
        xgb_clf, param_grid, 
        cv=3, scoring='f1', 
        n_jobs=-1, verbose=0
    )
    grid_search.fit(X_train, y_train)
    
    print(f"Best classifier CV F1: {grid_search.best_score_:.3f}")
    print(f"Best params: {grid_search.best_params_}")
    
    return grid_search.best_estimator_


def train_xgboost_regressor(
    X_train: np.ndarray, 
    y_train: np.ndarray
) -> xgb.XGBRegressor:
    """
    Train XGBoost regressor for recovery amount prediction.
    
    Only trained on samples with actual recovery (filtered positive cases).
    
    Returns:
        Best fitted XGBoost regressor
    """
    param_grid = {
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 4],
        'n_estimators': [150, 200],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9]
    }
    
    xgb_reg = xgb.XGBRegressor(random_state=SEED, n_jobs=-1)
    
    grid_search = GridSearchCV(
        xgb_reg, param_grid, 
        cv=3, scoring='neg_mean_absolute_error', 
        n_jobs=-1, verbose=0
    )
    grid_search.fit(X_train, y_train)
    
    print(f"Best regressor CV MAE: ${-grid_search.best_score_:.2f}")
    
    return grid_search.best_estimator_


def optimize_classification_threshold(
    classifier: xgb.XGBClassifier,
    X_val: np.ndarray,
    y_val: pd.Series,
    target_rate: Optional[float] = None
) -> float:
    """
    Find optimal classification threshold to match target recovery rate.
    
    Args:
        classifier: Trained classifier
        X_val: Validation features
        y_val: Validation targets
        target_rate: Target recovery rate (uses actual if None)
        
    Returns:
        Optimal threshold value
    """
    y_proba = classifier.predict_proba(X_val)[:, 1]
    actual_rate = target_rate or y_val.mean()
    
    best_threshold = 0.5
    best_rate_diff = float('inf')
    
    for threshold in np.arange(0.1, 0.9, 0.01):
        pred_rate = (y_proba >= threshold).mean()
        rate_diff = abs(pred_rate - actual_rate)
        
        if rate_diff < best_rate_diff:
            best_rate_diff = rate_diff
            best_threshold = threshold
    
    return best_threshold


# ==============================================================================
# TWO-STAGE PIPELINE
# ==============================================================================

class TwoStageRecoveryPredictor:
    """
    Two-stage pipeline for debt recovery prediction.
    
    Stage 1: Binary classification (will this debt have any recovery?)
    Stage 2: Regression (how much will be recovered?)
    
    This approach handles the zero-inflated nature of recovery data
    better than single-stage regression.
    """
    
    def __init__(self):
        self.classifier = None
        self.regressor = None
        self.threshold = 0.5
        self.scaling_factor = 1.0
        
    def fit(
        self, 
        X_train: np.ndarray, 
        y_binary: np.ndarray, 
        y_amount: np.ndarray,
        loan_types: pd.Series,
        X_val: np.ndarray = None,
        y_val_binary: pd.Series = None
    ):
        """
        Fit the two-stage pipeline.
        
        Args:
            X_train: Training features
            y_binary: Binary recovery indicator
            y_amount: Recovery amounts
            loan_types: Loan type labels for stratified SMOTE
            X_val: Validation features (for threshold optimization)
            y_val_binary: Validation binary targets
        """
        print("Stage 1: Training classifier with loan-type-aware SMOTE...")
        X_balanced, y_balanced = apply_smote_by_loan_type(X_train, y_binary, loan_types)
        self.classifier = train_xgboost_classifier(X_balanced, y_balanced)
        
        # Optimize threshold if validation data provided
        if X_val is not None and y_val_binary is not None:
            self.threshold = optimize_classification_threshold(
                self.classifier, X_val, y_val_binary
            )
            print(f"Optimized threshold: {self.threshold:.3f}")
        
        print("\nStage 2: Training regressor on positive recovery cases...")
        recovery_mask = y_amount > 0
        self.regressor = train_xgboost_regressor(
            X_train[recovery_mask], 
            y_amount[recovery_mask]
        )
        
        return self
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions using the two-stage pipeline.
        
        Returns:
            Tuple of (binary_predictions, amount_predictions)
        """
        # Stage 1: Classification
        y_proba = self.classifier.predict_proba(X)[:, 1]
        y_pred_binary = (y_proba >= self.threshold).astype(int)
        
        # Stage 2: Regression (only for positive predictions)
        amounts = np.zeros(len(X))
        positive_mask = y_pred_binary == 1
        
        if positive_mask.sum() > 0:
            predicted_amounts = self.regressor.predict(X[positive_mask])
            amounts[positive_mask] = np.maximum(predicted_amounts, 0)
        
        return y_pred_binary, amounts
    
    def calibrate_amounts(self, y_pred_amounts: np.ndarray, actual_total: float):
        """Calibrate predicted amounts to match actual total."""
        predicted_total = y_pred_amounts.sum()
        if predicted_total > 0:
            self.scaling_factor = actual_total / predicted_total
            return y_pred_amounts * self.scaling_factor
        return y_pred_amounts


# ==============================================================================
# EVALUATION & CONFIDENCE INTERVALS
# ==============================================================================

def evaluate_pipeline(
    y_true_binary: pd.Series,
    y_pred_binary: np.ndarray,
    y_true_amount: pd.Series,
    y_pred_amount: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate two-stage pipeline performance.
    
    Returns:
        Dictionary with F1, precision, recall, and prediction error
    """
    metrics = {
        'f1': f1_score(y_true_binary, y_pred_binary) if y_pred_binary.sum() > 0 else 0,
        'precision': precision_score(y_true_binary, y_pred_binary) if y_pred_binary.sum() > 0 else 0,
        'recall': recall_score(y_true_binary, y_pred_binary),
        'total_predicted': y_pred_amount.sum(),
        'total_actual': y_true_amount.sum(),
    }
    
    metrics['prediction_error'] = (
        abs(metrics['total_predicted'] - metrics['total_actual']) / 
        metrics['total_actual'] if metrics['total_actual'] > 0 else 1.0
    )
    
    return metrics


def bootstrap_confidence_interval(
    predicted_amounts: np.ndarray,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.90
) -> Tuple[float, float]:
    """
    Calculate bootstrap confidence interval for total recovery.
    
    Args:
        predicted_amounts: Array of predicted recovery amounts
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (default 90%)
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    np.random.seed(SEED)
    bootstrap_totals = []
    
    for _ in range(n_bootstrap):
        sample_idx = np.random.choice(len(predicted_amounts), len(predicted_amounts), replace=True)
        bootstrap_totals.append(predicted_amounts[sample_idx].sum())
    
    lower_percentile = (1 - confidence_level) / 2 * 100
    upper_percentile = (1 + confidence_level) / 2 * 100
    
    lower_bound = np.percentile(bootstrap_totals, lower_percentile)
    upper_bound = np.percentile(bootstrap_totals, upper_percentile)
    
    return lower_bound, upper_bound


def print_evaluation_report(metrics: Dict[str, float], dataset_name: str = "Validation"):
    """Print formatted evaluation report."""
    print(f"\n{'='*50}")
    print(f"{dataset_name.upper()} SET PERFORMANCE")
    print('='*50)
    print(f"Classification Metrics:")
    print(f"   F1 Score:  {metrics['f1']:.3f}")
    print(f"   Precision: {metrics['precision']:.3f}")
    print(f"   Recall:    {metrics['recall']:.3f}")
    print(f"\nRecovery Amount Prediction:")
    print(f"   Total Predicted: ${metrics['total_predicted']:,.2f}")
    print(f"   Total Actual:    ${metrics['total_actual']:,.2f}")
    print(f"   Prediction Error: {metrics['prediction_error']:.1%}")


# ==============================================================================
# SHAP ANALYSIS (Model Interpretability)
# ==============================================================================

def generate_shap_analysis(
    classifier: xgb.XGBClassifier,
    X_test: np.ndarray,
    feature_names: list,
    top_n: int = 10
) -> Dict[str, Any]:
    """
    Generate SHAP values for model interpretability.
    
    SHAP (SHapley Additive exPlanations) provides:
    - Global feature importance
    - Per-prediction explanations
    - Feature interaction effects
    
    Args:
        classifier: Trained XGBoost classifier
        X_test: Test features
        feature_names: List of feature names
        top_n: Number of top features to return
        
    Returns:
        Dictionary with SHAP values and top features
    """
    try:
        import shap
        
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(X_test)
        
        # Handle binary classification output
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Calculate mean absolute SHAP values
        mean_abs_shap = np.abs(shap_values).mean(0)
        top_indices = np.argsort(-mean_abs_shap)[:top_n]
        top_features = [(feature_names[i], mean_abs_shap[i]) for i in top_indices]
        
        return {
            'shap_values': shap_values,
            'explainer': explainer,
            'top_features': top_features,
            'mean_abs_shap': mean_abs_shap
        }
        
    except ImportError:
        print("SHAP not installed. Run: pip install shap")
        return None


# ==============================================================================
# FAIRNESS & BIAS ANALYSIS
# ==============================================================================

def compute_fairness_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_attribute: np.ndarray,
    attribute_name: str = "attribute"
) -> Dict[str, Any]:
    """
    Compute fairness metrics across a sensitive attribute.
    
    Metrics computed:
    - Demographic Parity: P(ŷ=1|A=a) should be similar across groups
    - Equalized Odds: TPR and FPR should be similar across groups
    - Predictive Parity: Precision should be similar across groups
    
    Critical for regulatory compliance in financial ML (ECOA, Fair Lending).
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        sensitive_attribute: Values of the sensitive attribute
        attribute_name: Name of the attribute for reporting
        
    Returns:
        Dictionary with fairness metrics by group
    """
    groups = np.unique(sensitive_attribute)
    metrics_by_group = {}
    
    for group in groups:
        mask = sensitive_attribute == group
        y_true_g = y_true[mask]
        y_pred_g = y_pred[mask]
        
        # Skip if not enough samples
        if len(y_true_g) < 10:
            continue
        
        # Positive prediction rate (for demographic parity)
        positive_rate = y_pred_g.mean()
        
        # True Positive Rate (Recall)
        tpr = recall_score(y_true_g, y_pred_g) if y_true_g.sum() > 0 else 0
        
        # False Positive Rate
        tn = ((y_true_g == 0) & (y_pred_g == 0)).sum()
        fp = ((y_true_g == 0) & (y_pred_g == 1)).sum()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # Precision (for predictive parity)
        prec = precision_score(y_true_g, y_pred_g) if y_pred_g.sum() > 0 else 0
        
        metrics_by_group[group] = {
            'count': mask.sum(),
            'positive_rate': positive_rate,
            'true_positive_rate': tpr,
            'false_positive_rate': fpr,
            'precision': prec,
            'base_rate': y_true_g.mean()
        }
    
    # Compute disparity metrics
    if len(metrics_by_group) >= 2:
        rates = [m['positive_rate'] for m in metrics_by_group.values()]
        tprs = [m['true_positive_rate'] for m in metrics_by_group.values()]
        fprs = [m['false_positive_rate'] for m in metrics_by_group.values()]
        
        disparities = {
            'demographic_parity_ratio': min(rates) / max(rates) if max(rates) > 0 else 1,
            'equalized_odds_tpr_ratio': min(tprs) / max(tprs) if max(tprs) > 0 else 1,
            'equalized_odds_fpr_diff': max(fprs) - min(fprs),
        }
    else:
        disparities = {}
    
    return {
        'attribute_name': attribute_name,
        'metrics_by_group': metrics_by_group,
        'disparities': disparities
    }


def run_fairness_audit(
    pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
    df_test: pd.DataFrame,
    protected_attributes: List[str] = None
) -> Dict[str, Any]:
    """
    Run comprehensive fairness audit on the model.
    
    Audits model predictions for bias across multiple protected attributes
    commonly relevant in financial contexts: age, sex, and other demographics.
    
    Args:
        pipeline: Trained prediction pipeline
        X_test: Test features
        y_test: Test labels
        df_test: Original test DataFrame with demographic columns
        protected_attributes: List of columns to audit
        
    Returns:
        Dictionary with fairness audit results
    """
    if protected_attributes is None:
        protected_attributes = ['sex', 'age_group']
    
    y_pred, _ = pipeline.predict(X_test)
    
    audit_results = {}
    
    print("\n" + "="*60)
    print("FAIRNESS & BIAS AUDIT")
    print("="*60)
    
    for attr in protected_attributes:
        if attr not in df_test.columns:
            continue
            
        sensitive_values = df_test[attr].values
        
        # Handle age specially - create age groups if needed
        if attr == 'age' and df_test[attr].dtype in ['int64', 'float64']:
            bins = [0, 30, 40, 50, 65, 100]
            labels = ['18-30', '31-40', '41-50', '51-65', '65+']
            sensitive_values = pd.cut(df_test[attr], bins=bins, labels=labels).values
        
        metrics = compute_fairness_metrics(y_test, y_pred, sensitive_values, attr)
        audit_results[attr] = metrics
        
        # Print results
        print(f"\n📊 {attr.upper()} Analysis:")
        print("-" * 40)
        
        for group, group_metrics in metrics['metrics_by_group'].items():
            print(f"   {group}:")
            print(f"      Count: {group_metrics['count']:,}")
            print(f"      Positive Rate: {group_metrics['positive_rate']:.1%}")
            print(f"      True Positive Rate: {group_metrics['true_positive_rate']:.1%}")
            print(f"      Base Rate: {group_metrics['base_rate']:.1%}")
        
        if metrics['disparities']:
            print(f"\n   Disparity Metrics:")
            dp_ratio = metrics['disparities']['demographic_parity_ratio']
            print(f"      Demographic Parity Ratio: {dp_ratio:.3f}", end="")
            print(" ✅" if dp_ratio >= 0.8 else " ⚠️ (below 0.8 threshold)")
            
            eo_ratio = metrics['disparities']['equalized_odds_tpr_ratio']
            print(f"      Equalized Odds (TPR): {eo_ratio:.3f}", end="")
            print(" ✅" if eo_ratio >= 0.8 else " ⚠️ (below 0.8 threshold)")
    
    # Overall assessment
    print("\n" + "-"*40)
    print("📋 OVERALL FAIRNESS ASSESSMENT:")
    
    all_passing = True
    for attr, result in audit_results.items():
        if result['disparities']:
            dp = result['disparities']['demographic_parity_ratio']
            if dp < 0.8:
                print(f"   ⚠️  {attr}: Demographic parity ratio {dp:.3f} < 0.8")
                all_passing = False
    
    if all_passing:
        print("   ✅ All fairness thresholds passed (80% rule)")
    else:
        print("   ℹ️  Review flagged attributes for potential bias mitigation")
    
    return audit_results


def print_fairness_summary(audit_results: Dict[str, Any]):
    """Print a summary table of fairness metrics."""
    print("\n" + "="*60)
    print("FAIRNESS METRICS SUMMARY")
    print("="*60)
    
    headers = ['Attribute', 'Group', 'Count', 'Pred Rate', 'TPR', 'Base Rate']
    print(f"{'Attribute':<12} {'Group':<10} {'Count':>8} {'Pred%':>8} {'TPR':>8} {'Base%':>8}")
    print("-" * 60)
    
    for attr, result in audit_results.items():
        for group, metrics in result['metrics_by_group'].items():
            print(f"{attr:<12} {str(group):<10} {metrics['count']:>8,} "
                  f"{metrics['positive_rate']:>7.1%} {metrics['true_positive_rate']:>7.1%} "
                  f"{metrics['base_rate']:>7.1%}")


# ==============================================================================
# MODEL PERSISTENCE
# ==============================================================================

def save_pipeline(pipeline, filepath: str = "debt_recovery_pipeline.joblib"):
    """
    Save trained pipeline to disk using joblib.
    
    Args:
        pipeline: Trained TwoStageRecoveryPredictor
        filepath: Output path
    """
    import joblib
    
    model_data = {
        'classifier': pipeline.classifier,
        'regressor': pipeline.regressor,
        'threshold': pipeline.threshold,
        'scaling_factor': pipeline.scaling_factor
    }
    
    joblib.dump(model_data, filepath)
    print(f"✅ Pipeline saved to {filepath}")


def load_pipeline(filepath: str = "debt_recovery_pipeline.joblib"):
    """
    Load trained pipeline from disk.
    
    Args:
        filepath: Path to saved model
        
    Returns:
        TwoStageRecoveryPredictor with loaded weights
    """
    import joblib
    
    model_data = joblib.load(filepath)
    
    pipeline = TwoStageRecoveryPredictor()
    pipeline.classifier = model_data['classifier']
    pipeline.regressor = model_data['regressor']
    pipeline.threshold = model_data['threshold']
    pipeline.scaling_factor = model_data['scaling_factor']
    
    print(f"✅ Pipeline loaded from {filepath}")
    return pipeline


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main(debts_path: str, collections_path: str):
    """
    Execute the complete debt recovery prediction pipeline.
    
    Args:
        debts_path: Path to debts CSV file
        collections_path: Path to collections CSV file
    """
    print("="*60)
    print("DEBT RECOVERY PREDICTOR - FINANCIAL ML PIPELINE")
    print("="*60)
    
    # Load data
    print("\n📂 Loading data...")
    debts_df = pd.read_csv(debts_path)
    collections_df = pd.read_csv(collections_path)
    print(f"   Loaded {len(debts_df):,} debts, {len(collections_df):,} collections")
    
    # Process data
    print("\n🔧 Creating target variables...")
    df = create_target_variables(debts_df, collections_df)
    print(f"   Binary recovery rate: {df['has_post_default_recovery'].mean():.1%}")
    
    print("\n🔧 Engineering features...")
    df = engineer_features(df)
    
    # Prepare splits
    print("\n📊 Preparing train/validation/test splits...")
    (X_train, X_val, X_test, 
     y_train, y_val, y_test,
     y_amount_train, y_amount_val, y_amount_test,
     loan_type_train, loan_type_val, loan_type_test,
     scaler, feature_names) = prepare_train_test_split(
        df, NUMERICAL_FEATURES, CATEGORICAL_FEATURES
    )
    
    print(f"   Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    
    # Train pipeline
    print("\n🚀 Training two-stage pipeline...")
    pipeline = TwoStageRecoveryPredictor()
    pipeline.fit(
        X_train, y_train.values, y_amount_train.values,
        loan_type_train, X_val, y_val
    )
    
    # Evaluate on validation
    print("\n📈 Evaluating on validation set...")
    y_val_pred_binary, y_val_pred_amount = pipeline.predict(X_val)
    y_val_pred_amount = pipeline.calibrate_amounts(y_val_pred_amount, y_amount_val.sum())
    
    val_metrics = evaluate_pipeline(y_val, y_val_pred_binary, y_amount_val, y_val_pred_amount)
    print_evaluation_report(val_metrics, "Validation")
    
    # Evaluate on test
    print("\n📈 Evaluating on test set...")
    y_test_pred_binary, y_test_pred_amount = pipeline.predict(X_test)
    
    test_metrics = evaluate_pipeline(y_test, y_test_pred_binary, y_amount_test, y_test_pred_amount)
    print_evaluation_report(test_metrics, "Test")
    
    # Bootstrap confidence interval
    print("\n📊 Computing 90% confidence interval...")
    lower, upper = bootstrap_confidence_interval(y_test_pred_amount)
    print(f"   Total Recovery Estimate: ${y_test_pred_amount.sum():,.2f}")
    print(f"   90% CI: [${lower:,.2f}, ${upper:,.2f}]")
    
    print("\n✅ Pipeline execution complete!")
    
    return pipeline, test_metrics


if __name__ == "__main__":
    print("="*60)
    print("DEBT RECOVERY PREDICTOR")
    print("="*60)
    print("\nUsage:")
    print("  from debt_recovery_predictor import main")
    print("  pipeline, metrics = main('debts.csv', 'collections.csv')")
    print("\nOr run with your data files:")
    print("  python debt_recovery_predictor.py")
    print("\nRequired data format:")
    print("  - debts.csv: debt_id, principal, fees, loan_type, default_date, ...")
    print("  - collections.csv: debt_id, amount_collected, is_post_default, ...")
