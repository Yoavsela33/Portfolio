"""
================================================================================
MATCH OUTCOME PREDICTOR - COMPETITIVE GAMING ANALYTICS
================================================================================
Machine learning system for predicting 1v1 match outcomes in competitive gaming
using player history, deck performance, ELO ratings, and head-to-head features.

Skills Demonstrated:
• Time-series aware feature engineering (no data leakage)
• ELO/Glicko rating system implementation
• Rolling window statistics for player performance
• Antisymmetric feature design for pairwise prediction
• Probability calibration for well-calibrated outputs
• Model comparison (Logistic Regression, Random Forest, XGBoost)
• Pairwise evaluation metrics

Author: Yoav Sela
================================================================================
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List

from sklearn.metrics import (precision_score, recall_score, f1_score, 
                            roc_auc_score, log_loss)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


# ==============================================================================
# CONFIGURATION
# ==============================================================================

SEED = 42
PLAYER_HISTORY_WINDOW = 10  # Number of recent games for player stats
PLAYER_DECK_HISTORY_WINDOW = 5  # Number of recent games per deck
TRAIN_SPLIT = 0.6
VAL_SPLIT = 0.8  # 60-20-20 split

# ELO Rating Parameters
ELO_INITIAL = 1500  # Starting rating for new players
ELO_K_FACTOR = 32   # Rating adjustment factor (higher = more volatile)

np.random.seed(SEED)


# ==============================================================================
# ELO RATING SYSTEM
# ==============================================================================

class ELORatingSystem:
    """
    Implementation of the ELO rating system for competitive gaming.
    
    The ELO system provides a skill-based rating that updates after each match.
    Originally developed for chess, it's now standard in competitive gaming.
    
    Key properties:
    - Zero-sum: Winner gains what loser loses
    - Expected outcome based: Updates proportional to surprise factor
    - Convergent: Ratings stabilize as more games are played
    
    Formula:
        E_A = 1 / (1 + 10^((R_B - R_A) / 400))  # Expected win prob for A
        R_A_new = R_A + K * (S_A - E_A)          # New rating for A
    """
    
    def __init__(self, k_factor: int = ELO_K_FACTOR, initial_rating: int = ELO_INITIAL):
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        self.ratings: Dict[str, float] = {}
        self.rating_history: Dict[str, List[Tuple[str, float]]] = {}  # player -> [(timestamp, rating)]
    
    def get_rating(self, player_id: str) -> float:
        """Get current rating for a player (initializes if new)."""
        if player_id not in self.ratings:
            self.ratings[player_id] = self.initial_rating
            self.rating_history[player_id] = []
        return self.ratings[player_id]
    
    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """
        Calculate expected score (win probability) for player A.
        
        Uses the standard ELO formula with 400-point scale.
        """
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))
    
    def update_ratings(
        self, 
        player_a: str, 
        player_b: str, 
        winner: str,
        timestamp: str = None
    ) -> Tuple[float, float]:
        """
        Update ratings after a match.
        
        Args:
            player_a: First player ID
            player_b: Second player ID  
            winner: ID of winning player (or None for draw)
            timestamp: Match timestamp for history tracking
            
        Returns:
            Tuple of (new_rating_a, new_rating_b)
        """
        rating_a = self.get_rating(player_a)
        rating_b = self.get_rating(player_b)
        
        expected_a = self.expected_score(rating_a, rating_b)
        expected_b = 1 - expected_a
        
        # Actual scores (1 for win, 0.5 for draw, 0 for loss)
        if winner == player_a:
            score_a, score_b = 1.0, 0.0
        elif winner == player_b:
            score_a, score_b = 0.0, 1.0
        else:  # Draw
            score_a, score_b = 0.5, 0.5
        
        # Update ratings
        new_rating_a = rating_a + self.k_factor * (score_a - expected_a)
        new_rating_b = rating_b + self.k_factor * (score_b - expected_b)
        
        self.ratings[player_a] = new_rating_a
        self.ratings[player_b] = new_rating_b
        
        # Track history
        if timestamp:
            self.rating_history[player_a].append((timestamp, new_rating_a))
            self.rating_history[player_b].append((timestamp, new_rating_b))
        
        return new_rating_a, new_rating_b
    
    def get_win_probability(self, player_a: str, player_b: str) -> float:
        """Get predicted win probability for player A vs player B."""
        rating_a = self.get_rating(player_a)
        rating_b = self.get_rating(player_b)
        return self.expected_score(rating_a, rating_b)


def compute_elo_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute ELO ratings for all players through match history.
    
    Processes matches chronologically to update ratings and assigns
    pre-match ratings to each row (no leakage - uses rating before match).
    
    Args:
        df: DataFrame sorted by match_timestamp with player_id, match_id, is_winner
        
    Returns:
        DataFrame with ELO features added
    """
    elo_system = ELORatingSystem()
    
    # Initialize columns
    df = df.copy()
    df['player_elo_before'] = ELO_INITIAL
    df['player_elo_expected_win'] = 0.5
    
    # Group by match to process each match once
    matches = df.groupby('match_id').first().reset_index()[['match_id', 'match_timestamp']]
    matches = matches.sort_values('match_timestamp')
    
    # Process each match chronologically
    for _, match_row in matches.iterrows():
        match_id = match_row['match_id']
        timestamp = str(match_row['match_timestamp'])
        
        # Get players in this match
        match_data = df[df['match_id'] == match_id]
        if len(match_data) != 2:
            continue
            
        players = match_data['player_id'].values
        player_a, player_b = players[0], players[1]
        
        # Get pre-match ratings
        rating_a = elo_system.get_rating(player_a)
        rating_b = elo_system.get_rating(player_b)
        
        # Calculate expected win probabilities
        expected_a = elo_system.expected_score(rating_a, rating_b)
        expected_b = 1 - expected_a
        
        # Assign pre-match ratings to dataframe
        mask_a = (df['match_id'] == match_id) & (df['player_id'] == player_a)
        mask_b = (df['match_id'] == match_id) & (df['player_id'] == player_b)
        
        df.loc[mask_a, 'player_elo_before'] = rating_a
        df.loc[mask_b, 'player_elo_before'] = rating_b
        df.loc[mask_a, 'player_elo_expected_win'] = expected_a
        df.loc[mask_b, 'player_elo_expected_win'] = expected_b
        
        # Determine winner and update ratings
        winner_mask = match_data['is_winner'] == 1
        if winner_mask.any():
            winner = match_data.loc[winner_mask, 'player_id'].values[0]
        else:
            winner = None  # Draw
            
        elo_system.update_ratings(player_a, player_b, winner, timestamp)
    
    # Create derived ELO features
    df['player_elo_above_avg'] = df['player_elo_before'] - ELO_INITIAL
    
    return df


# ==============================================================================
# FEATURE ENGINEERING
# ==============================================================================

def create_base_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create base features from raw match data.
    
    Prepares data for feature engineering by:
    - Converting timestamps
    - Sorting chronologically
    - Creating win indicator
    
    Args:
        data: Raw match data with player_id, match_id, deck_id, score
        
    Returns:
        DataFrame sorted by time with win indicator
    """
    df = data.copy()
    df['match_timestamp'] = pd.to_datetime(df['match_timestamp'])
    df = df.sort_values('match_timestamp')
    
    # Winner is the player with higher score (ties: neither wins)
    df['is_winner'] = (
        df['score'] != df.groupby('match_id')['score'].transform('min')
    ).astype(int)
    
    return df


def create_player_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create player-level historical performance features.
    
    Uses rolling windows to calculate:
    - Global play count
    - Recent win count and win rate
    - Recent average score
    
    All features use only past data (shift) to prevent leakage.
    
    Args:
        df: DataFrame sorted by match_timestamp
        
    Returns:
        DataFrame with player history features
    """
    # Global experience (total games played before this one)
    df['player_prev_plays_global'] = df.groupby('player_id').cumcount()
    
    # Capped version for recent performance
    df['player_prev_plays'] = df['player_prev_plays_global'].clip(upper=PLAYER_HISTORY_WINDOW)
    
    # Rolling win count (last N games)
    df['player_prev_wins'] = (
        df.groupby('player_id')['is_winner']
          .transform(lambda s: s.shift().rolling(PLAYER_HISTORY_WINDOW, min_periods=1).sum())
          .fillna(0)
          .astype(int)
    )
    
    # Win rate
    df['player_win_rate'] = (
        (df['player_prev_wins'] / df['player_prev_plays'].replace(0, np.nan))
        .fillna(0)
    )
    
    # Rolling average score
    df['player_prev_mean_score'] = (
        df.groupby('player_id')['score']
          .transform(lambda s: s.shift().rolling(PLAYER_HISTORY_WINDOW, min_periods=1).mean())
    )
    
    return df


def create_player_deck_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create player-deck interaction features.
    
    Key insight: A player's performance varies by deck. These features
    capture how well a player performs with their specific deck choice.
    
    Features include:
    - Deck's global average score
    - Player's performance on this specific deck
    - Relative performance (player vs average for this deck)
    
    Args:
        df: DataFrame with player history features
        
    Returns:
        DataFrame with player-deck interaction features
    """
    # Global deck performance (how well does this deck perform on average?)
    df['deck_prev_mean_score'] = (
        df.groupby('deck_id')['score']
          .transform(lambda s: s.shift().expanding().mean())
    )
    
    # Player-deck specific experience
    df['pd_prev_plays_global'] = df.groupby(['player_id', 'deck_id']).cumcount()
    df['pd_prev_plays'] = df['pd_prev_plays_global'].clip(upper=PLAYER_DECK_HISTORY_WINDOW)
    
    # Player-deck specific performance
    df['pd_prev_mean_score'] = (
        df.groupby(['player_id', 'deck_id'])['score']
          .transform(lambda s: s.shift().rolling(PLAYER_DECK_HISTORY_WINDOW, min_periods=1).mean())
    )
    
    df['pd_prev_win_rate'] = (
        df.groupby(['player_id', 'deck_id'])['is_winner']
          .transform(lambda s: s.shift().rolling(PLAYER_DECK_HISTORY_WINDOW, min_periods=1).mean())
    )
    
    # Relative performance: how does player perform on this deck vs average?
    eps = 1e-9
    ratio = (df['pd_prev_mean_score'] / (df['deck_prev_mean_score'] + eps)).replace(
        [np.inf, -np.inf], np.nan
    )
    
    df['pd_prev_relative_score'] = (
        ratio.groupby([df['player_id'], df['deck_id']])
             .transform(lambda s: s.shift().expanding().mean())
             .fillna(1.0)
    )
    
    return df


def create_player_general_relative_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create player's general relative score across all decks.
    
    For each player, looks at their most recent game with each deck
    and averages their relative performance. This captures overall
    skill level independent of current deck choice.
    
    Args:
        df: DataFrame with player-deck features
        
    Returns:
        DataFrame with general relative score feature
    """
    def player_general_last_ratio(grp):
        out = []
        for i in range(len(grp)):
            past = grp.iloc[:i]
            if past.empty:
                out.append(1.0)
                continue
            # Get most recent game per deck
            idx = past.groupby('deck_id')['match_timestamp'].idxmax()
            out.append(past.loc[idx, 'pd_prev_relative_score'].mean())
        return pd.Series(out, index=grp.index)
    
    df['player_general_relative_score'] = (
        df.groupby('player_id', group_keys=False).apply(player_general_last_ratio)
    )
    
    return df


def merge_opponent_features(df: pd.DataFrame, player_features: List[str]) -> pd.DataFrame:
    """
    Merge opponent's features into each row.
    
    For 1v1 matches, each player's prediction should consider their
    opponent's statistics. This creates _opp suffixed features.
    
    Args:
        df: DataFrame with all player features
        player_features: List of feature column names to merge
        
    Returns:
        DataFrame with opponent features added
    """
    opp_cols = ['match_id', 'player_id'] + player_features
    
    # Self-merge to get opponent data
    tmp = df[opp_cols].merge(
        df[opp_cols], 
        on='match_id', 
        suffixes=('', '_opp')
    )
    
    # Remove self-matches (player paired with themselves)
    tmp = tmp[tmp['player_id'] != tmp['player_id_opp']]
    
    # Keep only opponent features
    opp_feature_cols = ['match_id', 'player_id'] + [f'{c}_opp' for c in player_features]
    df = df.merge(tmp[opp_feature_cols], on=['match_id', 'player_id'], how='left')
    
    return df


def create_antisymmetric_features(
    df: pd.DataFrame, 
    player_features: List[str]
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create antisymmetric (delta) features: player - opponent.
    
    Key insight: For pairwise prediction, absolute values matter less
    than relative differences. If Player A has 70% win rate and Player B
    has 60%, the delta (+10%) is more predictive than either alone.
    
    Antisymmetric features also ensure that P(A wins) + P(B wins) = 1
    when using a model that respects symmetry.
    
    Args:
        df: DataFrame with player and opponent features
        player_features: List of player feature names
        
    Returns:
        Tuple of (DataFrame with delta features, list of delta feature names)
    """
    meta_cols = {'player_id', 'match_id', 'deck_id', 'match_timestamp'}
    label_cols = {'is_winner', 'score', 'rank_in_match', 'max_score_in_match'}
    
    # Only create deltas for numeric features that aren't meta/labels
    clean_features = [c for c in player_features if c not in meta_cols | label_cols]
    
    delta_features = []
    for c in clean_features:
        opp_col = f'{c}_opp'
        if opp_col in df.columns:
            delta_col = f'diff_{c}'
            df[delta_col] = df[c] - df[opp_col]
            delta_features.append(delta_col)
    
    return df, delta_features


def engineer_all_features(data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Execute complete feature engineering pipeline.
    
    Pipeline steps:
    1. Create base features (timestamps, win indicator)
    2. Compute ELO ratings (skill-based ranking)
    3. Player historical performance
    4. Player-deck interaction features
    5. General relative score
    6. Merge opponent features
    7. Create antisymmetric deltas
    
    Args:
        data: Raw match data
        
    Returns:
        Tuple of (processed DataFrame, list of feature column names)
    """
    print("   Creating base features...")
    df = create_base_features(data)
    
    print("   Computing ELO ratings...")
    df = compute_elo_features(df)
    
    print("   Creating player history features...")
    df = create_player_history_features(df)
    
    print("   Creating player-deck features...")
    df = create_player_deck_features(df)
    
    print("   Creating general relative score...")
    df = create_player_general_relative_score(df)
    
    # Identify player-side features for opponent merge
    meta_cols = ['player_id', 'match_id', 'deck_id', 'match_timestamp']
    exclude_cols = meta_cols + ['is_winner']
    player_features = [c for c in df.columns if c not in exclude_cols]
    
    print("   Merging opponent features...")
    df = merge_opponent_features(df, player_features)
    
    print("   Creating antisymmetric features...")
    df, delta_features = create_antisymmetric_features(df, player_features)
    
    # Final feature set: deltas + experience indicators + ELO
    X_features = delta_features + ['player_prev_plays_global', 'pd_prev_plays_global', 
                                   'player_elo_expected_win']
    
    return df, X_features


# ==============================================================================
# DATA PREPARATION
# ==============================================================================

def prepare_temporal_splits(
    df: pd.DataFrame, 
    X_features: List[str]
) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
    """
    Create time-based train/validation/test splits.
    
    Important: For time-series data, we must split by time to prevent
    future information from leaking into training. Uses 60-20-20 split.
    
    Args:
        df: Processed DataFrame with features
        X_features: List of feature column names
        
    Returns:
        Dictionary with train, val, test DataFrames and targets
    """
    # Clean data (drop rows with missing features)
    df = df.dropna(subset=X_features).copy()
    
    # Time-based split points
    t60 = df['match_timestamp'].quantile(TRAIN_SPLIT)
    t80 = df['match_timestamp'].quantile(VAL_SPLIT)
    
    # Fill remaining NAs with training mean (no leakage)
    num_feats = [c for c in X_features if pd.api.types.is_numeric_dtype(df[c])]
    means_t60 = df.loc[df['match_timestamp'] <= t60, num_feats].mean()
    df[num_feats] = df[num_feats].fillna(means_t60)
    
    # Create splits
    train = df[df['match_timestamp'] <= t60].copy()
    val = df[(df['match_timestamp'] > t60) & (df['match_timestamp'] <= t80)].copy()
    test = df[df['match_timestamp'] > t80].copy()
    
    # Also create 80-20 split for final training
    train_final = df[df['match_timestamp'] <= t80].copy()
    
    return {
        'train_tune': (train[X_features], train['is_winner'], train['match_id']),
        'val_tune': (val[X_features], val['is_winner'], val['match_id']),
        'test': (test[X_features], test['is_winner'], test['match_id']),
        'train_final': (train_final[X_features], train_final['is_winner'], train_final['match_id']),
    }


# ==============================================================================
# PAIRWISE EVALUATION
# ==============================================================================

def normalize_probs_per_match(match_ids: pd.Series, probs: np.ndarray) -> np.ndarray:
    """
    Normalize probabilities so they sum to 1 within each match.
    
    For 1v1 matches, P(A wins) + P(B wins) should equal 1.
    This normalizes raw model outputs to satisfy this constraint.
    
    Args:
        match_ids: Series of match IDs
        probs: Raw probability predictions
        
    Returns:
        Normalized probabilities
    """
    s = pd.Series(probs, index=match_ids.index).groupby(match_ids).transform('sum').values
    s = np.where(s <= 0, 1.0, s)  # Prevent division by zero
    return probs / s


def pairwise_argmax_predictions(match_ids: pd.Series, probs_norm: np.ndarray) -> np.ndarray:
    """
    Convert probabilities to binary predictions using pairwise argmax.
    
    Within each match, predict the player with higher probability as winner.
    
    Args:
        match_ids: Series of match IDs
        probs_norm: Normalized probabilities
        
    Returns:
        Binary predictions (1 = predicted winner)
    """
    df = pd.DataFrame({'match_id': match_ids.values, 'p': probs_norm})
    y_pred = np.zeros(len(df), dtype=int)
    y_pred[df.groupby('match_id')['p'].idxmax().values] = 1
    return y_pred


def evaluate_pairwise(
    name: str, 
    match_ids: pd.Series, 
    y_true: pd.Series, 
    probs: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate model using pairwise metrics.
    
    Metrics computed:
    - ROC-AUC: Discriminative ability
    - LogLoss: Calibration quality
    - Precision/Recall/F1: Classification performance
    
    Args:
        name: Model name
        match_ids: Series of match IDs
        y_true: True outcomes
        probs: Raw probability predictions
        
    Returns:
        Dictionary of evaluation metrics
    """
    p_norm = normalize_probs_per_match(match_ids, probs)
    y_pred = pairwise_argmax_predictions(match_ids, p_norm)
    
    return {
        'model': name,
        'ROC_AUC': roc_auc_score(y_true, p_norm),
        'LogLoss': log_loss(y_true, np.clip(p_norm, 1e-15, 1 - 1e-15)),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1': f1_score(y_true, y_pred),
    }


# ==============================================================================
# MODEL TRAINING
# ==============================================================================

def train_logistic_regression(
    X_train: pd.DataFrame, 
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    match_ids_val: pd.Series
) -> Tuple[Pipeline, Dict[str, float]]:
    """
    Train Logistic Regression with regularization tuning.
    
    Logistic Regression is often optimal for antisymmetric features
    since the linear decision boundary respects the symmetry.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        match_ids_val: Match IDs for pairwise evaluation
        
    Returns:
        Tuple of (best model, evaluation metrics)
    """
    Cs = [0.05, 0.1, 0.5, 1, 2, 5]
    best = None
    
    for C in Cs:
        pipe = Pipeline([
            ('scaler', StandardScaler(with_mean=False)),
            ('lr', LogisticRegression(C=C, max_iter=2000, n_jobs=-1))
        ])
        pipe.fit(X_train, y_train)
        
        p_val = pipe.predict_proba(X_val)[:, 1]
        metrics = evaluate_pairwise('lr', match_ids_val, y_val, p_val)
        
        if not best or (metrics['ROC_AUC'], -metrics['LogLoss']) > (best['ROC_AUC'], -best['LogLoss']):
            best = {'C': C, **metrics, 'model': pipe}
    
    return best['model'], {k: v for k, v in best.items() if k != 'model'}


def train_random_forest(
    X_train: pd.DataFrame, 
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    match_ids_val: pd.Series,
    X_train_final: pd.DataFrame,
    y_train_final: pd.Series
) -> Tuple[CalibratedClassifierCV, Dict[str, float]]:
    """
    Train calibrated Random Forest with hyperparameter tuning.
    
    Uses isotonic calibration to improve probability outputs.
    TimeSeriesSplit ensures temporal validity during calibration.
    
    Args:
        Training, validation, and final training data
        
    Returns:
        Tuple of (calibrated model, evaluation metrics)
    """
    n_estimators_list = [300, 400, 500, 600]
    max_depth_list = [3, 4, 5, 6]
    
    best_cfg, best_auc = None, -1.0
    
    for ne in n_estimators_list:
        for md in max_depth_list:
            rf = RandomForestClassifier(
                n_estimators=ne, max_depth=md,
                min_samples_leaf=10, max_features='sqrt',
                random_state=SEED, n_jobs=-1
            )
            rf.fit(X_train, y_train)
            
            p_val = rf.predict_proba(X_val)[:, 1]
            metrics = evaluate_pairwise('tmp', match_ids_val, y_val, p_val)
            
            if metrics['ROC_AUC'] > best_auc:
                best_cfg = {'n_estimators': ne, 'max_depth': md}
                best_auc = metrics['ROC_AUC']
    
    # Train calibrated model on full training data
    tscv = TimeSeriesSplit(n_splits=3)
    rf_best = RandomForestClassifier(
        **best_cfg, min_samples_leaf=10, max_features='sqrt',
        random_state=SEED, n_jobs=-1
    )
    rf_calibrated = CalibratedClassifierCV(estimator=rf_best, method='isotonic', cv=tscv)
    rf_calibrated.fit(X_train_final, y_train_final)
    
    return rf_calibrated, best_cfg


def train_xgboost(
    X_train: pd.DataFrame, 
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    match_ids_val: pd.Series,
    X_train_final: pd.DataFrame,
    y_train_final: pd.Series
) -> Tuple[CalibratedClassifierCV, Dict[str, float]]:
    """
    Train calibrated XGBoost with hyperparameter tuning.
    
    XGBoost can capture non-linear interactions that linear models miss.
    Calibration ensures probability outputs are well-calibrated.
    
    Args:
        Training, validation, and final training data
        
    Returns:
        Tuple of (calibrated model, best hyperparameters)
    """
    n_estimators_list = [300, 400, 500, 600]
    max_depth_list = [4, 5, 6, 7]
    
    best_params, best_auc = None, -1.0
    
    for ne in n_estimators_list:
        for md in max_depth_list:
            xgb = XGBClassifier(
                n_estimators=ne, max_depth=md,
                learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                objective='binary:logistic', tree_method='hist',
                random_state=SEED, n_jobs=-1
            )
            xgb.fit(X_train, y_train)
            
            p_val = xgb.predict_proba(X_val)[:, 1]
            metrics = evaluate_pairwise('tmp', match_ids_val, y_val, p_val)
            
            if metrics['ROC_AUC'] > best_auc:
                best_params = {'n_estimators': ne, 'max_depth': md}
                best_auc = metrics['ROC_AUC']
    
    # Train calibrated model
    tscv = TimeSeriesSplit(n_splits=3)
    xgb_best = XGBClassifier(
        **best_params, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        objective='binary:logistic', tree_method='hist',
        random_state=SEED, n_jobs=-1
    )
    xgb_calibrated = CalibratedClassifierCV(estimator=xgb_best, method='isotonic', cv=tscv)
    xgb_calibrated.fit(X_train_final, y_train_final)
    
    return xgb_calibrated, best_params


# ==============================================================================
# RESULTS REPORTING
# ==============================================================================

def print_results_table(results: List[Dict[str, float]]):
    """Print formatted comparison table of model results."""
    df = pd.DataFrame(results)
    df = df.sort_values('ROC_AUC', ascending=False)
    
    print("\n" + "="*70)
    print("MODEL COMPARISON RESULTS")
    print("="*70)
    print(f"{'Model':<20} {'ROC-AUC':>10} {'LogLoss':>10} {'F1':>10} {'Precision':>10}")
    print("-"*70)
    
    for _, row in df.iterrows():
        print(f"{row['model']:<20} {row['ROC_AUC']:>10.4f} {row['LogLoss']:>10.4f} "
              f"{row['F1']:>10.4f} {row['Precision']:>10.4f}")
    
    print("="*70)
    print(f"🏆 Best Model: {df.iloc[0]['model']} (ROC-AUC: {df.iloc[0]['ROC_AUC']:.4f})")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main(data_path: str) -> Dict[str, any]:
    """
    Execute complete match outcome prediction pipeline.
    
    Args:
        data_path: Path to match data CSV
        
    Returns:
        Dictionary with trained models and results
    """
    print("="*60)
    print("MATCH OUTCOME PREDICTOR - COMPETITIVE GAMING ANALYTICS")
    print("="*60)
    
    # Load data
    print("\n📂 Loading data...")
    data = pd.read_csv(data_path)
    print(f"   Loaded {len(data):,} records")
    print(f"   Players: {data['player_id'].nunique():,}")
    print(f"   Matches: {data['match_id'].nunique():,}")
    print(f"   Decks: {data['deck_id'].nunique():,}")
    
    # Feature engineering
    print("\n🔧 Engineering features...")
    df, X_features = engineer_all_features(data)
    print(f"   Created {len(X_features)} features")
    
    # Prepare splits
    print("\n📊 Preparing temporal train/val/test splits...")
    splits = prepare_temporal_splits(df, X_features)
    
    X_train, y_train, _ = splits['train_tune']
    X_val, y_val, match_ids_val = splits['val_tune']
    X_test, y_test, match_ids_test = splits['test']
    X_train_final, y_train_final, _ = splits['train_final']
    
    print(f"   Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    
    results = []
    
    # Train Logistic Regression
    print("\n🚀 Training Logistic Regression...")
    lr_model, _ = train_logistic_regression(X_train, y_train, X_val, y_val, match_ids_val)
    lr_model.fit(X_train_final, y_train_final)
    p_lr = lr_model.predict_proba(X_test)[:, 1]
    lr_results = evaluate_pairwise('LogReg', match_ids_test, y_test, p_lr)
    results.append(lr_results)
    print(f"   ROC-AUC: {lr_results['ROC_AUC']:.4f}")
    
    # Train Random Forest
    print("\n🚀 Training Random Forest (calibrated)...")
    rf_model, rf_params = train_random_forest(
        X_train, y_train, X_val, y_val, match_ids_val,
        X_train_final, y_train_final
    )
    p_rf = rf_model.predict_proba(X_test)[:, 1]
    rf_results = evaluate_pairwise('RF_Calibrated', match_ids_test, y_test, p_rf)
    results.append(rf_results)
    print(f"   ROC-AUC: {rf_results['ROC_AUC']:.4f}")
    
    # Train XGBoost
    print("\n🚀 Training XGBoost (calibrated)...")
    xgb_model, xgb_params = train_xgboost(
        X_train, y_train, X_val, y_val, match_ids_val,
        X_train_final, y_train_final
    )
    p_xgb = xgb_model.predict_proba(X_test)[:, 1]
    xgb_results = evaluate_pairwise('XGB_Calibrated', match_ids_test, y_test, p_xgb)
    results.append(xgb_results)
    print(f"   ROC-AUC: {xgb_results['ROC_AUC']:.4f}")
    
    # Print comparison
    print_results_table(results)
    
    print("\n✅ Pipeline execution complete!")
    
    return {
        'models': {
            'logistic_regression': lr_model,
            'random_forest': rf_model,
            'xgboost': xgb_model
        },
        'results': results,
        'features': X_features
    }


if __name__ == "__main__":
    print("="*60)
    print("MATCH OUTCOME PREDICTOR")
    print("="*60)
    print("\nUsage:")
    print("  from match_outcome_predictor import main")
    print("  output = main('match_data.csv')")
    print("\nOr run with your data file:")
    print("  python match_outcome_predictor.py")
    print("\nRequired data format:")
    print("  CSV with columns: player_id, match_id, deck_id, match_timestamp, score")
