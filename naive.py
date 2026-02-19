"""Naive baseline - always predicts low bike demand."""

from sklearn.dummy import DummyClassifier
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from SMASK_washington.main import import_data
from SMASK_washington.main import import_data
from model_evaluation import extract_features_and_evaluate
from report import print_results


def naive(df, vali_df, seed=1):
    df = df.copy()
    y = df["increase_stock"]
    X = df.drop(columns=["increase_stock"])
    
    pipe = Pipeline([
        ('model', DummyClassifier(strategy="constant", constant=0))
    ])
    
    f_beta_scorer = make_scorer(fbeta_score, beta=2.0)
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    
    scores, y, y_pred, features = extract_features_and_evaluate(
        pipe, vali_df.drop(columns=["increase_stock"]), vali_df["increase_stock"], cv, 
        transform_steps=[], 
        selector_name=None,
        f_beta_scorer=f_beta_scorer
    )
    
    return scores, y, y_pred, features


if __name__ == "__main__":
    data = import_data("data/training_data_VT2026.csv")
    cv_results, best_params, y_true, y_pred, features = naive(data)
    
    print_results(cv_results, "Naive Baseline (always low demand)", features, y_true, y_pred)

