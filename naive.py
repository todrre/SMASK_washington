"""Naive baseline - always predicts low bike demand."""

from sklearn.dummy import DummyClassifier
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from data_import import import_raw_data
from model_evaluation import extract_features_and_evaluate
from report import print_results


def naive(data_path, seed=1):
    df = import_raw_data(data_path)
    y = df["increase_stock"]
    X = df.drop(columns=["increase_stock"])
    
    pipe = Pipeline([
        ('model', DummyClassifier(strategy="constant", constant=0))
    ])
    
    f_beta_scorer = make_scorer(fbeta_score, beta=2.0)
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    
    scores, y, y_pred, features = extract_features_and_evaluate(
        pipe, X, y, cv, 
        transform_steps=[], 
        selector_name=None,
        f_beta_scorer=f_beta_scorer
    )
    
    return scores, y, y_pred, features


if __name__ == "__main__":
    cv_results, y_true, y_pred, features = naive("data/training_data_VT2026.csv")
    
    print_results(cv_results, "Naive Baseline (always low demand)", features, y_true, y_pred)

