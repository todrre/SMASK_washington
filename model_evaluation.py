import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, 
    recall_score, f1_score, roc_auc_score, average_precision_score, fbeta_score
)

def extract_features_and_evaluate(pipe, X_test, y_test, cv=None, transform_steps=['rush_hour', 'cyclical', 'dry_warm'], selector_name='lasso_select', f_beta_scorer=None):
    """
    Extraherar valda features och utvärderar modellen på test-setet.
    
    OBS: Pipe ska redan vara tränad (t.ex. från GridSearchCV.best_estimator_).
    Denna funktion gör EN evaluation på test-setet, inte cross-validation.
    """
    # Extrahera valda features (om selector finns)
    if selector_name and selector_name in pipe.named_steps:
        X_temp = X_test.copy()
        for step in transform_steps:
            if step in pipe.named_steps:
                X_temp = pipe.named_steps[step].transform(X_temp)
        
        selected_mask = pipe.named_steps[selector_name].get_support()
        selected_features = [X_temp.columns[i] for i in range(len(selected_mask)) if selected_mask[i]]
    else:
        selected_features = []
    
    # Utvärdera på test set (EN GÅNG, inte CV)
    y_pred = pipe.predict(X_test)
    y_pred_proba = pipe.predict_proba(X_test)[:, 1]
    
    # Beräkna alla metrics
    scores = {
        "test_acc": [accuracy_score(y_test, y_pred)],
        "test_bal_acc": [balanced_accuracy_score(y_test, y_pred)],
        "test_precision": [precision_score(y_test, y_pred, zero_division=0)],
        "test_recall": [recall_score(y_test, y_pred, zero_division=0)],
        "test_f1": [f1_score(y_test, y_pred, zero_division=0)],
        "test_roc_auc": [roc_auc_score(y_test, y_pred_proba)],
        "test_pr_auc": [average_precision_score(y_test, y_pred_proba)],
    }
    
    if f_beta_scorer:
        scores["test_fbeta"] = [fbeta_score(y_test, y_pred, beta=2.0)]
    
    return scores, y_test, y_pred, selected_features
