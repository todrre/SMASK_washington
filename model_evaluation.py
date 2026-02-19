import numpy as np
from sklearn.model_selection import cross_validate, cross_val_predict

def extract_features_and_evaluate(pipe, X, y, cv, transform_steps=['rush_hour', 'cyclical', 'dry_warm'], selector_name='lasso_select', f_beta_scorer=None):
    """
    Extraherar valda features och kör cross-validation.
    """
    # Extrahera valda features (om selector finns)
    if selector_name and selector_name in pipe.named_steps:
        X_temp = X.copy()
        for step in transform_steps:
            if step in pipe.named_steps:
                X_temp = pipe.named_steps[step].transform(X_temp)
        
        selected_mask = pipe.named_steps[selector_name].get_support()
        selected_features = [X_temp.columns[i] for i in range(len(selected_mask)) if selected_mask[i]]
    else:
        selected_features = []
    
    # Cross-validation scores
    scoring = {
        "acc": "accuracy",
        "bal_acc": "balanced_accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
    }
    
    if f_beta_scorer:
        scoring["fbeta"] = f_beta_scorer
    
    scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring, return_train_score=False)
    y_pred_oof = cross_val_predict(pipe, X, y, cv=cv)
    
    return scores, y, y_pred_oof, selected_features
