import numpy as np

from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV

try:
    from .data_import import import_raw_data
    from .feature_transformers import CyclicalEncoder, RushHourEncoder, DryWarmIndexEncoder
    from .model_evaluation import extract_features_and_evaluate
    from .report import print_results
except ImportError:
    from data_import import import_raw_data
    from feature_transformers import CyclicalEncoder, RushHourEncoder, DryWarmIndexEncoder
    from model_evaluation import extract_features_and_evaluate
    from report import print_results

def qda(df, vali_df, seed=1):
    y_train = df["increase_stock"]
    X_train = df.drop(columns=["increase_stock"])
    
    pipe = Pipeline([
        ("rush_hour", RushHourEncoder(n_std=1.45)),
        ("cyclical", CyclicalEncoder({'hour_of_day': 24, 'day_of_week': 7, 'month': 12}, dropOriginal=True)),
        ("dry_warm", DryWarmIndexEncoder()),
        ("scaler", StandardScaler()),
        ("lasso_select", SelectFromModel(LassoCV(random_state=seed))),
        ("model", QuadraticDiscriminantAnalysis()),
    ])
    
    param_grid = {'model__reg_param': np.arange(0.01, 0.51, 0.01)}

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    f_beta_scorer = make_scorer(fbeta_score, beta=2.0)
    
    # Använd CV på träningsdatan för att hitta bästa hyperparametrar
    grid_search = GridSearchCV(pipe, param_grid, cv=cv, scoring=f_beta_scorer, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    # Använd bästa modellen och utvärdera på test-setet EN GÅNG
    best_pipe = grid_search.best_estimator_
    scores, y, y_pred, features = extract_features_and_evaluate(
        best_pipe, 
        vali_df.drop(columns=["increase_stock"]), 
        vali_df["increase_stock"], 
        cv=None,  # Vi gör inte CV på test-setet
        f_beta_scorer=f_beta_scorer
    )
    
    return scores, grid_search.best_params_, y, y_pred, features

if __name__ == "__main__":
    data = import_raw_data("data/training_data_VT2026.csv")
    df, vali_df = train_test_split(data, test_size=0.2, random_state=42, stratify=data["increase_stock"])
    cv_results, best_params, y_true, y_pred, features = qda(df, vali_df)
    
    model_name = f"QDA 10-fold CV (k={len(features)}, reg_param={best_params['model__reg_param']:.2f})"
    print_results(cv_results, model_name, features, y_true, y_pred)
