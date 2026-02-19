import numpy as np

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.metrics import make_scorer, fbeta_score

from data_import import import_raw_data
from SMASK_washington.feature_transformers import CyclicalEncoder, RushHourEncoder, DryWarmIndexEncoder
from SMASK_washington.model_evaluation import extract_features_and_evaluate
from SMASK_washington.report import print_results

def knn(data_path, seed=1):
    df = import_raw_data(data_path)
    y = df["increase_stock"]
    X = df.drop(columns=["increase_stock"])
    
    pipe = Pipeline([
        ("rush_hour", RushHourEncoder(n_std=1.45)),
        ("cyclical", CyclicalEncoder({'hour_of_day': 24, 'day_of_week': 7, 'month': 12}, dropOriginal=True)),
        ("dry_warm", DryWarmIndexEncoder(dropOriginal=False)),
        ("scaler", StandardScaler()),
        ("lasso_select", SelectFromModel(LassoCV(cv=5, random_state=seed))),
        ("model", KNeighborsClassifier()),
    ])
    
    param_grid = {'model__n_neighbors': np.arange(1, 100, 1)}

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    f_beta_scorer = make_scorer(fbeta_score, beta=2.0)
    grid_search = GridSearchCV(pipe, param_grid, cv=cv, scoring=f_beta_scorer, n_jobs=-1, verbose=1)
    grid_search.fit(X, y)
    
    pipe = grid_search.best_estimator_
    scores, y, y_pred, features = extract_features_and_evaluate(pipe, X, y, cv, f_beta_scorer=f_beta_scorer)
    
    return scores, grid_search.best_params_, y, y_pred, features

if __name__ == "__main__":
    cv_results, best_params, y_true, y_pred, features = knn("data/training_data_VT2026.csv")
    
    model_name = f"KNN 10-fold CV (k={len(features)}, n_neighbors={best_params['model__n_neighbors']})"
    print_results(cv_results, model_name, features, y_true, y_pred)
