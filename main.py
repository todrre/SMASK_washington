"""Main script - kör och jämför alla modeller."""

from pathlib import Path

from sklearn.model_selection import train_test_split

try:
    from .data_import import import_raw_data
    from .qda import qda
    from .knn import knn
    from .naive import naive
    from .report import print_comparison
except ImportError:
    from data_import import import_raw_data
    from qda import qda
    from knn import knn
    from naive import naive
    from report import print_comparison

def import_data(data_path=None):
    if data_path is None:
        data_path = Path(__file__).resolve().parent / "data" / "training_data_VT2026.csv"
    data = import_raw_data(data_path)
    df, vali_df = train_test_split(data, test_size=0.2, random_state=42, stratify=data["increase_stock"])
    return df, vali_df

def main(data_path=None):
    """Kör alla modeller och jämför resultaten."""
    if data_path is None:
        data_path = Path(__file__).resolve().parent / "data" / "training_data_VT2026.csv"

    df, vali_df = import_data(data_path)

    models = [
        ('Naive', naive),
        ('QDA', qda),
        ('KNN', knn),
    ]
    
    results = {}
    
    for model_name, model_func in models:
        print(f"\n{'='*60}")
        print(f"Running {model_name}...")
        print('='*60)
        
        if model_name == 'Naive':
            cv_results, y_true, y_pred, features = model_func(df, vali_df)
        else:
            cv_results, best_params, y_true, y_pred, features = model_func(df, vali_df)
        
        results[model_name] = cv_results
    
    # Visa jämförelse
    print_comparison(results)


if __name__ == "__main__":
    main()
