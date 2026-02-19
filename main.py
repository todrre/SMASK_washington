"""Main script - kör och jämför alla modeller."""

from qda import qda
from knn import knn
from naive import naive
from report import print_comparison


def main(data_path="data/training_data_VT2026.csv"):
    """Kör alla modeller och jämför resultaten."""
    
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
            cv_results, y_true, y_pred, features = model_func(data_path)
        else:
            cv_results, best_params, y_true, y_pred, features = model_func(data_path)
        
        results[model_name] = cv_results
    
    # Visa jämförelse
    print_comparison(results)


if __name__ == "__main__":
    main()
