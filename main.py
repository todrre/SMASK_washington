"""Main script - kör och jämför alla modeller."""

from SMASK_washington.qda import qda
from SMASK_washington.knn import knn
from SMASK_washington.naive import naive
from SMASK_washington.report import print_comparison


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
