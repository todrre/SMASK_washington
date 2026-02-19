import numpy as np
from sklearn.metrics import confusion_matrix
from rich.console import Console
from rich.table import Table
from rich import box


def print_comparison(results_dict):
    """Jämför flera modeller i en tabell.
    
    Args:
        results_dict: Dict med {model_name: cv_results}
    """
    console = Console()
    
    # Metrics names för display
    names = {
        'acc': 'Accuracy',
        'bal_acc': 'Balanced Accuracy',
        'roc_auc': 'ROC-AUC',
        'pr_auc': 'PR-AUC',
        'precision': 'Precision',
        'recall': 'Recall',
        'f1': 'F1-Score',
        'fbeta': 'F_beta (β=2.0)'
    }
    
    # Samla alla metrics
    all_metrics = set()
    summaries = {}
    for model_name, cv_results in results_dict.items():
        summary = {}
        for key, values in cv_results.items():
            if key.startswith('test_'):
                metric_name = key.replace('test_', '')
                summary[metric_name] = np.mean(values)
                all_metrics.add(metric_name)
        summaries[model_name] = summary
    
    # Skapa tabell
    table = Table(title="Model Comparison (10-fold CV)", box=box.ROUNDED, show_header=True)
    table.add_column("Metric", style="bold cyan", width=20)
    
    for model_name in results_dict.keys():
        table.add_column(model_name, justify="right", style="green", width=12)
    
    # Sortera metrics
    metrics_order = ['acc', 'bal_acc', 'roc_auc', 'pr_auc', 'precision', 'recall', 'f1', 'fbeta']
    sorted_metrics = [m for m in metrics_order if m in all_metrics]
    
    # Lägg till rader
    for metric in sorted_metrics:
        display_name = names.get(metric, metric.replace('_', ' ').title())
        row = [display_name]
        
        values = [summaries[model_name].get(metric, 0.0) for model_name in results_dict.keys()]
        best_val = max(values)
        
        for model_name in results_dict.keys():
            val = summaries[model_name].get(metric, 0.0)
            if val == best_val and val > 0:
                row.append(f"[bold]{val:.4f}[/bold]")
            else:
                row.append(f"{val:.4f}")
        
        table.add_row(*row)
    
    console.print()
    console.print(table)
    console.print()


def print_results(cv_results, model_name, features, y_true, y_pred):
    """Printar metricer och confusion matrix snyggt."""
    console = Console()
    
    # Sammanfatta CV resultat
    summary = {}
    for key, values in cv_results.items():
        if key.startswith('test_'):
            metric_name = key.replace('test_', '')
            summary[metric_name] = (np.mean(values), np.std(values))
    
    console.print()
    console.print(f"[bold cyan]{model_name}[/bold cyan]")
    console.print(f"[bold]Features:[/bold] {len(features)}")
    for i in range(0, len(features), 5):
        console.print(f"  → {', '.join(features[i:i+5])}")
    console.print()
    
    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    cm_table = Table(title="Confusion Matrix (Out-of-Fold)", box=box.SIMPLE, show_header=True)
    cm_table.add_column("", style="bold")
    cm_table.add_column("y = 0", justify="center", style="cyan")
    cm_table.add_column("y = 1", justify="center", style="cyan")
    cm_table.add_row("ŷ = 0", f"[green]TN={tn}[/green]", f"[red]FN={fn}[/red]")
    cm_table.add_row("ŷ = 1", f"[yellow]FP={fp}[/yellow]", f"[green]TP={tp}[/green]")
    console.print(cm_table)
    console.print()
    
    # Metrics table
    names = {
        'acc': 'Accuracy',
        'bal_acc': 'Balanced Accuracy',
        'roc_auc': 'ROC-AUC',
        'pr_auc': 'PR-AUC',
        'precision': 'Precision',
        'recall': 'Recall',
        'f1': 'F1-Score',
        'fbeta': 'Fbeta_Score'
    }
    
    table = Table(title="Test Performance (CV)", box=box.SIMPLE, show_header=True)
    table.add_column("Metric", style="bold", width=18)
    table.add_column("Mean", justify="right", style="cyan", width=10)
    table.add_column("Std Dev", justify="right", style="yellow", width=10)
    table.add_column("Range", justify="center", style="dim", width=22)
    
    for metric, (mean, std) in sorted(summary.items()):
        display_name = names.get(metric, metric.replace('_', ' ').title())
        lower, upper = max(0, mean - std), min(1, mean + std)
        table.add_row(display_name, f"{mean:.4f}", f"±{std:.4f}", f"[{lower:.4f}, {upper:.4f}]")
    
    console.print(table)
    console.print()