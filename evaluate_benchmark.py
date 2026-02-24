"""
Evaluate benchmark results from benchmark_pixel_level.py
Generates metrics and visualizations for experiment comparison.
"""

import json
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from scipy import stats
from datetime import datetime




def normalize_results(raw: dict) -> dict:
    """Normalize results to {exp: {patient_ids, pred_medians, gt_medians, config}}."""
    normalized = {}
    for exp_name, data in raw.items():
        if isinstance(data, dict) and 'patients' in data:
            patients = data.get('patients', [])
            patient_ids = [p.get('patient_id') for p in patients]
            gt = [p.get('gt') for p in patients]
            pred = [p.get('pred') for p in patients]
            normalized[exp_name] = {
                'patient_ids': patient_ids,
                'gt_medians': gt,
                'pred_medians': pred,
                'config': data.get('config'),
            }
        else:
            normalized[exp_name] = data
    return normalized


def load_results(results_path) -> dict:
    """Load benchmark results from JSON or YAML."""
    results_path = Path(results_path)
    with open(results_path, 'r') as f:
        if results_path.suffix in (".yaml", ".yml"):
            return yaml.safe_load(f)
        return json.load(f)


def filter_by_gt_threshold(results: dict, min_gt_ff: float = 0.0) -> dict:
    """Filter results to only include patients above a GT FF threshold.

    Args:
        results: Benchmark results dict
        min_gt_ff: Minimum ground truth FF (as fraction, e.g., 0.15 for 15%)

    Returns:
        Filtered results dict
    """
    if min_gt_ff <= 0:
        return results

    filtered = {}
    for exp_name, data in results.items():
        pred = np.array(data['pred_medians'])
        gt = np.array(data['gt_medians'])
        patient_ids = data['patient_ids']

        # Filter by GT threshold
        mask = gt >= min_gt_ff

        filtered[exp_name] = {
            'pred_medians': pred[mask].tolist(),
            'gt_medians': gt[mask].tolist(),
            'patient_ids': [pid for pid, m in zip(patient_ids, mask) if m]
        }

    return filtered


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Compute comprehensive metrics."""
    # Remove any NaN values
    valid = ~(np.isnan(pred) | np.isnan(gt))
    pred = pred[valid]
    gt = gt[valid]

    if len(pred) == 0:
        return {}

    errors = pred - gt
    abs_errors = np.abs(errors)

    metrics = {
        'n_samples': len(pred),
        'mae': np.mean(abs_errors),
        'mae_std': np.std(abs_errors),
        'rmse': np.sqrt(np.mean(errors ** 2)),
        'median_ae': np.median(abs_errors),
        'max_ae': np.max(abs_errors),
        'correlation': np.corrcoef(pred, gt)[0, 1],
        'bias': np.mean(errors),  # Mean signed error
        'bias_std': np.std(errors),
        'gt_mean': np.mean(gt),
        'gt_std': np.std(gt),
        'pred_mean': np.mean(pred),
        'pred_std': np.std(pred),
    }

    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(gt, pred)
    metrics['r_squared'] = r_value ** 2
    metrics['slope'] = slope
    metrics['intercept'] = intercept

    return metrics


def print_metrics_table(results: dict):
    """Print formatted metrics table."""
    print("\n" + "=" * 80)
    print("BENCHMARK METRICS SUMMARY")
    print("=" * 80)

    # Header
    print(f"\n{'Metric':<25} ", end="")
    for exp in results.keys():
        print(f"{exp:>18} ", end="")
    print()
    print("-" * (25 + 19 * len(results)))

    # Compute metrics for each experiment
    all_metrics = {}
    for exp_name, data in results.items():
        pred = np.array(data['pred_medians'])
        gt = np.array(data['gt_medians'])
        all_metrics[exp_name] = compute_metrics(pred, gt)

    # Print rows
    metric_labels = [
        ('n_samples', 'Samples', '', 'd'),
        ('mae', 'MAE', '%', '.2f'),
        ('mae_std', 'MAE Std', '%', '.2f'),
        ('rmse', 'RMSE', '%', '.2f'),
        ('median_ae', 'Median AE', '%', '.2f'),
        ('max_ae', 'Max AE', '%', '.2f'),
        ('correlation', 'Correlation', '', '.4f'),
        ('r_squared', 'R-squared', '', '.4f'),
        ('bias', 'Bias (Pred-GT)', '%', '.3f'),
        ('bias_std', 'Bias Std', '%', '.3f'),
        ('slope', 'Regression Slope', '', '.3f'),
        ('intercept', 'Regression Intercept', '', '.4f'),
    ]

    for key, label, unit, fmt in metric_labels:
        print(f"{label:<25} ", end="")
        for exp in results.keys():
            val = all_metrics[exp].get(key, np.nan)
            # Convert to percentage for error metrics
            if unit == '%':
                val *= 100
            if fmt == 'd':
                print(f"{val:>18d} ", end="")
            else:
                print(f"{val:>18{fmt}} ", end="")
        print()

    print("\n" + "=" * 80)


def plot_scatter_comparison(results: dict, output_dir: Path = None, return_fig: bool = False):
    """Create scatter plots comparing predicted vs ground truth."""
    n_exp = len(results)
    fig, axes = plt.subplots(1, n_exp, figsize=(6 * n_exp, 5))

    if n_exp == 1:
        axes = [axes]

    for ax, (exp_name, data) in zip(axes, results.items()):
        pred = np.array(data['pred_medians']) * 100  # Convert to %
        gt = np.array(data['gt_medians']) * 100

        # Remove NaN
        valid = ~(np.isnan(pred) | np.isnan(gt))
        pred = pred[valid]
        gt = gt[valid]

        # Scatter plot
        ax.scatter(gt, pred, alpha=0.5, s=20, edgecolors='none')

        # Identity line
        lims = [0, max(gt.max(), pred.max()) * 1.1]
        ax.plot(lims, lims, 'k--', alpha=0.5, label='Identity')

        # Regression line
        slope, intercept, r_value, _, _ = stats.linregress(gt, pred)
        x_fit = np.linspace(0, lims[1], 100)
        y_fit = slope * x_fit + intercept
        ax.plot(x_fit, y_fit, 'r-', alpha=0.7, label=f'Fit (R²={r_value**2:.3f})')

        ax.set_xlabel('Ground Truth Median FF (%)')
        ax.set_ylabel('Predicted Median FF (%)')
        ax.set_title(exp_name)
        ax.legend(loc='upper left')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_dir:
        plt.savefig(output_dir / 'scatter_comparison.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {output_dir / 'scatter_comparison.png'}")

    if return_fig:
        return fig
    plt.close()


def plot_bland_altman(results: dict, output_dir: Path = None, return_fig: bool = False):
    """Create Bland-Altman plots for each experiment."""
    n_exp = len(results)
    fig, axes = plt.subplots(1, n_exp, figsize=(6 * n_exp, 5))

    if n_exp == 1:
        axes = [axes]

    for ax, (exp_name, data) in zip(axes, results.items()):
        pred = np.array(data['pred_medians']) * 100
        gt = np.array(data['gt_medians']) * 100

        valid = ~(np.isnan(pred) | np.isnan(gt))
        pred = pred[valid]
        gt = gt[valid]

        mean_vals = (pred + gt) / 2
        diff = pred - gt

        mean_diff = np.mean(diff)
        std_diff = np.std(diff)

        ax.scatter(mean_vals, diff, alpha=0.5, s=20, edgecolors='none')
        ax.axhline(mean_diff, color='red', linestyle='-', label=f'Bias: {mean_diff:.2f}%')
        ax.axhline(mean_diff + 1.96 * std_diff, color='gray', linestyle='--',
                   label=f'+1.96 SD: {mean_diff + 1.96 * std_diff:.2f}%')
        ax.axhline(mean_diff - 1.96 * std_diff, color='gray', linestyle='--',
                   label=f'-1.96 SD: {mean_diff - 1.96 * std_diff:.2f}%')

        ax.set_xlabel('Mean of Predicted and GT (%)')
        ax.set_ylabel('Difference (Predicted - GT) (%)')
        ax.set_title(f'{exp_name} - Bland-Altman')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_dir:
        plt.savefig(output_dir / 'bland_altman.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {output_dir / 'bland_altman.png'}")

    if return_fig:
        return fig
    plt.close()


def plot_error_distribution(results: dict, output_dir: Path = None, return_fig: bool = False):
    """Plot error distribution histograms."""
    n_exp = len(results)
    fig, axes = plt.subplots(1, n_exp, figsize=(6 * n_exp, 5))

    if n_exp == 1:
        axes = [axes]

    for ax, (exp_name, data) in zip(axes, results.items()):
        pred = np.array(data['pred_medians']) * 100
        gt = np.array(data['gt_medians']) * 100

        valid = ~(np.isnan(pred) | np.isnan(gt))
        pred = pred[valid]
        gt = gt[valid]

        errors = pred - gt
        abs_errors = np.abs(errors)

        ax.hist(errors, bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', linewidth=2)
        ax.axvline(np.mean(errors), color='green', linestyle='-', linewidth=2,
                   label=f'Mean: {np.mean(errors):.2f}%')

        ax.set_xlabel('Error (Predicted - GT) (%)')
        ax.set_ylabel('Count')
        ax.set_title(f'{exp_name}\nMAE: {np.mean(abs_errors):.2f}%')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_dir:
        plt.savefig(output_dir / 'error_distribution.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {output_dir / 'error_distribution.png'}")

    if return_fig:
        return fig
    plt.close()


def plot_error_vs_gt(results: dict, output_dir: Path = None, return_fig: bool = False):
    """Plot absolute error vs ground truth value."""
    n_exp = len(results)
    fig, axes = plt.subplots(1, n_exp, figsize=(6 * n_exp, 5))

    if n_exp == 1:
        axes = [axes]

    for ax, (exp_name, data) in zip(axes, results.items()):
        pred = np.array(data['pred_medians']) * 100
        gt = np.array(data['gt_medians']) * 100

        valid = ~(np.isnan(pred) | np.isnan(gt))
        pred = pred[valid]
        gt = gt[valid]

        abs_errors = np.abs(pred - gt)

        ax.scatter(gt, abs_errors, alpha=0.5, s=20, edgecolors='none')

        # Trend line
        z = np.polyfit(gt, abs_errors, 1)
        p = np.poly1d(z)
        x_line = np.linspace(gt.min(), gt.max(), 100)
        ax.plot(x_line, p(x_line), 'r-', alpha=0.7, label='Trend')

        ax.set_xlabel('Ground Truth Median FF (%)')
        ax.set_ylabel('Absolute Error (%)')
        ax.set_title(exp_name)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_dir:
        plt.savefig(output_dir / 'error_vs_gt.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {output_dir / 'error_vs_gt.png'}")

    if return_fig:
        return fig
    plt.close()


def classify_ff(ff_values: np.ndarray) -> np.ndarray:
    """Classify fat fraction values into categories.

    Categories:
        0: Healthy (<5%)
        1: Slight (5-15%)
        2: Mild (15-25%)
        3: Strong (>25%)
    """
    classes = np.zeros_like(ff_values, dtype=int)
    classes[(ff_values >= 0.05) & (ff_values < 0.15)] = 1
    classes[(ff_values >= 0.15) & (ff_values < 0.25)] = 2
    classes[ff_values >= 0.25] = 3
    return classes


def evaluate_classification(results: dict, output_dir: Path = None, return_fig: bool = False):
    """Evaluate classification performance based on FF categories."""
    print("\n" + "=" * 80)
    print("CLASSIFICATION RESULTS")
    print("Categories: Healthy (<5%), Slight (5-15%), Mild (15-25%), Strong (>25%)")
    print("=" * 80)

    class_names = ['Healthy', 'Slight', 'Mild', 'Strong']
    classification_text = []

    for exp_name, data in results.items():
        pred = np.array(data['pred_medians'])
        gt = np.array(data['gt_medians'])

        # Remove NaN
        valid = ~(np.isnan(pred) | np.isnan(gt))
        pred = pred[valid]
        gt = gt[valid]

        # Classify
        pred_classes = classify_ff(pred)
        gt_classes = classify_ff(gt)

        # Overall accuracy
        accuracy = np.mean(pred_classes == gt_classes) * 100

        # Per-class metrics
        print(f"\n{exp_name}:")
        print(f"  Overall Accuracy: {accuracy:.1f}%")

        # Confusion matrix
        n_classes = 4
        conf_matrix = np.zeros((n_classes, n_classes), dtype=int)
        for gt_c, pred_c in zip(gt_classes, pred_classes):
            conf_matrix[gt_c, pred_c] += 1

        print(f"\n  Confusion Matrix (rows=GT, cols=Pred):")
        print(f"  {'':>10} ", end="")
        for name in class_names:
            print(f"{name:>8} ", end="")
        print()

        for i, name in enumerate(class_names):
            print(f"  {name:>10} ", end="")
            for j in range(n_classes):
                print(f"{conf_matrix[i, j]:>8} ", end="")
            print()

        # Per-class statistics
        print(f"\n  Per-Class Performance:")
        print(f"  {'Class':<10} {'N (GT)':>8} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print(f"  {'-'*48}")

        exp_text = [f"{exp_name}:", f"  Overall Accuracy: {accuracy:.1f}%", ""]
        exp_text.append("  Per-Class Performance:")
        exp_text.append(f"  {'Class':<10} {'N (GT)':>8} {'Precision':>10} {'Recall':>10} {'F1':>10}")

        for i, name in enumerate(class_names):
            n_gt = np.sum(gt_classes == i)
            n_pred = np.sum(pred_classes == i)
            tp = conf_matrix[i, i]

            precision = tp / n_pred * 100 if n_pred > 0 else 0
            recall = tp / n_gt * 100 if n_gt > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            print(f"  {name:<10} {n_gt:>8} {precision:>9.1f}% {recall:>9.1f}% {f1:>9.1f}%")
            exp_text.append(f"  {name:<10} {n_gt:>8} {precision:>9.1f}% {recall:>9.1f}% {f1:>9.1f}%")

        # Adjacent accuracy (within 1 class)
        adjacent_correct = np.abs(pred_classes - gt_classes) <= 1
        adjacent_accuracy = np.mean(adjacent_correct) * 100
        print(f"\n  Adjacent Accuracy (within 1 class): {adjacent_accuracy:.1f}%")
        exp_text.append(f"\n  Adjacent Accuracy (within 1 class): {adjacent_accuracy:.1f}%")
        classification_text.append("\n".join(exp_text))

    # Plot confusion matrices
    n_exp = len(results)
    fig, axes = plt.subplots(1, n_exp, figsize=(5 * n_exp, 4))

    if n_exp == 1:
        axes = [axes]

    for ax, (exp_name, data) in zip(axes, results.items()):
        pred = np.array(data['pred_medians'])
        gt = np.array(data['gt_medians'])

        valid = ~(np.isnan(pred) | np.isnan(gt))
        pred = pred[valid]
        gt = gt[valid]

        pred_classes = classify_ff(pred)
        gt_classes = classify_ff(gt)

        # Confusion matrix
        n_classes = 4
        conf_matrix = np.zeros((n_classes, n_classes), dtype=int)
        for gt_c, pred_c in zip(gt_classes, pred_classes):
            conf_matrix[gt_c, pred_c] += 1

        # Normalize by row (recall)
        conf_matrix_norm = conf_matrix.astype(float)
        row_sums = conf_matrix_norm.sum(axis=1, keepdims=True)
        conf_matrix_norm = np.divide(conf_matrix_norm, row_sums, where=row_sums != 0) * 100

        im = ax.imshow(conf_matrix_norm, cmap='Blues', vmin=0, vmax=100)

        # Add text annotations
        for i in range(n_classes):
            for j in range(n_classes):
                val = conf_matrix[i, j]
                pct = conf_matrix_norm[i, j]
                color = 'white' if pct > 50 else 'black'
                ax.text(j, i, f'{val}\n({pct:.0f}%)', ha='center', va='center',
                        color=color, fontsize=9)

        ax.set_xticks(range(n_classes))
        ax.set_yticks(range(n_classes))
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_yticklabels(class_names)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Ground Truth')
        ax.set_title(f'{exp_name}\nAccuracy: {np.mean(pred_classes == gt_classes)*100:.1f}%')

    plt.tight_layout()

    if output_dir:
        plt.savefig(output_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
        print(f"\nSaved: {output_dir / 'confusion_matrix.png'}")

    if return_fig:
        return fig, classification_text
    plt.close()
    return None, classification_text


def evaluate_ranking(results: dict) -> list:
    """Evaluate how well predictions preserve patient ranking by fat fraction."""
    print("\n" + "=" * 80)
    print("RANKING METRICS")
    print("How well does the model preserve patient ordering by fat fraction?")
    print("=" * 80)

    ranking_text = []

    for exp_name, data in results.items():
        pred = np.array(data['pred_medians'])
        gt = np.array(data['gt_medians'])
        patient_ids = data['patient_ids']

        # Remove NaN
        valid = ~(np.isnan(pred) | np.isnan(gt))
        pred = pred[valid]
        gt = gt[valid]
        patient_ids = [pid for pid, v in zip(patient_ids, valid) if v]

        # Compute rank correlations
        spearman_corr, spearman_p = stats.spearmanr(gt, pred)
        kendall_tau, kendall_p = stats.kendalltau(gt, pred)

        print(f"\n{exp_name}:")
        print(f"  Spearman's rho: {spearman_corr:.4f} (p={spearman_p:.2e})")
        print(f"  Kendall's tau:  {kendall_tau:.4f} (p={kendall_p:.2e})")

        exp_text = [f"{exp_name}:",
                    f"  Spearman's rho: {spearman_corr:.4f} (p={spearman_p:.2e})",
                    f"  Kendall's tau:  {kendall_tau:.4f} (p={kendall_p:.2e})"]

        # Get rankings
        gt_ranks = np.argsort(np.argsort(gt))  # Rank 0 = lowest FF
        pred_ranks = np.argsort(np.argsort(pred))

        # Check top-k accuracy (are the top-k patients by GT also in top-k by prediction?)
        n = len(gt)
        for k in [5, 10, 20]:
            if k > n:
                continue
            gt_top_k = set(np.argsort(gt)[-k:])  # Indices of top-k by GT
            pred_top_k = set(np.argsort(pred)[-k:])  # Indices of top-k by prediction
            overlap = len(gt_top_k & pred_top_k)
            print(f"  Top-{k} overlap: {overlap}/{k} ({overlap/k*100:.0f}%)")
            exp_text.append(f"  Top-{k} overlap: {overlap}/{k} ({overlap/k*100:.0f}%)")

        # Show actual top patients comparison
        print(f"\n  Top 5 patients by GT vs Prediction:")
        print(f"  {'Rank':<6} {'GT Patient':<15} {'GT FF%':>8} {'Pred Patient':<15} {'Pred FF%':>8}")
        print(f"  {'-'*58}")

        exp_text.append(f"\n  Top 5 patients by GT vs Prediction:")
        exp_text.append(f"  {'Rank':<6} {'GT Patient':<15} {'GT FF%':>8} {'Pred Patient':<15} {'Pred FF%':>8}")

        gt_sorted_idx = np.argsort(gt)[::-1]  # Descending
        pred_sorted_idx = np.argsort(pred)[::-1]

        for rank in range(min(5, n)):
            gt_idx = gt_sorted_idx[rank]
            pred_idx = pred_sorted_idx[rank]
            line = (f"  {rank+1:<6} {patient_ids[gt_idx]:<15} {gt[gt_idx]*100:>7.2f}% "
                    f"{patient_ids[pred_idx]:<15} {pred[pred_idx]*100:>7.2f}%")
            print(line)
            exp_text.append(line)

        # Rank difference analysis
        rank_diffs = np.abs(gt_ranks - pred_ranks)
        print(f"\n  Rank difference statistics:")
        print(f"    Mean rank diff: {np.mean(rank_diffs):.1f} positions")
        print(f"    Median rank diff: {np.median(rank_diffs):.1f} positions")
        print(f"    Max rank diff: {np.max(rank_diffs)} positions")
        print(f"    Within 10 ranks: {np.mean(rank_diffs <= 10)*100:.1f}%")
        print(f"    Within 20 ranks: {np.mean(rank_diffs <= 20)*100:.1f}%")

        exp_text.extend([
            f"\n  Rank difference statistics:",
            f"    Mean rank diff: {np.mean(rank_diffs):.1f} positions",
            f"    Median rank diff: {np.median(rank_diffs):.1f} positions",
            f"    Max rank diff: {np.max(rank_diffs)} positions",
            f"    Within 10 ranks: {np.mean(rank_diffs <= 10)*100:.1f}%",
            f"    Within 20 ranks: {np.mean(rank_diffs <= 20)*100:.1f}%"
        ])

        ranking_text.append("\n".join(exp_text))

    return ranking_text


def identify_outliers(results: dict, threshold_pct: float = 10.0) -> list:
    """Identify patients with high prediction errors."""
    print("\n" + "=" * 80)
    print(f"OUTLIERS (Absolute Error > {threshold_pct}%)")
    print("=" * 80)

    outlier_text = []

    for exp_name, data in results.items():
        pred = np.array(data['pred_medians']) * 100
        gt = np.array(data['gt_medians']) * 100
        patient_ids = data['patient_ids']

        abs_errors = np.abs(pred - gt)
        outlier_mask = abs_errors > threshold_pct

        exp_text = []
        if outlier_mask.sum() > 0:
            print(f"\n{exp_name}: {outlier_mask.sum()} outliers")
            print(f"{'Patient ID':<15} {'GT (%)':>10} {'Pred (%)':>10} {'Error (%)':>10}")
            print("-" * 50)

            exp_text.append(f"{exp_name}: {outlier_mask.sum()} outliers")
            exp_text.append(f"{'Patient ID':<15} {'GT (%)':>10} {'Pred (%)':>10} {'Error (%)':>10}")
            exp_text.append("-" * 50)

            outlier_indices = np.where(outlier_mask)[0]
            # Sort by error
            sorted_indices = outlier_indices[np.argsort(abs_errors[outlier_indices])[::-1]]

            for idx in sorted_indices[:10]:  # Show top 10
                line = f"{patient_ids[idx]:<15} {gt[idx]:>10.2f} {pred[idx]:>10.2f} {pred[idx] - gt[idx]:>+10.2f}"
                print(line)
                exp_text.append(line)
        else:
            print(f"\n{exp_name}: No outliers")
            exp_text.append(f"{exp_name}: No outliers")

        outlier_text.append("\n".join(exp_text))

    return outlier_text


def create_metrics_table_figure(results: dict):
    """Create a figure with the metrics table."""
    # Compute metrics for each experiment
    all_metrics = {}
    for exp_name, data in results.items():
        pred = np.array(data['pred_medians'])
        gt = np.array(data['gt_medians'])
        all_metrics[exp_name] = compute_metrics(pred, gt)

    # Define metric rows
    metric_labels = [
        ('n_samples', 'Samples', '', 'd'),
        ('mae', 'MAE', '%', '.2f'),
        ('mae_std', 'MAE Std', '%', '.2f'),
        ('rmse', 'RMSE', '%', '.2f'),
        ('median_ae', 'Median AE', '%', '.2f'),
        ('max_ae', 'Max AE', '%', '.2f'),
        ('correlation', 'Correlation', '', '.4f'),
        ('r_squared', 'R-squared', '', '.4f'),
        ('bias', 'Bias (Pred-GT)', '%', '.3f'),
        ('bias_std', 'Bias Std', '%', '.3f'),
        ('slope', 'Regression Slope', '', '.3f'),
        ('intercept', 'Regression Intercept', '', '.4f'),
    ]

    # Create table data
    col_labels = ['Metric'] + list(results.keys())
    table_data = []

    for key, label, unit, fmt in metric_labels:
        row = [label]
        for exp in results.keys():
            val = all_metrics[exp].get(key, np.nan)
            if unit == '%':
                val *= 100
            if fmt == 'd':
                row.append(f"{val:d}")
            else:
                row.append(f"{val:{fmt}}")
        table_data.append(row)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    ax.set_title('Benchmark Metrics Summary', fontsize=14, fontweight='bold', pad=20)

    # Create table
    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    # Style header
    for i in range(len(col_labels)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    plt.tight_layout()
    return fig


def create_text_figure(title: str, text_content: list, fontsize: int = 9):
    """Create a figure with text content."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    # Join all text
    full_text = "\n\n".join(text_content)

    # Add text to figure
    ax.text(0.02, 0.98, full_text, transform=ax.transAxes,
            fontsize=fontsize, verticalalignment='top',
            fontfamily='monospace', wrap=True)

    plt.tight_layout()
    return fig


def generate_pdf_report(results: dict, output_dir: Path, min_gt_ff_pct: float = 0.0,
                        classification_text: list = None, ranking_text: list = None,
                        outlier_text: list = None):
    """Generate a comprehensive PDF report with all evaluation results.

    Args:
        results: Benchmark results dict
        output_dir: Output directory for the PDF
        min_gt_ff_pct: Minimum GT FF threshold in percentage
        classification_text: Pre-computed classification text (to avoid duplicate printing)
        ranking_text: Pre-computed ranking text
        outlier_text: Pre-computed outlier text
    """
    pdf_path = output_dir / 'benchmark_report.pdf'

    with PdfPages(pdf_path) as pdf:
        # Title page
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.axis('off')
        ax.text(0.5, 0.6, 'Benchmark Evaluation Report',
                ha='center', va='center', fontsize=24, fontweight='bold')
        ax.text(0.5, 0.45, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                ha='center', va='center', fontsize=12)
        ax.text(0.5, 0.35, f'Experiments: {", ".join(results.keys())}',
                ha='center', va='center', fontsize=12)
        n_patients = len(list(results.values())[0]['patient_ids'])
        ax.text(0.5, 0.25, f'Patients evaluated: {n_patients}',
                ha='center', va='center', fontsize=12)
        if min_gt_ff_pct > 0:
            ax.text(0.5, 0.15, f'GT FF threshold: >= {min_gt_ff_pct}%',
                    ha='center', va='center', fontsize=12, color='red')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # Config pages (if available in results)
        for exp_name, data in results.items():
            config = data.get("config")
            if config is None:
                continue
            config_text = json.dumps(config, indent=2, sort_keys=True)
            fig = create_text_figure(f"Config: {exp_name}", [config_text], fontsize=7)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

        # Metrics table
        fig = create_metrics_table_figure(results)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # Scatter plots
        fig = plot_scatter_comparison(results, return_fig=True)
        if fig:
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

        # Bland-Altman plots
        fig = plot_bland_altman(results, return_fig=True)
        if fig:
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

        # Error distribution
        fig = plot_error_distribution(results, return_fig=True)
        if fig:
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

        # Error vs GT
        fig = plot_error_vs_gt(results, return_fig=True)
        if fig:
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

        # Confusion matrix figure (generated fresh for PDF, no printing)
        fig = create_confusion_matrix_figure(results)
        if fig:
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

        # Classification text page
        if classification_text:
            fig = create_text_figure('Classification Results', classification_text)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

        # Ranking metrics
        if ranking_text:
            fig = create_text_figure('Ranking Metrics', ranking_text)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

        # Outliers
        if outlier_text:
            fig = create_text_figure('Outliers (Absolute Error > 10%)', outlier_text)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    print(f"\nPDF report saved to: {pdf_path}")
    return pdf_path


def create_confusion_matrix_figure(results: dict):
    """Create confusion matrix figure without printing."""
    class_names = ['Healthy', 'Slight', 'Mild', 'Strong']

    n_exp = len(results)
    fig, axes = plt.subplots(1, n_exp, figsize=(5 * n_exp, 4))

    if n_exp == 1:
        axes = [axes]

    for ax, (exp_name, data) in zip(axes, results.items()):
        pred = np.array(data['pred_medians'])
        gt = np.array(data['gt_medians'])

        valid = ~(np.isnan(pred) | np.isnan(gt))
        pred = pred[valid]
        gt = gt[valid]

        pred_classes = classify_ff(pred)
        gt_classes = classify_ff(gt)

        # Confusion matrix
        n_classes = 4
        conf_matrix = np.zeros((n_classes, n_classes), dtype=int)
        for gt_c, pred_c in zip(gt_classes, pred_classes):
            conf_matrix[gt_c, pred_c] += 1

        # Normalize by row (recall)
        conf_matrix_norm = conf_matrix.astype(float)
        row_sums = conf_matrix_norm.sum(axis=1, keepdims=True)
        conf_matrix_norm = np.divide(conf_matrix_norm, row_sums, where=row_sums != 0) * 100

        im = ax.imshow(conf_matrix_norm, cmap='Blues', vmin=0, vmax=100)

        # Add text annotations
        for i in range(n_classes):
            for j in range(n_classes):
                val = conf_matrix[i, j]
                pct = conf_matrix_norm[i, j]
                color = 'white' if pct > 50 else 'black'
                ax.text(j, i, f'{val}\n({pct:.0f}%)', ha='center', va='center',
                        color=color, fontsize=9)

        ax.set_xticks(range(n_classes))
        ax.set_yticks(range(n_classes))
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_yticklabels(class_names)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Ground Truth')
        ax.set_title(f'{exp_name}\nAccuracy: {np.mean(pred_classes == gt_classes)*100:.1f}%')

    plt.tight_layout()
    return fig


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate benchmark results')
    parser.add_argument('--min-gt-ff', type=float, default=0.0,
                        help='Minimum GT FF threshold as percentage (e.g., 15 for 15%%). Default: 0 (no filter)')
    parser.add_argument('--results', type=str,
                        default="/home/homesOnMaster/dgeiger/repos/Liver_FF_Predictor/outputs/scalar_regression_run/benchmark_evaluation/experiment_20260123_131810_experiment_20260123_131929/benchmark_results_scalar_regression.json",
                        help='Path to benchmark_results.json')
    parser.add_argument('--outlier-threshold', type=float, default=10.0,
                        help='Absolute error threshold (%%) for outlier detection. Default: 10.0')

    # Toggle individual outputs (all enabled by default)
    parser.add_argument('--no-scatter', action='store_true', help='Skip scatter plot')
    parser.add_argument('--no-bland-altman', action='store_true', help='Skip Bland-Altman plot')
    parser.add_argument('--no-error-distribution', action='store_true', help='Skip error distribution histogram')
    parser.add_argument('--no-error-vs-gt', action='store_true', help='Skip error vs GT plot')
    parser.add_argument('--no-classification', action='store_true', help='Skip classification evaluation')
    parser.add_argument('--no-ranking', action='store_true', help='Skip ranking metrics')
    parser.add_argument('--no-outliers', action='store_true', help='Skip outlier detection')
    parser.add_argument('--no-pdf', action='store_true', help='Skip PDF report generation')
    parser.add_argument('--no-metrics', action='store_true', help='Skip metrics table printout')

    args = parser.parse_args()

    # Convert percentage to fraction
    min_gt_ff = args.min_gt_ff / 100.0

    # Paths - output goes in the same directory as the results file
    results_path = Path(args.results)
    output_dir = results_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    print("Loading benchmark results...")
    results = load_results(results_path)
    results = normalize_results(results)
    print(f"Found {len(results)} experiments")

    # Get total patient count before filtering
    total_patients = len(list(results.values())[0]['patient_ids'])

    # Filter by GT threshold
    if min_gt_ff > 0:
        results = filter_by_gt_threshold(results, min_gt_ff)
        filtered_patients = len(list(results.values())[0]['patient_ids'])
        print(f"\nFiltering: GT FF >= {args.min_gt_ff}%")
        print(f"Patients: {total_patients} -> {filtered_patients} ({filtered_patients/total_patients*100:.1f}%)")

    # Print metrics table
    if not args.no_metrics:
        print_metrics_table(results)

    # Generate plots (save as PNG)
    print("\nGenerating visualizations...")
    if not args.no_scatter:
        plot_scatter_comparison(results, output_dir)
    if not args.no_bland_altman:
        plot_bland_altman(results, output_dir)
    if not args.no_error_distribution:
        plot_error_distribution(results, output_dir)
    if not args.no_error_vs_gt:
        plot_error_vs_gt(results, output_dir)

    # Classification evaluation (save confusion matrix as PNG and capture text)
    classification_text = None
    if not args.no_classification:
        _, classification_text = evaluate_classification(results, output_dir)

    # Ranking evaluation (capture text)
    ranking_text = None
    if not args.no_ranking:
        ranking_text = evaluate_ranking(results)

    # Identify outliers (capture text)
    outlier_text = None
    if not args.no_outliers:
        outlier_text = identify_outliers(results, threshold_pct=args.outlier_threshold)

    # Generate PDF report (pass captured text to avoid duplicate printing)
    if not args.no_pdf:
        print("\nGenerating PDF report...")
        generate_pdf_report(
            results, output_dir,
            min_gt_ff_pct=args.min_gt_ff,
            classification_text=classification_text,
            ranking_text=ranking_text,
            outlier_text=outlier_text
        )

    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
