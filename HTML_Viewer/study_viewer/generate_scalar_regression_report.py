"""
Generate a PDF report evaluating the scalar regression model on the study viewer test set.

Loads the study test set (excluding patients from exluding_patients.txt), runs
inference with the scalar regression model, classifies predictions into steatosis
grades, and generates a PDF report with confusion matrix, per-class metrics, and
misclassified patient details.

Usage:
    python HTML_Viewer/study_viewer/generate_scalar_regression_report.py
    python HTML_Viewer/study_viewer/generate_scalar_regression_report.py --checkpoint outputs/scalar_regression_run/experiment_20260123_131810/checkpoint_best.pth
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import yaml
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scalar_regression.dataset_scalar import LiverFatScalarDataset, pad_collate_scalar
from scalar_regression.model_scalar import get_scalar_model
from scalar_regression.old_model import get_scalar_model as get_old_scalar_model

# Steatosis grade definitions (same as study viewer)
CLASS_ORDER = ['healthy', 'mild', 'moderate', 'severe']
CLASS_LABELS = ['Healthy (0-5%)', 'Mild (5-15%)', 'Moderate (15-25%)', 'Severe (>25%)']
CLASS_THRESHOLDS = [0.05, 0.15, 0.25]  # upper bounds for healthy, mild, moderate
N_CLASSES = len(CLASS_ORDER)


def ff_to_class(ff: float) -> int:
    """Convert a fat fraction value [0, 1] to class index."""
    for i, threshold in enumerate(CLASS_THRESHOLDS):
        if ff < threshold:
            return i
    return N_CLASSES - 1


def load_study_cases(test_set_path: Path, exclude_path: Path) -> list:
    """Load study test set cases, excluding patients from the exclusion list."""
    with open(test_set_path, 'r') as f:
        study = yaml.safe_load(f)

    excluded_ids = set()
    if exclude_path.exists():
        with open(exclude_path, 'r') as f:
            for line in f:
                pid = line.strip()
                if pid:
                    excluded_ids.add(pid)

    cases = []
    for case in study['cases']:
        pid = str(case['patient_id'])
        if pid not in excluded_ids:
            cases.append({
                'patient_id': pid,
                'median_ff': float(case['median_ff']),
            })

    return cases


def run_inference(
    cases: list,
    checkpoint_path: Path,
    data_dir: Path,
    device: torch.device,
    batch_size: int = 2,
    num_workers: int = 2,
) -> list:
    """Run scalar regression inference on the study cases."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    model_config = config.get('model', {})
    data_cfg = config.get('data', {})

    # Detect old vs new model: old model uses in_channels=1 and single-arg forward
    in_channels = model_config.get('in_channels', 1)
    base_channels = model_config.get('base_channels', 16)
    use_old_model = (in_channels == 1)

    if use_old_model:
        model = get_old_scalar_model(in_channels=in_channels, base_channels=base_channels).to(device)
    else:
        model = get_scalar_model(in_channels=in_channels, base_channels=base_channels).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    patient_ids = [c['patient_id'] for c in cases]
    dataset = LiverFatScalarDataset(
        data_dir=str(data_dir),
        patient_ids=patient_ids,
        t2_suffix=data_cfg.get('t2_suffix', '_t2_original'),
        ff_suffix=data_cfg.get('ff_suffix', '_ff_normalized'),
        mask_suffix=data_cfg.get('mask_suffix', '_segmentation'),
        input_mask_suffix=data_cfg.get('input_mask_suffix', '_t2_original_segmentation'),
        use_subdirs=data_cfg.get('use_subdirs', False),
        use_patient_subdirs=data_cfg.get('use_patient_subdirs', True),
        t2_subdir=data_cfg.get('t2_subdir', 't2_images'),
        ff_subdir=data_cfg.get('ff_subdir', 'fat_fraction_maps'),
        mask_subdir=data_cfg.get('mask_subdir', 'liver_masks'),
        normalize_t2=data_cfg.get('normalize_t2', True),
        normalize_ff=data_cfg.get('normalize_ff', True),
        mask_erosion=data_cfg.get('mask_erosion', 3),
        augment=False,
        validate_files=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=pad_collate_scalar,
    )

    predictions = {}
    with torch.no_grad():
        for batch in tqdm(loader, desc="Running inference"):
            t2 = batch['t2'].to(device)
            if use_old_model:
                output = model(t2).squeeze(1).cpu().numpy()
            else:
                mask = batch['mask'].to(device)
                output = model(t2, mask).squeeze(1).cpu().numpy()
            for pid, pred in zip(batch['patient_id'], output):
                predictions[pid] = float(np.clip(pred, 0.0, 1.0))

    # Build results aligned with cases
    results = []
    for case in cases:
        pid = case['patient_id']
        if pid in predictions:
            results.append({
                'patient_id': pid,
                'gt_ff': case['median_ff'],
                'pred_ff': predictions[pid],
                'gt_class': ff_to_class(case['median_ff']),
                'pred_class': ff_to_class(predictions[pid]),
            })

    return results


def compute_confusion_matrix(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    for g, p in zip(gt, pred):
        cm[g, p] += 1
    return cm


def make_title_page(results: list, checkpoint_name: str):
    gt = np.array([r['gt_class'] for r in results])
    pred = np.array([r['pred_class'] for r in results])
    correct = int(np.sum(gt == pred))
    incorrect = len(results) - correct
    accuracy = correct / len(results) if results else 0

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')
    ax.text(0.5, 0.70, 'Scalar Regression Model',
            ha='center', va='center', fontsize=24, fontweight='bold')
    ax.text(0.5, 0.62, 'Study Viewer Test Set Report',
            ha='center', va='center', fontsize=18)
    ax.text(0.5, 0.50, f'Model: {checkpoint_name}',
            ha='center', va='center', fontsize=11, color='gray')
    ax.text(0.5, 0.40, f'Total patients: {len(results)}',
            ha='center', va='center', fontsize=13)
    ax.text(0.5, 0.33, f'Correct: {correct}  |  Incorrect: {incorrect}',
            ha='center', va='center', fontsize=13)
    accuracy_pct = accuracy * 100
    ax.text(0.5, 0.24, f'Overall Accuracy: {accuracy_pct:.1f}%',
            ha='center', va='center', fontsize=16, fontweight='bold',
            color='#2e7d32' if accuracy_pct >= 70 else '#c62828')

    # Regression metrics
    gt_ff = np.array([r['gt_ff'] for r in results]) * 100
    pred_ff = np.array([r['pred_ff'] for r in results]) * 100
    mae = np.mean(np.abs(gt_ff - pred_ff))
    bias = np.mean(pred_ff - gt_ff)
    corr = np.corrcoef(gt_ff, pred_ff)[0, 1] if len(gt_ff) > 1 else 0
    ax.text(0.5, 0.15, f'MAE: {mae:.2f}%  |  Bias: {bias:+.2f}%  |  Pearson r: {corr:.3f}',
            ha='center', va='center', fontsize=12)

    ax.text(0.5, 0.05, f'Report generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
            ha='center', va='center', fontsize=10, color='gray')
    plt.tight_layout()
    return fig


def make_confusion_matrix_page(gt: np.ndarray, pred: np.ndarray):
    cm = compute_confusion_matrix(gt, pred)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm.astype(float), row_sums, where=row_sums != 0) * 100

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=100)

    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            count = cm[i, j]
            pct = cm_norm[i, j]
            color = 'white' if pct > 50 else 'black'
            ax.text(j, i, f'{count}\n({pct:.0f}%)', ha='center', va='center',
                    color=color, fontsize=11)

    ax.set_xticks(range(N_CLASSES))
    ax.set_yticks(range(N_CLASSES))
    ax.set_xticklabels(CLASS_LABELS, rotation=30, ha='right')
    ax.set_yticklabels(CLASS_LABELS)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Ground Truth', fontsize=12)

    accuracy = np.mean(gt == pred) * 100
    ax.set_title(f'Confusion Matrix  (Accuracy: {accuracy:.1f}%)', fontsize=14, fontweight='bold')

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Row %')
    plt.tight_layout()
    return fig


def make_metrics_page(gt: np.ndarray, pred: np.ndarray):
    cm = compute_confusion_matrix(gt, pred)

    lines = []
    lines.append(f"{'Class':<22} {'N (GT)':>8} {'N (Pred)':>10} {'Precision':>11} {'Recall':>11} {'F1':>11}")
    lines.append("-" * 75)

    for i, label in enumerate(CLASS_LABELS):
        n_gt = int(np.sum(gt == i))
        n_pred = int(np.sum(pred == i))
        tp = cm[i, i]
        precision = tp / n_pred * 100 if n_pred > 0 else 0
        recall = tp / n_gt * 100 if n_gt > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        lines.append(f"{label:<22} {n_gt:>8} {n_pred:>10} {precision:>10.1f}% {recall:>10.1f}% {f1:>10.1f}%")

    lines.append("")
    accuracy = np.mean(gt == pred) * 100
    adjacent = np.mean(np.abs(gt - pred) <= 1) * 100
    lines.append(f"Overall Accuracy:           {accuracy:.1f}%")
    lines.append(f"Adjacent Accuracy (+/-1):   {adjacent:.1f}%")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    ax.set_title('Per-Class Classification Metrics', fontsize=14, fontweight='bold', pad=20)
    ax.text(0.05, 0.90, "\n".join(lines), transform=ax.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace')
    plt.tight_layout()
    return fig


def make_scatter_page(results: list):
    """Scatter plot of predicted vs ground truth FF."""
    gt_ff = np.array([r['gt_ff'] for r in results]) * 100
    pred_ff = np.array([r['pred_ff'] for r in results]) * 100

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(gt_ff, pred_ff, alpha=0.6, s=30, edgecolors='k', linewidths=0.3)

    # Identity line
    lims = [0, max(gt_ff.max(), pred_ff.max()) * 1.05]
    ax.plot(lims, lims, 'r--', linewidth=1, label='Identity')

    # Class boundaries
    for threshold in CLASS_THRESHOLDS:
        t_pct = threshold * 100
        ax.axvline(t_pct, color='gray', linestyle=':', alpha=0.5)
        ax.axhline(t_pct, color='gray', linestyle=':', alpha=0.5)

    ax.set_xlabel('Ground Truth FF (%)', fontsize=12)
    ax.set_ylabel('Predicted FF (%)', fontsize=12)
    ax.set_title('Predicted vs Ground Truth Fat Fraction', fontsize=14, fontweight='bold')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.legend()
    ax.set_aspect('equal')
    plt.tight_layout()
    return fig


def make_misclassified_page(results: list):
    wrong = [r for r in results if r['gt_class'] != r['pred_class']]
    if not wrong:
        return None

    # Sort by severity of misclassification
    wrong.sort(key=lambda r: abs(r['gt_class'] - r['pred_class']), reverse=True)

    lines = []
    lines.append(f"{'Patient ID':<16} {'GT FF%':>8} {'Pred FF%':>10} {'Ground Truth':<26} {'Predicted':<26} {'Off-by':<6}")
    lines.append("-" * 100)
    for r in wrong:
        gt_label = CLASS_LABELS[r['gt_class']]
        pred_label = CLASS_LABELS[r['pred_class']]
        dist = abs(r['gt_class'] - r['pred_class'])
        lines.append(
            f"{r['patient_id']:<16} {r['gt_ff']*100:>7.2f} {r['pred_ff']*100:>9.2f} "
            f"{gt_label:<26} {pred_label:<26} {dist}"
        )

    header = lines[:2]
    data_lines = lines[2:]
    page_size = 35
    pages = []

    for start in range(0, len(data_lines), page_size):
        chunk = header + data_lines[start:start + page_size]
        page_num = start // page_size + 1
        total_pages = (len(data_lines) + page_size - 1) // page_size
        suffix = f" ({page_num}/{total_pages})" if total_pages > 1 else ""

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')
        ax.set_title(f'Misclassified Patients ({len(wrong)} total){suffix}',
                     fontsize=14, fontweight='bold', pad=20)
        ax.text(0.02, 0.95, "\n".join(chunk), transform=ax.transAxes,
                fontsize=8.5, verticalalignment='top', fontfamily='monospace')
        plt.tight_layout()
        pages.append(fig)

    return pages


def generate_report(results: list, output_path: Path, checkpoint_name: str):
    gt = np.array([r['gt_class'] for r in results])
    pred = np.array([r['pred_class'] for r in results])

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(output_path) as pdf:
        fig = make_title_page(results, checkpoint_name)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        fig = make_confusion_matrix_page(gt, pred)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        fig = make_metrics_page(gt, pred)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        fig = make_scatter_page(results)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        pages = make_misclassified_page(results)
        if pages:
            for fig in pages:
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

    print(f"Report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate scalar regression model on study viewer test set'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='outputs/scalar_regression_run/experiment_20260123_131810/checkpoint_best.pth',
        help='Path to model checkpoint',
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='datasets/patient_data_regression_setup',
        help='Path to patient data directory',
    )
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--num-workers', type=int, default=2)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = (REPO_ROOT / checkpoint_path).resolve()

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = (REPO_ROOT / data_dir).resolve()

    test_set_path = SCRIPT_DIR / 'test_data' / 'study_test_set.yaml'
    exclude_path = SCRIPT_DIR / 'test_data' / 'exluding_patients.txt'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load study cases
    cases = load_study_cases(test_set_path, exclude_path)
    print(f"Study test set: {len(cases)} patients (after exclusions)")

    # Run inference
    results = run_inference(
        cases, checkpoint_path, data_dir, device,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )
    print(f"Inference complete: {len(results)} patients")

    # Generate report
    exp_name = checkpoint_path.parent.name
    output_path = SCRIPT_DIR / 'results' / 'reports' / f'scalar_regression_{exp_name}_report.pdf'
    generate_report(results, output_path, exp_name)

    # Also save raw results as YAML
    yaml_path = output_path.with_suffix('.yaml')
    yaml_results = {
        'model': exp_name,
        'checkpoint': str(checkpoint_path),
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_patients': len(results),
        'results': [
            {
                'patient_id': r['patient_id'],
                'gt_ff': round(r['gt_ff'], 6),
                'pred_ff': round(r['pred_ff'], 6),
                'ground_truth': CLASS_ORDER[r['gt_class']],
                'prediction': CLASS_ORDER[r['pred_class']],
            }
            for r in results
        ],
    }
    with open(yaml_path, 'w') as f:
        yaml.safe_dump(yaml_results, f, sort_keys=False)
    print(f"Results saved to: {yaml_path}")


if __name__ == '__main__':
    main()
