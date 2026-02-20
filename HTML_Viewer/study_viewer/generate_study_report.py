"""
Generate a PDF report from study viewer classification results.

Usage:
    python HTML_Viewer/study_viewer/generate_study_report.py HTML_Viewer/study_viewer/results/study_YYYYMMDD_HHMMSS.yaml
"""

import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from datetime import datetime


# Ordered class definitions
CLASS_ORDER = ['healthy', 'mild', 'moderate', 'severe']
CLASS_LABELS = ['Healthy (0-5%)', 'Mild (5-15%)', 'Moderate (15-25%)', 'Severe (>25%)']
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_ORDER)}
N_CLASSES = len(CLASS_ORDER)


def load_study_results(yaml_path: str) -> dict:
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


def build_arrays(study: dict):
    """Extract ground truth and prediction class index arrays."""
    gt_indices = []
    pred_indices = []
    patient_ids = []
    for r in study['results']:
        gt_indices.append(CLASS_TO_IDX[r['ground_truth']])
        pred_indices.append(CLASS_TO_IDX[r['classification']])
        patient_ids.append(str(r['patient_id']))
    return np.array(gt_indices), np.array(pred_indices), patient_ids


def compute_confusion_matrix(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    for g, p in zip(gt, pred):
        cm[g, p] += 1
    return cm


def make_title_page(study: dict):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')
    ax.text(0.5, 0.65, 'Study Classification Report',
            ha='center', va='center', fontsize=24, fontweight='bold')
    ax.text(0.5, 0.50, f'Submitted: {study["submitted_at"]}',
            ha='center', va='center', fontsize=12)
    ax.text(0.5, 0.40, f'Total classified: {study["total_classified"]}',
            ha='center', va='center', fontsize=13)
    ax.text(0.5, 0.33, f'Correct: {study["correct"]}  |  Incorrect: {study["incorrect"]}',
            ha='center', va='center', fontsize=13)
    accuracy_pct = study['accuracy'] * 100
    ax.text(0.5, 0.24, f'Overall Accuracy: {accuracy_pct:.1f}%',
            ha='center', va='center', fontsize=16, fontweight='bold',
            color='#2e7d32' if accuracy_pct >= 70 else '#c62828')
    ax.text(0.5, 0.10, f'Report generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
            ha='center', va='center', fontsize=10, color='gray')
    plt.tight_layout()
    return fig


def make_confusion_matrix_page(gt: np.ndarray, pred: np.ndarray):
    cm = compute_confusion_matrix(gt, pred)

    # Row-normalize for percentages
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
    ax.set_xlabel('Classified As', fontsize=12)
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
    lines.append(f"Adjacent Accuracy (±1):     {adjacent:.1f}%")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    ax.set_title('Per-Class Classification Metrics', fontsize=14, fontweight='bold', pad=20)
    ax.text(0.05, 0.90, "\n".join(lines), transform=ax.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace')
    plt.tight_layout()
    return fig


def make_misclassified_page(gt: np.ndarray, pred: np.ndarray, patient_ids: list):
    wrong_mask = gt != pred
    if not np.any(wrong_mask):
        return None

    indices = np.where(wrong_mask)[0]
    # Sort by severity of misclassification (largest class distance first)
    class_dist = np.abs(gt[indices] - pred[indices])
    sorted_order = np.argsort(class_dist)[::-1]
    indices = indices[sorted_order]

    lines = []
    lines.append(f"{'Patient ID':<16} {'Classified As':<26} {'Ground Truth':<26} {'Off-by':<6}")
    lines.append("-" * 76)
    for idx in indices:
        pid = patient_ids[idx]
        pred_label = CLASS_LABELS[pred[idx]]
        gt_label = CLASS_LABELS[gt[idx]]
        dist = abs(int(gt[idx]) - int(pred[idx]))
        lines.append(f"{pid:<16} {pred_label:<26} {gt_label:<26} {dist}")

    # Split into pages of ~40 lines each
    header = lines[:2]
    data_lines = lines[2:]
    page_size = 40
    pages = []

    for start in range(0, len(data_lines), page_size):
        chunk = header + data_lines[start:start + page_size]
        page_num = start // page_size + 1
        total_pages = (len(data_lines) + page_size - 1) // page_size
        suffix = f" ({page_num}/{total_pages})" if total_pages > 1 else ""

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.axis('off')
        ax.set_title(f'Misclassified Patients ({len(indices)} total){suffix}',
                     fontsize=14, fontweight='bold', pad=20)
        ax.text(0.02, 0.95, "\n".join(chunk), transform=ax.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace')
        plt.tight_layout()
        pages.append(fig)

    return pages


def generate_report(yaml_path: str):
    study = load_study_results(yaml_path)
    gt, pred, patient_ids = build_arrays(study)

    # Output path
    yaml_p = Path(yaml_path)
    reports_dir = yaml_p.parent / 'reports'
    reports_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = reports_dir / f'{yaml_p.stem}_report.pdf'

    with PdfPages(pdf_path) as pdf:
        # Title page
        fig = make_title_page(study)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # Confusion matrix
        fig = make_confusion_matrix_page(gt, pred)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # Per-class metrics
        fig = make_metrics_page(gt, pred)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # Misclassified patients
        pages = make_misclassified_page(gt, pred, patient_ids)
        if pages:
            for fig in pages:
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

    print(f"Report saved to: {pdf_path}")
    return pdf_path


def main():
    parser = argparse.ArgumentParser(description='Generate PDF report from study viewer results')
    parser.add_argument('yaml_file', type=str, help='Path to study results YAML file')
    args = parser.parse_args()
    generate_report(args.yaml_file)


if __name__ == '__main__':
    main()
