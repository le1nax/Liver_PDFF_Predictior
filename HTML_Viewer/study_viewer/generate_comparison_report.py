"""
Generate a PDF report comparing Doctor vs AI (scalar regression) classification
performance, stratified by radiologic labels (steatosis, cirrhosis, etc.).

Reads:
  - Doctor results:  results/study_20260220_113311.yaml
  - AI results:      results/reports/scalar_regression_experiment_20260123_131810_report.yaml
  - Patient labels:  datasets/patient_data_regression_setup/labels.yaml
  - Study test set:  test_data/study_test_set.yaml  (for median FF values)

Usage:
    python HTML_Viewer/study_viewer/generate_comparison_report.py
    python HTML_Viewer/study_viewer/generate_comparison_report.py \
        --doctor results/study_20260220_113311.yaml \
        --ai results/reports/scalar_regression_experiment_20260123_131810_report.yaml \
        --labels ../../datasets/patient_data_regression_setup/labels.yaml
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

SCRIPT_DIR = Path(__file__).resolve().parent

CLASS_ORDER = ['healthy', 'mild', 'moderate', 'severe']
CLASS_LABELS = ['Healthy\n(0-5%)', 'Mild\n(5-15%)', 'Moderate\n(15-25%)', 'Severe\n(>25%)']
CLASS_LABELS_SHORT = ['Healthy', 'Mild', 'Moderate', 'Severe']
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_ORDER)}
N_CLASSES = len(CLASS_ORDER)

FINDING_LABELS = ['steatosis', 'cirrhosis', 'cholestasis', 'high_iron', 'metastasis']
FINDING_DISPLAY = {
    'steatosis': 'Steatosis',
    'cirrhosis': 'Cirrhosis',
    'cholestasis': 'Cholestasis',
    'high_iron': 'High Iron',
    'metastasis': 'Metastasis',
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_labels(labels_path: Path) -> dict:
    """Load labels.yaml into {patient_id: {label: bool, ...}}."""
    with open(labels_path) as f:
        raw = yaml.safe_load(f)
    merged: dict[str, dict[str, bool]] = {}
    for entry in raw:
        pid = str(entry['id'])
        if pid not in merged:
            merged[pid] = {lbl: bool(entry.get(lbl, False)) for lbl in FINDING_LABELS}
        else:
            for lbl in FINDING_LABELS:
                merged[pid][lbl] = merged[pid][lbl] or bool(entry.get(lbl, False))
    return merged


def load_ff_lookup(test_set_path: Path) -> dict:
    """Load median FF values from study_test_set.yaml keyed by patient ID."""
    if not test_set_path.exists():
        return {}
    with open(test_set_path) as f:
        study = yaml.safe_load(f)
    return {str(c['patient_id']): float(c['median_ff']) for c in study['cases']}


def load_doctor_results(yaml_path: Path) -> list[dict]:
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    results = []
    for r in data['results']:
        results.append({
            'patient_id': str(r['patient_id']),
            'gt': r['ground_truth'],
            'pred': r['classification'],
        })
    return results


def load_ai_results(yaml_path: Path) -> dict[str, str]:
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    return {str(r['patient_id']): r['prediction'] for r in data['results']}


def merge_data(doctor_results, ai_lookup, labels, ff_lookup):
    """Build unified per-patient records."""
    records = []
    for r in doctor_results:
        pid = r['patient_id']
        ai_pred = ai_lookup.get(pid)
        if ai_pred is None:
            continue
        patient_labels = labels.get(pid, {})
        rec = {
            'patient_id': pid,
            'gt': r['gt'],
            'gt_idx': CLASS_TO_IDX[r['gt']],
            'doctor_pred': r['pred'],
            'doctor_idx': CLASS_TO_IDX[r['pred']],
            'ai_pred': ai_pred,
            'ai_idx': CLASS_TO_IDX[ai_pred],
            'has_labels': pid in labels,
            'gt_ff': ff_lookup.get(pid),
        }
        for lbl in FINDING_LABELS:
            rec[lbl] = patient_labels.get(lbl, False)
        # "clean" = no pathologic findings at all
        rec['clean'] = not any(rec[lbl] for lbl in FINDING_LABELS)
        records.append(rec)
    return records


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def compute_confusion_matrix(gt, pred, n=N_CLASSES):
    cm = np.zeros((n, n), dtype=int)
    for g, p in zip(gt, pred):
        cm[g, p] += 1
    return cm


def accuracy(gt, pred):
    gt, pred = np.asarray(gt), np.asarray(pred)
    return np.mean(gt == pred) * 100 if len(gt) > 0 else 0.0


def adjacent_accuracy(gt, pred):
    gt, pred = np.asarray(gt), np.asarray(pred)
    return np.mean(np.abs(gt - pred) <= 1) * 100 if len(gt) > 0 else 0.0


def per_class_recall(gt, pred):
    gt, pred = np.asarray(gt), np.asarray(pred)
    recalls = []
    for i in range(N_CLASSES):
        mask = gt == i
        if mask.sum() == 0:
            recalls.append(0.0)
        else:
            recalls.append(np.mean(pred[mask] == i) * 100)
    return recalls


# ---------------------------------------------------------------------------
# PDF pages
# ---------------------------------------------------------------------------

def make_title_page(records):
    n = len(records)
    n_labeled = sum(1 for r in records if r['has_labels'])
    doc_acc = accuracy([r['gt_idx'] for r in records], [r['doctor_idx'] for r in records])
    ai_acc = accuracy([r['gt_idx'] for r in records], [r['ai_idx'] for r in records])

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')

    ax.text(0.5, 0.75, 'Doctor vs AI Comparison Report',
            ha='center', va='center', fontsize=24, fontweight='bold')
    ax.text(0.5, 0.67, 'with Radiologic Findings Stratification',
            ha='center', va='center', fontsize=16, color='#555555')

    ax.text(0.5, 0.55, f'Patients compared: {n}   |   With labels: {n_labeled}',
            ha='center', va='center', fontsize=13)

    # Side-by-side accuracy
    ax.text(0.30, 0.42, 'Doctor', ha='center', va='center', fontsize=14, fontweight='bold')
    ax.text(0.70, 0.42, 'AI (Scalar Regression)', ha='center', va='center', fontsize=14, fontweight='bold')
    doc_color = '#2e7d32' if doc_acc >= 70 else '#c62828'
    ai_color = '#2e7d32' if ai_acc >= 70 else '#c62828'
    ax.text(0.30, 0.34, f'{doc_acc:.1f}%', ha='center', va='center',
            fontsize=22, fontweight='bold', color=doc_color)
    ax.text(0.70, 0.34, f'{ai_acc:.1f}%', ha='center', va='center',
            fontsize=22, fontweight='bold', color=ai_color)

    # Label counts
    label_counts = {lbl: sum(1 for r in records if r[lbl]) for lbl in FINDING_LABELS}
    clean_count = sum(1 for r in records if r['clean'] and r['has_labels'])
    label_str = '   '.join(f'{FINDING_DISPLAY[lbl]}: {label_counts[lbl]}' for lbl in FINDING_LABELS)
    ax.text(0.5, 0.22, f'Finding prevalence (N={n_labeled} labeled):',
            ha='center', va='center', fontsize=11, color='#555555')
    ax.text(0.5, 0.17, label_str,
            ha='center', va='center', fontsize=11)
    ax.text(0.5, 0.12, f'Clean (no findings): {clean_count}',
            ha='center', va='center', fontsize=11)

    ax.text(0.5, 0.03, f'Report generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
            ha='center', va='center', fontsize=10, color='gray')
    plt.tight_layout()
    return fig


def make_comparison_confusion_page(records, suptitle='Overall Confusion Matrices', subtitle=None):
    """Side-by-side confusion matrices for doctor and AI on a given subset."""
    if not records:
        return None

    gt = [r['gt_idx'] for r in records]
    doc = [r['doctor_idx'] for r in records]
    ai = [r['ai_idx'] for r in records]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, pred, title in [(axes[0], doc, 'Doctor'), (axes[1], ai, 'AI (Scalar Regression)')]:
        cm = compute_confusion_matrix(gt, pred)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm.astype(float), row_sums, where=row_sums != 0) * 100

        im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=100)
        for i in range(N_CLASSES):
            for j in range(N_CLASSES):
                color = 'white' if cm_norm[i, j] > 50 else 'black'
                ax.text(j, i, f'{cm[i, j]}\n({cm_norm[i, j]:.0f}%)',
                        ha='center', va='center', color=color, fontsize=10)

        ax.set_xticks(range(N_CLASSES))
        ax.set_yticks(range(N_CLASSES))
        ax.set_xticklabels(CLASS_LABELS, fontsize=9)
        ax.set_yticklabels(CLASS_LABELS, fontsize=9)
        ax.set_xlabel('Predicted', fontsize=11)
        ax.set_ylabel('Ground Truth', fontsize=11)

        acc = accuracy(gt, pred)
        ax.set_title(f'{title}\nAccuracy: {acc:.1f}%', fontsize=13, fontweight='bold')

    sup = suptitle
    if subtitle:
        sup += f'\n{subtitle}'
    fig.suptitle(sup, fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def make_per_class_comparison_page(records):
    """Bar chart comparing Doctor vs AI recall per class."""
    gt = np.array([r['gt_idx'] for r in records])
    doc = np.array([r['doctor_idx'] for r in records])
    ai = np.array([r['ai_idx'] for r in records])

    doc_recall = per_class_recall(gt, doc)
    ai_recall = per_class_recall(gt, ai)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(N_CLASSES)
    width = 0.35

    bars1 = ax.bar(x - width / 2, doc_recall, width, label='Doctor',
                   color='#1976D2', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width / 2, ai_recall, width, label='AI',
                   color='#F57C00', edgecolor='black', linewidth=0.5)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., h + 1,
                    f'{h:.1f}%', ha='center', va='bottom', fontsize=10)

    # Per-class N counts
    class_counts = Counter(gt)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{CLASS_LABELS_SHORT[i]}\n(N={class_counts.get(i, 0)})'
                        for i in range(N_CLASSES)], fontsize=11)
    ax.set_ylabel('Recall (%)', fontsize=12)
    ax.set_title('Per-Class Recall: Doctor vs AI', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig


def make_findings_stratified_page(records):
    """Accuracy stratified by radiologic finding (present vs absent)."""
    # Build subsets: for each finding, split into has/doesn't have
    # Only use labeled patients
    labeled = [r for r in records if r['has_labels']]
    if not labeled:
        return None

    findings_to_show = [lbl for lbl in FINDING_LABELS
                        if sum(1 for r in labeled if r[lbl]) >= 3]

    categories = ['clean'] + findings_to_show
    display_names = ['Clean\n(no findings)'] + [FINDING_DISPLAY[f] for f in findings_to_show]

    doc_accs = []
    ai_accs = []
    counts = []

    for cat in categories:
        if cat == 'clean':
            subset = [r for r in labeled if r['clean']]
        else:
            subset = [r for r in labeled if r[cat]]

        n = len(subset)
        counts.append(n)
        if n == 0:
            doc_accs.append(0)
            ai_accs.append(0)
            continue

        gt = [r['gt_idx'] for r in subset]
        doc_accs.append(accuracy(gt, [r['doctor_idx'] for r in subset]))
        ai_accs.append(accuracy(gt, [r['ai_idx'] for r in subset]))

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(x - width / 2, doc_accs, width, label='Doctor',
                   color='#1976D2', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width / 2, ai_accs, width, label='AI',
                   color='#F57C00', edgecolor='black', linewidth=0.5)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2., h + 1,
                        f'{h:.0f}%', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([f'{display_names[i]}\n(N={counts[i]})'
                        for i in range(len(categories))], fontsize=10)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Accuracy by Radiologic Finding', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig


def make_findings_detail_page(records):
    """Text table: per-finding breakdown of N, accuracy, per-class recall."""
    labeled = [r for r in records if r['has_labels']]
    if not labeled:
        return None

    lines = []
    lines.append(f'{"Subgroup":<18} {"N":>4}  {"Doctor":>8} {"AI":>8}  '
                 f'{"Doc Hlthy":>10} {"Doc Mild":>10} {"Doc Mod":>10} {"Doc Sev":>10}  '
                 f'{"AI Hlthy":>10} {"AI Mild":>10} {"AI Mod":>10} {"AI Sev":>10}')
    lines.append('=' * 140)

    def add_row(label, subset):
        if len(subset) == 0:
            return
        gt = np.array([r['gt_idx'] for r in subset])
        doc = np.array([r['doctor_idx'] for r in subset])
        ai = np.array([r['ai_idx'] for r in subset])
        doc_acc = accuracy(gt, doc)
        ai_acc = accuracy(gt, ai)
        doc_rcl = per_class_recall(gt, doc)
        ai_rcl = per_class_recall(gt, ai)

        # Format recall with N count: "80%(4/5)"
        def fmt_recall(rcl_list, gt_arr):
            parts = []
            for i in range(N_CLASSES):
                n = int((gt_arr == i).sum())
                if n == 0:
                    parts.append(f'{"--":>10}')
                else:
                    correct = int(((gt_arr == i) & (doc if 'Doc' in label else ai) == i).sum()) if False else 0
                    parts.append(f'{rcl_list[i]:>5.0f}%({n})')
            return ''.join(parts)

        doc_parts = ''.join(
            f'{rcl:>5.0f}%({int((gt == i).sum())})' if (gt == i).sum() > 0 else f'{"--":>10}'
            for i, rcl in enumerate(doc_rcl)
        )
        ai_parts = ''.join(
            f'{rcl:>5.0f}%({int((gt == i).sum())})' if (gt == i).sum() > 0 else f'{"--":>10}'
            for i, rcl in enumerate(ai_rcl)
        )

        lines.append(f'{label:<18} {len(subset):>4}  {doc_acc:>7.1f}% {ai_acc:>7.1f}%  {doc_parts}  {ai_parts}')

    add_row('ALL (labeled)', labeled)
    add_row('Clean', [r for r in labeled if r['clean']])
    lines.append('-' * 140)
    for lbl in FINDING_LABELS:
        subset = [r for r in labeled if r[lbl]]
        if len(subset) >= 1:
            add_row(FINDING_DISPLAY[lbl], subset)
    # Multi-finding
    multi = [r for r in labeled if sum(r[lbl] for lbl in FINDING_LABELS) >= 2]
    if multi:
        lines.append('-' * 140)
        add_row('Multi-finding', multi)

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis('off')
    ax.set_title('Detailed Accuracy by Radiologic Finding', fontsize=14, fontweight='bold', pad=20)
    ax.text(0.02, 0.92, '\n'.join(lines), transform=ax.transAxes,
            fontsize=7.5, verticalalignment='top', fontfamily='monospace')
    plt.tight_layout()
    return fig


def make_agreement_page(records):
    """Doctor vs AI agreement analysis."""
    both_correct = []
    both_wrong = []
    doc_only = []
    ai_only = []

    for r in records:
        dc = r['doctor_idx'] == r['gt_idx']
        ac = r['ai_idx'] == r['gt_idx']
        if dc and ac:
            both_correct.append(r)
        elif dc and not ac:
            doc_only.append(r)
        elif not dc and ac:
            ai_only.append(r)
        else:
            both_wrong.append(r)

    n = len(records)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Pie chart of agreement
    ax = axes[0]
    sizes = [len(both_correct), len(both_wrong), len(doc_only), len(ai_only)]
    labels_pie = [
        f'Both correct\n({sizes[0]})',
        f'Both wrong\n({sizes[1]})',
        f'Doctor only\n({sizes[2]})',
        f'AI only\n({sizes[3]})',
    ]
    colors = ['#4CAF50', '#F44336', '#2196F3', '#FF9800']
    ax.pie(sizes, labels=labels_pie, colors=colors, autopct='%1.0f%%',
           startangle=90, textprops={'fontsize': 11})
    ax.set_title('Agreement Breakdown', fontsize=13, fontweight='bold')

    # For the "both wrong" cases, show finding distribution
    ax = axes[1]
    if both_wrong:
        labeled_wrong = [r for r in both_wrong if r['has_labels']]
        finding_counts = {FINDING_DISPLAY[lbl]: sum(1 for r in labeled_wrong if r[lbl])
                          for lbl in FINDING_LABELS}
        finding_counts['Clean'] = sum(1 for r in labeled_wrong if r['clean'])

        names = list(finding_counts.keys())
        vals = list(finding_counts.values())
        y_pos = np.arange(len(names))
        ax.barh(y_pos, vals, color='#EF5350', edgecolor='black', linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=11)
        ax.set_xlabel('Count', fontsize=11)
        ax.set_title(f'Findings in "Both Wrong" Cases (N={len(both_wrong)})',
                     fontsize=13, fontweight='bold')
        for i, v in enumerate(vals):
            if v > 0:
                ax.text(v + 0.2, i, str(v), va='center', fontsize=10)
        ax.grid(axis='x', alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No cases where both were wrong',
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.set_title('Findings in "Both Wrong" Cases', fontsize=13, fontweight='bold')

    plt.tight_layout()
    return fig


def make_error_by_finding_page(records):
    """For each finding, show which direction errors go (under vs over-estimation)."""
    labeled = [r for r in records if r['has_labels']]
    if not labeled:
        return None

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    subsets = [('Clean', [r for r in labeled if r['clean']])]
    for lbl in FINDING_LABELS:
        subset = [r for r in labeled if r[lbl]]
        if subset:
            subsets.append((FINDING_DISPLAY[lbl], subset))

    for idx, (ax, (name, subset)) in enumerate(zip(axes, subsets)):
        if not subset:
            ax.axis('off')
            continue

        doc_errors = [r['doctor_idx'] - r['gt_idx'] for r in subset if r['doctor_idx'] != r['gt_idx']]
        ai_errors = [r['ai_idx'] - r['gt_idx'] for r in subset if r['ai_idx'] != r['gt_idx']]

        bins_range = np.arange(-3.5, 4.5, 1)
        ax.hist(doc_errors, bins=bins_range, alpha=0.6, color='#1976D2',
                label=f'Doctor ({len(doc_errors)} err)', edgecolor='black', linewidth=0.5)
        ax.hist(ai_errors, bins=bins_range, alpha=0.6, color='#F57C00',
                label=f'AI ({len(ai_errors)} err)', edgecolor='black', linewidth=0.5)

        n_sub = len(subset)
        doc_acc = accuracy([r['gt_idx'] for r in subset], [r['doctor_idx'] for r in subset])
        ai_acc = accuracy([r['gt_idx'] for r in subset], [r['ai_idx'] for r in subset])
        ax.set_title(f'{name} (N={n_sub})\nDoc {doc_acc:.0f}% | AI {ai_acc:.0f}%',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('Prediction - GT (negative = under-estimate)', fontsize=8)
        ax.set_ylabel('Count', fontsize=9)
        ax.legend(fontsize=8)
        ax.set_xticks(range(-3, 4))
        ax.grid(axis='y', alpha=0.3)

    # Hide unused axes
    for i in range(len(subsets), len(axes)):
        axes[i].axis('off')

    fig.suptitle('Error Direction by Finding (negative = under-estimation)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def make_misclassified_with_findings_page(records):
    """List misclassified patients with their findings."""
    # Patients where at least one of doctor/AI got it wrong
    wrong = [r for r in records if r['doctor_idx'] != r['gt_idx'] or r['ai_idx'] != r['gt_idx']]
    if not wrong:
        return []

    # Sort by GT class, then by most severe disagreement
    wrong.sort(key=lambda r: (
        r['gt_idx'],
        -max(abs(r['doctor_idx'] - r['gt_idx']), abs(r['ai_idx'] - r['gt_idx']))
    ))

    lines = []
    lines.append(
        f'{"Patient ID":<14} {"GT":>9} {"Doctor":>9} {"AI":>9}  '
        f'{"Steat":>5} {"Cirrh":>5} {"Chol":>5} {"Iron":>5} {"Meta":>5}  '
        f'{"GT FF%":>7}'
    )
    lines.append('=' * 100)

    for r in wrong:
        gt_lbl = CLASS_LABELS_SHORT[r['gt_idx']]
        doc_lbl = CLASS_LABELS_SHORT[r['doctor_idx']]
        ai_lbl = CLASS_LABELS_SHORT[r['ai_idx']]

        doc_mark = doc_lbl if r['doctor_idx'] != r['gt_idx'] else f'  {doc_lbl}'
        ai_mark = ai_lbl if r['ai_idx'] != r['gt_idx'] else f'  {ai_lbl}'

        # Findings flags
        def flag(val):
            return '  X  ' if val else '  .  '

        ff_str = f'{r["gt_ff"] * 100:.1f}' if r['gt_ff'] is not None else '  --'

        lines.append(
            f'{r["patient_id"]:<14} {gt_lbl:>9} {doc_mark:>9} {ai_mark:>9}  '
            f'{flag(r["steatosis"])}{flag(r["cirrhosis"])}{flag(r["cholestasis"])}'
            f'{flag(r["high_iron"])}{flag(r["metastasis"])}  '
            f'{ff_str:>7}'
        )

    # Paginate
    header = lines[:2]
    data_lines = lines[2:]
    page_size = 40
    pages = []

    for start in range(0, len(data_lines), page_size):
        chunk = header + data_lines[start:start + page_size]
        page_num = start // page_size + 1
        total_pages = (len(data_lines) + page_size - 1) // page_size
        suffix = f' ({page_num}/{total_pages})' if total_pages > 1 else ''

        fig, ax = plt.subplots(figsize=(14, 9))
        ax.axis('off')
        ax.set_title(f'Misclassified Patients with Findings ({len(wrong)} total){suffix}',
                     fontsize=14, fontweight='bold', pad=20)
        ax.text(0.02, 0.95, '\n'.join(chunk), transform=ax.transAxes,
                fontsize=8, verticalalignment='top', fontfamily='monospace')
        plt.tight_layout()
        pages.append(fig)

    return pages


# ---------------------------------------------------------------------------
# Main report generation
# ---------------------------------------------------------------------------

def generate_report(records, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(output_path) as pdf:
        # 1. Title page
        fig = make_title_page(records)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # 2. Side-by-side confusion matrices (all patients)
        fig = make_comparison_confusion_page(records, suptitle='Overall Confusion Matrices')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # 3. Filtered confusion matrices (exclude cirrhosis, cholestasis, high iron)
        EXCLUDE_FINDINGS = ['cirrhosis', 'cholestasis', 'high_iron']
        filtered = [r for r in records
                    if r['has_labels'] and not any(r[lbl] for lbl in EXCLUDE_FINDINGS)]
        excluded_n = sum(1 for r in records if r['has_labels']) - len(filtered)
        fig = make_comparison_confusion_page(
            filtered,
            suptitle='Filtered Confusion Matrices',
            subtitle=f'Excluding cirrhosis, cholestasis, high iron  (N={len(filtered)}, {excluded_n} removed)',
        )
        if fig:
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

        # 4. Per-class recall comparison
        fig = make_per_class_comparison_page(records)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # 5. Accuracy stratified by finding (bar chart)
        fig = make_findings_stratified_page(records)
        if fig:
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

        # 6. Detailed findings table
        fig = make_findings_detail_page(records)
        if fig:
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

        # 7. Agreement analysis
        fig = make_agreement_page(records)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # 8. Error direction by finding
        fig = make_error_by_finding_page(records)
        if fig:
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

        # 9. Misclassified patients with findings
        pages = make_misclassified_with_findings_page(records)
        for fig in pages:
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    print(f'Report saved to: {output_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Generate Doctor vs AI comparison report with radiologic findings stratification'
    )
    parser.add_argument(
        '--doctor', type=Path,
        default=SCRIPT_DIR / 'results' / 'study_20260220_113311.yaml',
        help='Path to doctor study results YAML',
    )
    parser.add_argument(
        '--ai', type=Path,
        default=SCRIPT_DIR / 'results' / 'reports' / 'scalar_regression_experiment_20260123_131810_report.yaml',
        help='Path to AI results YAML',
    )
    parser.add_argument(
        '--labels', type=Path,
        default=SCRIPT_DIR.parents[1] / 'datasets' / 'patient_data_regression_setup' / 'labels.yaml',
        help='Path to patient labels YAML',
    )
    parser.add_argument(
        '--output', type=Path, default=None,
        help='Output PDF path (default: results/reports/doctor_vs_ai_comparison_report.pdf)',
    )
    args = parser.parse_args()

    output_path = args.output or (SCRIPT_DIR / 'results' / 'reports' / 'doctor_vs_ai_comparison_report.pdf')

    # Load data
    print(f'Loading doctor results:  {args.doctor}')
    doctor_results = load_doctor_results(args.doctor)
    print(f'  {len(doctor_results)} patients')

    print(f'Loading AI results:      {args.ai}')
    ai_lookup = load_ai_results(args.ai)
    print(f'  {len(ai_lookup)} patients')

    print(f'Loading labels:          {args.labels}')
    labels = load_labels(args.labels)
    print(f'  {len(labels)} unique patients')

    test_set_path = SCRIPT_DIR / 'test_data' / 'study_test_set.yaml'
    ff_lookup = load_ff_lookup(test_set_path)
    print(f'  FF lookup: {len(ff_lookup)} patients')

    # Merge
    records = merge_data(doctor_results, ai_lookup, labels, ff_lookup)
    print(f'\nMerged records: {len(records)} patients')
    print(f'  With labels: {sum(1 for r in records if r["has_labels"])}')
    print(f'  Clean (no findings): {sum(1 for r in records if r["clean"] and r["has_labels"])}')

    # Generate report
    generate_report(records, output_path)


if __name__ == '__main__':
    main()
