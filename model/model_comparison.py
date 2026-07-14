"""
model/model_comparison.py
Utilities for generating model comparison charts and extended benchmark reports.
Reads from saved metrics.json and returns chart-ready data structures.
"""
import json
from pathlib import Path
from data.dataset_loader import SPECIES

SAVED_MODELS_DIR = Path('saved_models')

# Canonical model display order
MODEL_ORDER = [
    'CNN (MobileNetV2)',
    'Random Forest',
    'SVM (RBF)',
    'KNN',
    'Gradient Boosting',
    'Logistic Regression',
]

# Colour palette (used by Chart.js frontend)
MODEL_COLORS = {
    'CNN (MobileNetV2)':    'rgba(0, 255, 136, 0.85)',
    'Random Forest':        'rgba(124, 58, 237, 0.85)',
    'SVM (RBF)':            'rgba(245, 158, 11, 0.85)',
    'KNN':                  'rgba(239, 68, 68, 0.85)',
    'Gradient Boosting':    'rgba(59, 130, 246, 0.85)',
    'Logistic Regression':  'rgba(236, 72, 153, 0.85)',
}


def load_metrics() -> dict | None:
    p = SAVED_MODELS_DIR / 'metrics.json'
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def get_chart_data() -> dict:
    """
    Return Chart.js-ready data for the accuracy bar chart and radar chart.
    """
    metrics = load_metrics()
    if not metrics:
        return {}

    ordered = [m for m in MODEL_ORDER if m in metrics]

    return {
        'bar_chart': {
            'labels':   ordered,
            'accuracy': [metrics[m]['accuracy']  for m in ordered],
            'f1':       [metrics[m]['f1']         for m in ordered],
            'colors':   [MODEL_COLORS.get(m, 'rgba(200,200,200,0.8)') for m in ordered],
        },
        'radar_chart': {
            'models': ordered,
            'datasets': [
                {
                    'label':  m,
                    'data':   [
                        metrics[m].get('accuracy',  0),
                        metrics[m].get('precision', 0),
                        metrics[m].get('recall',    0),
                        metrics[m].get('f1',        0),
                    ],
                    'color': MODEL_COLORS.get(m, 'rgba(200,200,200,0.8)'),
                }
                for m in ordered
            ],
            'axes': ['Accuracy', 'Precision', 'Recall', 'F1'],
        },
        'table': [
            {
                'model':     m,
                'accuracy':  metrics[m]['accuracy'],
                'precision': metrics[m]['precision'],
                'recall':    metrics[m]['recall'],
                'f1':        metrics[m]['f1'],
                'color':     MODEL_COLORS.get(m, '#aaa'),
                'rank':      i + 1,
            }
            for i, m in enumerate(
                sorted(ordered, key=lambda x: metrics[x]['accuracy'], reverse=True)
            )
        ],
        'winner': max(ordered, key=lambda m: metrics[m]['accuracy'])
                  if ordered else 'N/A',
    }


def get_full_report() -> dict:
    """Return full benchmark report including metrics + chart data."""
    metrics   = load_metrics()
    chart_data = get_chart_data()
    return {
        'metrics':    metrics,
        'chart_data': chart_data,
        'species':    SPECIES,
    }
