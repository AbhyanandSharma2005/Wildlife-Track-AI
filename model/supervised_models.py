"""
model/supervised_models.py
Classical supervised learning models trained on CNN-extracted embeddings.
All models are wrapped with a common interface for training, prediction and
serialisation.

Models included:
  1. Random Forest (RF)
  2. Support Vector Machine with RBF kernel (SVM)
  3. K-Nearest Neighbours (KNN)
  4. Gradient Boosting Machine (GBM)
  5. Logistic Regression (LR)
"""
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ── Canonical model registry ──────────────────────────────────────────────────
# Each entry: display_name → (estimator, safe_filename_stem)
_MODEL_REGISTRY: dict[str, tuple] = {
    'Random Forest': (
        RandomForestClassifier(n_estimators=200, max_depth=None,
                               random_state=42, n_jobs=-1),
        'Random_Forest',
    ),
    'SVM (RBF)': (
        SVC(kernel='rbf', C=10, gamma='scale',
            probability=True, random_state=42),
        'SVM_RBF',
    ),
    'KNN': (
        KNeighborsClassifier(n_neighbors=7, metric='cosine'),
        'KNN',
    ),
    'Gradient Boosting': (
        GradientBoostingClassifier(n_estimators=150, learning_rate=0.1,
                                   max_depth=5, random_state=42),
        'Gradient_Boosting',
    ),
    'Logistic Regression': (
        LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs',
                           multi_class='multinomial', random_state=42),
        'Logistic_Regression',
    ),
}


def get_all_models() -> dict[str, tuple]:
    """
    Return a fresh dict of {display_name: (sklearn_estimator, safe_name)}.
    Estimators are re-instantiated so the registry stays clean.
    """
    return {
        name: (estimator.__class__(**estimator.get_params()), safe_name)
        for name, (estimator, safe_name) in _MODEL_REGISTRY.items()
    }


def build_pipeline(estimator) -> Pipeline:
    """
    Wrap an estimator in a StandardScaler → Estimator pipeline.
    Handles feature scaling internally.
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('clf',    estimator),
    ])


def get_display_names() -> list[str]:
    """Return ordered list of model display names."""
    return list(_MODEL_REGISTRY.keys())


def safe_name_to_display(safe_name: str) -> str:
    """Reverse-map a filename stem to its display name."""
    for disp, (_, sn) in _MODEL_REGISTRY.items():
        if sn == safe_name:
            return disp
    return safe_name.replace('_', ' ')
