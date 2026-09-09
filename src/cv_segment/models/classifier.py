"""LightGBM Pairwise Classifier Wrapper.

Assigned Engineer: Engineer 3
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np


class LightGBMPairwiseClassifier:
    """Inference wrapper for LightGBM trained on 36-dimensional tabular pairwise features."""

    def __init__(self, model_path: Optional[Path] = None) -> None:
        self.model_path = model_path
        self.model: Optional[Any] = None
        if model_path and model_path.exists():
            self.load_model(model_path)

    def load_model(self, path: Path) -> None:
        """Loads serialized LightGBM booster."""
        try:
            import lightgbm as lgb
            self.model = lgb.Booster(model_file=str(path))
        except Exception:
            self.model = None

    def predict_split_probability(self, feature_vector: dict[str, float]) -> float:
        """Predicts probability P(boundary = 1) between adjacent pages."""
        if self.model is None:
            # Fallback heuristic probability if model file is not yet trained
            jaccard = feature_vector.get("token_jaccard_sim", 0.0)
            delta_pii = feature_vector.get("delta_pii_score", 0.0)
            # High Jaccard -> same CV; High delta PII -> split
            prob = 0.5 + (0.25 * delta_pii) - (0.3 * jaccard)
            return float(np.clip(prob, 0.0, 1.0))

        features_array = np.array([list(feature_vector.values())], dtype=np.float32)
        pred = self.model.predict(features_array)[0]
        return float(pred)
