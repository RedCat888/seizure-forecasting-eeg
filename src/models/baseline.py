"""
Baseline classifiers using handcrafted EEG features.

Supports:
- Logistic Regression
- XGBoost
- MLP (scikit-learn)
"""

from typing import Optional, Dict, Any, Tuple
import numpy as np
from pathlib import Path
import pickle
from omegaconf import DictConfig

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


class BaselineClassifier:
    """
    Wrapper for baseline classifiers with common interface.
    """
    
    def __init__(
        self,
        model_type: str = "xgboost",
        cfg: Optional[DictConfig] = None,
        **kwargs
    ):
        """
        Args:
            model_type: One of 'logreg', 'xgboost', 'mlp'
            cfg: Optional config with model parameters
            **kwargs: Additional arguments for the model
        """
        self.model_type = model_type
        self.cfg = cfg
        self.scaler = StandardScaler()
        self.model = None
        
        self._init_model(**kwargs)
    
    def _init_model(self, **kwargs):
        """Initialize the underlying model."""
        if self.cfg is not None:
            base_cfg = self.cfg.baseline
        else:
            base_cfg = None
        
        if self.model_type == "logreg":
            self.model = LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                **kwargs
            )
        
        elif self.model_type == "xgboost":
            if not HAS_XGBOOST:
                raise ImportError("XGBoost not installed. Run: pip install xgboost")
            
            n_estimators = kwargs.get("n_estimators", 
                base_cfg.xgb_n_estimators if base_cfg else 200)
            max_depth = kwargs.get("max_depth",
                base_cfg.xgb_max_depth if base_cfg else 6)
            learning_rate = kwargs.get("learning_rate",
                base_cfg.xgb_learning_rate if base_cfg else 0.1)
            
            self.model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                use_label_encoder=False,
                eval_metric="logloss",
                **kwargs
            )
        
        elif self.model_type == "mlp":
            hidden_dims = kwargs.get("hidden_dims",
                base_cfg.mlp_hidden_dims if base_cfg else [128, 64])
            
            self.model = MLPClassifier(
                hidden_layer_sizes=tuple(hidden_dims),
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                **kwargs
            )
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "BaselineClassifier":
        """
        Fit the classifier.
        
        Args:
            X: Feature matrix [N, F]
            y: Labels [N]
            sample_weight: Optional sample weights [N]
            
        Returns:
            self
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit model
        if sample_weight is not None and self.model_type in ["logreg", "xgboost"]:
            self.model.fit(X_scaled, y, sample_weight=sample_weight)
        else:
            self.model.fit(X_scaled, y)
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Feature matrix [N, F]
            
        Returns:
            Probability matrix [N, 2]
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Feature matrix [N, F]
            
        Returns:
            Predicted labels [N]
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, float]:
        """
        Evaluate model on data.
        
        Args:
            X: Feature matrix [N, F]
            y: True labels [N]
            
        Returns:
            Dict of metrics
        """
        proba = self.predict_proba(X)[:, 1]
        pred = self.predict(X)
        
        auroc = roc_auc_score(y, proba)
        auprc = average_precision_score(y, proba)
        acc = np.mean(pred == y)
        
        # Sensitivity and specificity
        tp = np.sum((pred == 1) & (y == 1))
        tn = np.sum((pred == 0) & (y == 0))
        fp = np.sum((pred == 1) & (y == 0))
        fn = np.sum((pred == 0) & (y == 1))
        
        sensitivity = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        
        return {
            "auroc": auroc,
            "auprc": auprc,
            "accuracy": acc,
            "sensitivity": sensitivity,
            "specificity": specificity,
        }
    
    def save(self, path: str | Path) -> None:
        """Save model to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "wb") as f:
            pickle.dump({
                "model_type": self.model_type,
                "model": self.model,
                "scaler": self.scaler,
            }, f)
    
    @classmethod
    def load(cls, path: str | Path) -> "BaselineClassifier":
        """Load model from file."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        instance = cls(model_type=data["model_type"])
        instance.model = data["model"]
        instance.scaler = data["scaler"]
        
        return instance


def train_baseline_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: DictConfig,
    sample_weight: Optional[np.ndarray] = None,
) -> Tuple[BaselineClassifier, Dict[str, float], Dict[str, float]]:
    """
    Train and evaluate baseline classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        cfg: Config
        sample_weight: Optional sample weights
        
    Returns:
        Tuple of (model, train_metrics, val_metrics)
    """
    model_type = cfg.baseline.get("model_type", "xgboost")
    
    model = BaselineClassifier(model_type=model_type, cfg=cfg)
    model.fit(X_train, y_train, sample_weight=sample_weight)
    
    train_metrics = model.evaluate(X_train, y_train)
    val_metrics = model.evaluate(X_val, y_val)
    
    return model, train_metrics, val_metrics


def compute_sample_weights(
    y_tte: np.ndarray,
    y_cls: np.ndarray,
    preictal_min: float = 10.0,
) -> np.ndarray:
    """
    Compute sample weights based on time-to-event.
    
    Higher weights for preictal windows closer to seizure onset.
    
    Args:
        y_tte: Time-to-event in seconds (-1 for interictal)
        y_cls: Class labels
        preictal_min: Preictal horizon in minutes
        
    Returns:
        Sample weights
    """
    preictal_sec = preictal_min * 60
    
    weights = np.ones(len(y_cls))
    
    # For preictal samples, weight by proximity to onset
    preictal_mask = y_cls == 1
    if np.any(preictal_mask):
        t = y_tte[preictal_mask]
        # Higher weight for windows closer to onset
        # w = 1 + (1 - t / preictal_sec), range [1, 2]
        w = 1 + (1 - t / preictal_sec)
        weights[preictal_mask] = w
    
    return weights
