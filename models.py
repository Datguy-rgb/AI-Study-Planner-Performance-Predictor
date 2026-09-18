"""Machine Learning pipeline integrating regression, classification, and clustering."""

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

class PerformancePredictorPipeline:
    def __init__(self):
        self.scaler = StandardScaler()
        self.regressor = LinearRegression()
        self.classifier = LogisticRegression(max_iter=1000)
        self.clusterer = KMeans(n_clusters=3, random_state=42, n_init=10)
        self.is_trained = False

    def train(self, X: np.ndarray, y_score: np.ndarray, y_risk: np.ndarray):
        """Trains models using a train-validation split to prevent overfitting."""
        X_train, X_val, y_score_train, y_score_val, y_risk_train, y_risk_val = train_test_split(
            X, y_score, y_risk, test_size=0.2, random_state=42
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # Linear Regression for continuous score prediction
        self.regressor.fit(X_train_scaled, y_score_train)
        val_score_preds = self.regressor.predict(X_val_scaled)
        mse = mean_squared_error(y_score_val, val_score_preds)
        r2 = r2_score(y_score_val, val_score_preds)

        # Logistic Regression for risk classification
        self.classifier.fit(X_train_scaled, y_risk_train)
        val_risk_preds = self.classifier.predict(X_val_scaled)
        acc = accuracy_score(y_risk_val, val_risk_preds)

        # K-Means for unsupervised student clustering
        self.clusterer.fit(X_train_scaled)

        self.is_trained = True
        return {"reg_r2": float(r2), "reg_mse": float(mse), "clf_acc": float(acc)}

    def predict(self, raw_input: np.ndarray):
        """Executes inference across all three models."""
        if not self.is_trained:
            raise RuntimeError("Pipeline must be trained before inference.")

        scaled = self.scaler.transform(raw_input)
        pred_score = float(self.regressor.predict(scaled)[0])
        pred_risk_idx = int(self.classifier.predict(scaled)[0])
        pred_cluster = int(self.clusterer.predict(scaled)[0])

        return {
            "expected_score_range": (round(max(0.0, pred_score - 3.0), 1), 
                                     round(min(100.0, pred_score + 3.0), 1)),
            "risk_index": pred_risk_idx,
            "cluster_id": pred_cluster
        }