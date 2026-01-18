import sys
import os
import json
import shutil
from datetime import datetime
import numpy as np
from src.exception import MyException
from src.logger import logging
from src.constants import (
    BEST_MODEL_DIR,
    BEST_MODEL_METRICS_PATH,
    GENRE_PRECISION_THRESHOLD,
    TFIDF_VECTORIZER_PATH,
    TFIDF_MATRIX_PATH,
    COSINE_SIMILARITY_PATH,
    TRAINING_DF_PATH
)


class RecommenderEvaluation:
    def __init__(self, df, cosine_sim, recommend_fn):
        """
        df          : SAME dataframe used for TF-IDF training
        cosine_sim  : cosine similarity matrix
        recommend_fn: recommend_by_index(idx, top_n)
        """
        self.df = df
        self.cosine_sim = cosine_sim
        self.recommend_fn = recommend_fn

    # ==================================================
    # Precision / Recall / F1 @ K
    # ==================================================
    def precision_recall_f1_at_k(self, k=10):
        try:
            tp = fp = fn = 0
            n = len(self.df)

            for idx in range(n):
                # Ground truth from cosine similarity
                true_indices = [
                    i for i, _ in sorted(
                        enumerate(self.cosine_sim[idx]),
                        key=lambda x: x[1],
                        reverse=True
                    )[1 : k + 1]
                ]

                preds = self.recommend_fn(idx, top_n=k)
                if preds is None or preds.empty:
                    continue

                pred_indices = [
                    self.df.index[self.df["title"] == t][0]
                    for t in preds["title"]
                ]

                true_vec = np.zeros(n)
                pred_vec = np.zeros(n)

                true_vec[true_indices] = 1
                pred_vec[pred_indices] = 1

                tp += np.sum((true_vec == 1) & (pred_vec == 1))
                fp += np.sum((true_vec == 0) & (pred_vec == 1))
                fn += np.sum((true_vec == 1) & (pred_vec == 0))

            precision = tp / (tp + fp + 1e-6)
            recall = tp / (tp + fn + 1e-6)
            f1 = 2 * precision * recall / (precision + recall + 1e-6)

            logging.info(f"Precision@{k}: {precision:.4f}")
            logging.info(f"Recall@{k}: {recall:.4f}")
            logging.info(f"F1@{k}: {f1:.4f}")

            return precision, recall, f1

        except Exception as e:
            raise MyException(e, sys)

    # ==================================================
    # Genre Precision @ K
    # ==================================================
    def genre_precision_at_k(self, k=10):
        try:
            total = 0
            match = 0

            for idx in range(len(self.df)):
                base_genre_val = self.df.iloc[idx]["genres"]
                if not isinstance(base_genre_val, str) or not base_genre_val:
                    continue
                base_genres = set(base_genre_val.split())

                preds = self.recommend_fn(idx, top_n=k)
                if preds is None or preds.empty:
                    continue

                for _, row in preds.iterrows():
                    rec_genre_val = row["genres"]
                    if not isinstance(rec_genre_val, str) or not rec_genre_val:
                        continue
                    total += 1
                    rec_genres = set(rec_genre_val.split())
                    if base_genres & rec_genres:
                        match += 1

            precision = match / total if total else 0.0

            logging.info(f"Genre Precision@{k}: {precision:.4f}")
            logging.info(f"  Matched: {match}/{total} recommendations")

            return precision

        except Exception as e:
            raise MyException(e, sys)

    # ==================================================
    # Best-Model Selection (Genre Precision)
    # ==================================================
    def _load_best_metrics(self):
        if os.path.exists(BEST_MODEL_METRICS_PATH):
            with open(BEST_MODEL_METRICS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def _save_best_metrics(self, metrics: dict) -> None:
        os.makedirs(BEST_MODEL_DIR, exist_ok=True)
        with open(BEST_MODEL_METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    def _copy_best_artifacts(self) -> None:
        os.makedirs(BEST_MODEL_DIR, exist_ok=True)
        shutil.copy2(TFIDF_VECTORIZER_PATH, os.path.join(BEST_MODEL_DIR, os.path.basename(TFIDF_VECTORIZER_PATH)))
        shutil.copy2(TFIDF_MATRIX_PATH, os.path.join(BEST_MODEL_DIR, os.path.basename(TFIDF_MATRIX_PATH)))
        shutil.copy2(COSINE_SIMILARITY_PATH, os.path.join(BEST_MODEL_DIR, os.path.basename(COSINE_SIMILARITY_PATH)))
        if os.path.exists(TRAINING_DF_PATH):
            shutil.copy2(TRAINING_DF_PATH, os.path.join(BEST_MODEL_DIR, os.path.basename(TRAINING_DF_PATH)))

    def update_best_model_if_needed(
        self,
        precision: float,
        recall: float,
        f1: float,
        genre_precision: float,
        k: int = 10
    ) -> bool:
        try:
            best_metrics = self._load_best_metrics()
            best_genre_precision = 0.0
            if best_metrics and "genre_precision_at_k" in best_metrics:
                best_genre_precision = best_metrics["genre_precision_at_k"] or 0.0

            candidate_genre_precision = genre_precision or 0.0

            if candidate_genre_precision < GENRE_PRECISION_THRESHOLD:
                logging.info(
                    f"Candidate genre precision {candidate_genre_precision:.4f} "
                    f"below threshold {GENRE_PRECISION_THRESHOLD:.2f}. Not updating best model."
                )
                return False

            if best_metrics and candidate_genre_precision <= best_genre_precision:
                logging.info(
                    f"Candidate genre precision {candidate_genre_precision:.4f} "
                    f"not better than best {best_genre_precision:.4f}."
                )
                return False

            self._copy_best_artifacts()
            metrics_payload = {
                "precision_at_k": precision,
                "recall_at_k": recall,
                "f1_at_k": f1,
                "genre_precision_at_k": candidate_genre_precision,
                "k": k,
                "updated_at": datetime.utcnow().isoformat()
            }
            self._save_best_metrics(metrics_payload)

            logging.info(
                f"Best model updated. GenrePrecision@{k}={candidate_genre_precision:.4f}"
            )
            return True

        except Exception as e:
            raise MyException(e, sys)
