import sys
import numpy as np
from src.exception import MyException
from src.logger import logging


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
