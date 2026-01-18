import os
import sys
import numpy as np
import pandas as pd
import unicodedata
import re
from difflib import SequenceMatcher

from src.exception import MyException
from src.logger import logging
from src.constants import COSINE_SIMILARITY_PATH
from src.entity.config_entity import RecommenderModelConfig


# =====================================================
# Text normalization (accents, symbols, acronyms)
# =====================================================
def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = re.sub(r"[.\-_:]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def seq_ratio(a, b):
    return SequenceMatcher(None, a, b).ratio()


def jaccard(a, b):
    if not a or not b:
        return 0
    return len(a & b) / len(a | b)


# =====================================================
# Movie Recommender
# =====================================================
class MovieRecommender:
    def __init__(self):
        try:
            from src.entity.config_entity import RecommenderModelConfig

            logging.info("Loading recommender artifacts")

            model_config = RecommenderModelConfig()
            self.model_dir = model_config.model_dir

            # Load SAME df used for training
            self.df = pd.read_csv(
                os.path.join(self.model_dir, "training_df.csv")
            )

            self.cosine_sim = np.load(model_config.cosine_similarity_path)

            # Search helpers (for USER queries only)
            self.df["title_norm"] = self.df["title"].apply(normalize_text)
            self.df["title_tokens"] = self.df["title_norm"].apply(
                lambda x: set(x.split())
            )

            self.df["rating"] = self.df["rating"].fillna(
                self.df["rating"].mean()
            )

            logging.info(f"DataFrame shape: {self.df.shape}")
            logging.info(f"Cosine similarity shape: {self.cosine_sim.shape}")
            logging.info("Recommender artifacts loaded successfully")

        except Exception as e:
            raise MyException(e, sys)

    # ==================================================
    # ✅ INDEX-BASED RECOMMEND (FOR EVALUATION ONLY)
    # ==================================================
    def recommend_by_index(self, idx: int, top_n: int = 10):
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(
            sim_scores, key=lambda x: x[1], reverse=True
        )[1 : top_n + 1]

        movie_indices = [i[0] for i in sim_scores]

        return self.df.iloc[movie_indices][
            ["title", "genres", "rating", "poster_url"]
        ]

    # ==================================================
    # (KEEP your existing recommend(title) for users)
    # ==================================================


    # -------------------------------------------------
    def _load_latest_dataframe(self) -> pd.DataFrame:
        try:
            for root, _, files in os.walk("src/artifacts"):
                for file in files:
                    if file == "movies.csv":
                        df = pd.read_csv(os.path.join(root, file))
                        logging.info(f"Loaded movies.csv from {os.path.join(root, file)}")
                        return df
            raise Exception("movies.csv not found in artifacts")
        except Exception as e:
            raise MyException(e, sys)

    # -------------------------------------------------
    def find_movie(self, query: str):
        q = normalize_text(query)
        q_tokens = set(q.split())
        q_acronym = " ".join(list(q))  # kgf → k g f

        # ---------- SHORT TITLES ----------
        if len(q) <= 4:
            exact = self.df[self.df["title_norm"] == q]
            if not exact.empty:
                return exact.iloc[0]["title"]

            sub = self.df[self.df["title_norm"].str.contains(q, regex=False)]
            if not sub.empty:
                return sub.sort_values("vote_count", ascending=False).iloc[0]["title"]

            acro = self.df[self.df["title_norm"].str.startswith(q_acronym)]
            if not acro.empty:
                return acro.sort_values("vote_count", ascending=False).iloc[0]["title"]

        # ---------- NORMAL TITLES ----------
        exact = self.df[self.df["title_norm"] == q]
        if not exact.empty:
            return exact.iloc[0]["title"]

        sub = self.df[self.df["title_norm"].str.contains(q, regex=False)]
        if not sub.empty:
            return sub.sort_values("vote_count", ascending=False).iloc[0]["title"]

        acro = self.df[self.df["title_norm"].str.startswith(q_acronym)]
        if not acro.empty:
            return acro.sort_values("vote_count", ascending=False).iloc[0]["title"]

        # ---------- FUZZY FALLBACK ----------
        best_title = None
        best_score = 0

        for _, row in self.df.iterrows():
            row_tokens = set(row["title_tokens"]) if isinstance(row["title_tokens"], list) \
                else set(str(row["title_tokens"]).split())

            expanded_tokens = q_tokens | set(q_acronym.split())

            score = (
                0.7 * jaccard(expanded_tokens, row_tokens)
                + 0.3 * seq_ratio(q, row["title_norm"])
            )

            if score > best_score and score > 0.5:
                best_score = score
                best_title = row["title"]

        return best_title

    # -------------------------------------------------
    def recommend(self, movie_name: str, top_n: int = 10):
        try:
            logging.info(f"Generating recommendations for input: {movie_name}")

            matched_title = self.find_movie(movie_name)
            if matched_title is None:
                logging.warning(f"Movie not found for query: {movie_name}")
                raise Exception("Movie not found")

            idx_list = self.df.index[self.df["title"] == matched_title].tolist()
            if not idx_list:
                raise Exception(f"Title '{matched_title}' not found in dataframe")

            idx = idx_list[0]

            if idx >= self.cosine_sim.shape[0]:
                logging.error(
                    f"Index {idx} out of bounds for cosine_sim shape {self.cosine_sim.shape}"
                )
                raise Exception(f"Invalid index: {idx} >= {self.cosine_sim.shape[0]}")

            sim_scores = list(enumerate(self.cosine_sim[idx]))
            sim_scores = sorted(
                sim_scores, key=lambda x: x[1], reverse=True
            )[1 : top_n + 1]

            movie_indices = [i[0] for i in sim_scores]

            recommendations = self.df.iloc[movie_indices][
                ["title", "genres", "rating", "poster_url"]
            ]

            return matched_title, recommendations

        except Exception as e:
            raise MyException(e, sys)