# # # below code is to check the logging config
# import os
# from src.logger import logging

# from from_root import from_root

# logging.debug("This is a debug message.")
# logging.info("This is an info message.")
# logging.warning("This is a warning message.")
# logging.error("This is an error message.")
# logging.critical("This is a critical message.")

# # print("from_root():", from_root())

# # print("cwd:", os.getcwd())

from src.pipline.training_pipeline import TrainingPipeline

pipeline = TrainingPipeline()
pipeline.run_pipeline()


# from src.pipeline.prediction_pipeline import MovieRecommender


# recommender = MovieRecommender()

# matched_movie, recs = recommender.recommend(" bahubali", top_n=10)

# print(f"\n🎬 Matched Movie: {matched_movie}\n")
# print("📌 Recommendations:")
# print(recs)