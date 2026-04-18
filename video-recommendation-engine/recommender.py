import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class VideoRecommender:
    def __init__(self, ratings_path):
        """
        Load data and build the user similarity matrix.
        This runs once when the server starts.
        """
        # Load the CSV file into a DataFrame (like a spreadsheet in memory)
        self.ratings_df = pd.read_csv(ratings_path)

        # Build the user-video matrix
        # Rows = users, Columns = videos, Values = ratings
        # Missing ratings are filled with 0
        self.user_video_matrix = self.ratings_df.pivot_table(
            index='user_id',
            columns='video_id',
            values='rating',
            fill_value=0
        )

        # Calculate how similar each user is to every other user
        self.similarity_matrix = self._calculate_similarity()

    def _calculate_similarity(self):
        """
        Cosine similarity measures the angle between two users'
        taste vectors. Score of 1 = identical taste, 0 = nothing in common.
        """
        similarity = cosine_similarity(self.user_video_matrix)

        # Wrap result in a DataFrame for easy lookup by user_id
        return pd.DataFrame(
            similarity,
            index=self.user_video_matrix.index,
            columns=self.user_video_matrix.index
        )

    def get_recommendations(self, user_id, top_n=5):
        """
        For a given user, find similar users and recommend
        videos the user hasn't watched yet.
        """
        # Check if user exists
        if user_id not in self.user_video_matrix.index:
            return {"error": f"User {user_id} not found"}

        # Get similarity scores for this user vs all others
        # Sort by most similar, skip the first (that's the user themselves)
        similar_users = self.similarity_matrix[user_id].sort_values(
            ascending=False
        )[1:]

        # Videos this user has already watched (rating > 0)
        watched_videos = set(
            self.user_video_matrix.loc[user_id][
                self.user_video_matrix.loc[user_id] > 0
            ].index
        )

        # Score each unwatched video
        video_scores = {}

        for similar_user, similarity_score in similar_users.items():
            # Get videos this similar user has watched
            similar_user_ratings = self.user_video_matrix.loc[similar_user]

            for video, rating in similar_user_ratings.items():
                # Skip videos user already watched
                if video in watched_videos:
                    continue
                # Skip videos the similar user hasn't watched
                if rating == 0:
                    continue

                # Weighted score: higher similarity = more influence
                if video not in video_scores:
                    video_scores[video] = 0
                video_scores[video] += similarity_score * rating

        # Sort by score, return top N
        recommended = sorted(
            video_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]

        return {
            "user_id": user_id,
            "recommendations": [
                {"video_id": video, "score": round(float(score), 2)}
                for video, score in recommended
            ]
        }

    def get_similar_users(self, user_id):
        """
        Return the top 3 most similar users to the given user.
        Useful for debugging and understanding the system.
        """
        if user_id not in self.similarity_matrix.index:
            return {"error": f"User {user_id} not found"}

        similar = self.similarity_matrix[user_id].sort_values(
            ascending=False
        )[1:4]  # skip self, take next 3

        return {
            "user_id": user_id,
            "similar_users": [
                {"user_id": int(uid), "similarity": round(float(score), 2)}
                for uid, score in similar.items()
            ]
        }