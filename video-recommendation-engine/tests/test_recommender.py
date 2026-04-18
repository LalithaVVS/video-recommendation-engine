import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recommender import VideoRecommender

# Use our sample data for tests
@pytest.fixture
def recommender():
    return VideoRecommender('data/ratings.csv')


def test_recommendations_returned_for_valid_user(recommender):
    result = recommender.get_recommendations(1)
    assert "recommendations" in result
    assert len(result["recommendations"]) > 0


def test_recommendations_exclude_watched_videos(recommender):
    # User 1 has watched cooking_101 - it should NOT appear in recommendations
    result = recommender.get_recommendations(1)
    video_ids = [r["video_id"] for r in result["recommendations"]]
    assert "cooking_101" not in video_ids


def test_invalid_user_returns_error(recommender):
    result = recommender.get_recommendations(999)
    assert "error" in result


def test_similar_users_returned(recommender):
    result = recommender.get_similar_users(1)
    assert "similar_users" in result
    assert len(result["similar_users"]) > 0


def test_user1_similar_to_user2(recommender):
    # User 1 and User 2 both love cooking - they should be similar
    result = recommender.get_similar_users(1)
    similar_ids = [u["user_id"] for u in result["similar_users"]]
    assert 2 in similar_ids