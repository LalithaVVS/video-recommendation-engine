from flask import Flask, jsonify, request
from recommender import VideoRecommender

app = Flask(__name__)

# Load the recommender once when server starts
# Think of this like turning on the engine before driving
recommender = VideoRecommender('data/ratings.csv')


@app.route('/')
def home():
    return jsonify({
        "service": "Video Recommendation Engine",
        "endpoints": {
            "recommendations": "/recommend/<user_id>",
            "similar_users": "/similar-users/<user_id>"
        }
    })


@app.route('/recommend/<int:user_id>')
def recommend(user_id):
    """
    GET /recommend/1
    Returns top 5 video recommendations for user 1
    """
    top_n = request.args.get('top_n', default=5, type=int)
    result = recommender.get_recommendations(user_id, top_n)
    return jsonify(result)


@app.route('/similar-users/<int:user_id>')
def similar_users(user_id):
    """
    GET /similar-users/1
    Returns the 3 most similar users to user 1
    """
    result = recommender.get_similar_users(user_id)
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, port=5000)