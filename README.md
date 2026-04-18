# Video Recommendation Engine

A content recommendation system built in Python using 
collaborative filtering — the same core technique used 
by YouTube, Netflix, and Spotify.

## How It Works
1. Tracks which videos each user has watched and rated
2. Calculates similarity between users using cosine similarity
3. Recommends unwatched videos from similar users

## Tech Stack
- Python 3.10
- Flask (REST API)
- Pandas + NumPy (data processing)
- Scikit-learn (cosine similarity)

## API Endpoints
- GET /recommend/<user_id> — get video recommendations
- GET /similar-users/<user_id> — find similar users

## Example
curl http://localhost:5000/recommend/1

## Run Locally
pip install flask numpy pandas scikit-learn
python3 app.py
