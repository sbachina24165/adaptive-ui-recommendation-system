# Adaptive UI Movie Recommender

This project demonstrates an adaptive user interface using AI-related techniques.

## What makes it adaptive
The system changes its recommendation behavior based on user feedback:
- If the user likes the recommendations, the system continues favoring similar genres.
- If the user does not like the recommendations, the system tries to avoid those genres.

## AI-related techniques used
- TF-IDF vectorization
- Cosine similarity
- Feedback-based adaptive behavior

## Files
- `adaptive_recommender.py` - main adaptive recommender program
- `movies.csv` - sample movie dataset
- `requirements.txt` - required Python packages

## Run
```bash
pip install -r requirements.txt
python adaptive_recommender.py
```

## Example flow
1. Enter a movie title such as `Batman Begins`
2. Review recommendations
3. Answer `yes` or `no`
4. The system adapts the next round based on your response
