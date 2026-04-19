import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_movies(csv_path: str = "movies.csv") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required_cols = {"title", "genre", "description"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["combined_features"] = (
        df["genre"].fillna("") + " " + df["description"].fillna("")
    )
    return df


def build_similarity(df: pd.DataFrame):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(df["combined_features"])
    similarity_matrix = cosine_similarity(tfidf_matrix)
    return similarity_matrix


def find_movie_index(df: pd.DataFrame, movie_title: str):
    exact_match = df[df["title"].str.lower() == movie_title.strip().lower()]
    if not exact_match.empty:
        return exact_match.index[0]

    partial_match = df[df["title"].str.lower().str.contains(movie_title.strip().lower(), na=False)]
    if not partial_match.empty:
        return partial_match.index[0]

    return None


def get_recommendations(df: pd.DataFrame, similarity_matrix, movie_index: int, top_n: int = 5):
    scores = list(enumerate(similarity_matrix[movie_index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    recs = []
    for idx, score in scores[1: top_n + 1]:
        recs.append({
            "title": df.loc[idx, "title"],
            "genre": df.loc[idx, "genre"],
            "score": round(float(score), 3),
        })
    return recs


def adaptive_recommendation_loop():
    print("\nAdaptive Movie Recommendation System")
    print("-" * 40)
    print("This interface adapts based on your feedback.")

    df = load_movies()
    similarity_matrix = build_similarity(df)

    current_input = input("Enter a movie title: ").strip()
    if not current_input:
        print("No input provided.")
        return

    movie_index = find_movie_index(df, current_input)
    if movie_index is None:
        print("Movie not found.")
        return

    liked_genres = []
    disliked_genres = []

    for round_num in range(1, 3):
        selected_title = df.loc[movie_index, "title"]
        print(f"\nRound {round_num}: Recommendations based on {selected_title}")
        recs = get_recommendations(df, similarity_matrix, movie_index, top_n=5)

        filtered_recs = []
        for rec in recs:
            rec_genre = rec["genre"].lower()
            if disliked_genres and any(g in rec_genre for g in disliked_genres):
                continue
            filtered_recs.append(rec)

        if not filtered_recs:
            filtered_recs = recs

        for i, rec in enumerate(filtered_recs[:5], start=1):
            print(f"{i}. {rec['title']} | {rec['genre']} | Similarity: {rec['score']}")

        feedback = input("\nDid you like these recommendations? (yes/no): ").strip().lower()

        if feedback == "yes":
            liked_genres.extend(df.loc[movie_index, "genre"].lower().split())
            print("Great! The system will keep favoring similar genres.")
        else:
            disliked_genres.extend(df.loc[movie_index, "genre"].lower().split())
            print("Okay. The system will try to avoid those genres next.")

        next_title = input("Enter one of the recommended movies to continue, or press Enter to stop: ").strip()
        if not next_title:
            break

        next_index = find_movie_index(df, next_title)
        if next_index is None:
            print("Movie not found. Stopping.")
            break

        movie_index = next_index

    print("\nThank you for using the adaptive movie recommender.")


if __name__ == "__main__":
    adaptive_recommendation_loop()
