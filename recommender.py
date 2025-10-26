import pandas as pd
import ast  # To safely parse stringified lists

# Load dataset
movies = pd.read_csv('movies.csv')

# Display initial info
print("Original columns:", movies.columns)

# Keep only relevant columns
movies = movies[['title', 'overview', 'genres', 'keywords']]

# Fill missing values with empty string
for col in ['overview', 'genres', 'keywords']:
    movies[col] = movies[col].fillna('')

# Function to extract names from JSON-like strings
def extract_names(obj):
    try:
        return " ".join([d['name'] for d in ast.literal_eval(obj)])
    except:
        return ""

# Apply to genres and keywords
movies['genres'] = movies['genres'].apply(extract_names)
movies['keywords'] = movies['keywords'].apply(extract_names)

# Create a combined feature column
movies['combined_features'] = (
    movies['overview'] + " " +
    movies['genres'] + " " +
    movies['keywords']
)

# Show the new dataframe
print("\nData after preprocessing:\n")
print(movies[['title', 'combined_features']].head())
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Step 4: Convert text to TF-IDF matrix
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['combined_features'])

print("\nTF-IDF Matrix Shape:", tfidf_matrix.shape)

# Step 5: Compute cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

print("\nCosine Similarity Matrix Shape:", cosine_sim.shape)
# Reset index to ensure titles line up with similarity matrix
movies = movies.reset_index()
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

# Function to recommend movies
def recommend_movies(title, num_recommendations=5):
    # Check if title exists
    if title not in indices:
        print(f"Movie '{title}' not found in dataset.")
        return

    # Get index of the movie that matches the title
    idx = indices[title]

    # Get pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort movies by similarity score in descending order
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Skip the first one (itself), get next top results
    sim_scores = sim_scores[1:num_recommendations + 1]

    # Get the indices of recommended movies
    movie_indices = [i[0] for i in sim_scores]

    # Print recommended titles
    print(f"\nTop {num_recommendations} movies similar to '{title}':\n")
    print(movies['title'].iloc[movie_indices].to_string(index=False))

# -------- Get input from user --------
user_input = input("\nEnter a movie title to get recommendations: ")
# Improve user input matching using lowercase and partial matches
# -------- Improved Input Matching --------
def search_movie(title):
    title = title.lower()
    matches = [movie for movie in movies['title'] if title in movie.lower()]
    if not matches:
        print(f"\n❌ No matches found for '{title}'.")
        return None
    elif len(matches) == 1:
        return matches[0]
    else:
        print(f"\n🔍 Multiple matches found for '{title}':")
        for i, match in enumerate(matches):
            print(f"{i + 1}. {match}")
        try:
            choice = int(input("Enter the number of the correct movie: "))
            return matches[choice - 1]
        except:
            print("⚠️ Invalid choice.")
            return None

# -------- Run the recommender with improved input --------
user_input = input("\nEnter a movie title to get recommendations: ")
matched_title = search_movie(user_input)
if matched_title:
    recommend_movies(matched_title)

