import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("movies.csv")
    df = df[['title', 'overview', 'genres', 'keywords']].fillna('')

    def extract_names(obj):
        try:
            return " ".join([d['name'] for d in ast.literal_eval(obj)])
        except:
            return ""

    df['genres'] = df['genres'].apply(extract_names)
    df['keywords'] = df['keywords'].apply(extract_names)
    df['combined_features'] = df['overview'] + ' ' + df['genres'] + ' ' + df['keywords']
    return df

# Compute similarity
@st.cache_resource
def compute_similarity(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return sim_matrix

# Recommend function
def recommend(title, df, sim_matrix, num=5):
    title = title.lower()
    matches = [t for t in df['title'] if title in t.lower()]
    if not matches:
        return ["No matches found."]
    idx = df[df['title'] == matches[0]].index[0]
    sim_scores = list(enumerate(sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num+1]
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices].tolist()

# UI
st.title("🎬 Movie Recommendation System")

df = load_data()
sim_matrix = compute_similarity(df)

movie_input = st.text_input("Enter a movie title:", "")

if movie_input:
    recommendations = recommend(movie_input, df, sim_matrix)
    st.write(f"Top similar movies to **{movie_input}**:")
    for i, rec in enumerate(recommendations, start=1):
        st.write(f"{i}. {rec}")
