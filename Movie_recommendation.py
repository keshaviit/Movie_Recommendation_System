import streamlit as st
import pickle
import requests

# Load data
movies = pickle.load(open(
    'anaconda_projects_9c4ad3d0-7ea1-4062-a737-4b775f3df230_movie_list.pkl', 'rb'
))
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()
similarity = cosine_similarity(vectors)

st.title("🎬 Movie Recommender System")

# Fetch poster
def fetch_poster(movie_id):
    url = (
        f"https://api.themoviedb.org/3/movie/{movie_id}"
        f"?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
    )
    data = requests.get(url).json()
    poster_path = data.get('poster_path')
    if poster_path:
        return "https://image.tmdb.org/t/p/w500/" + poster_path
    return ""

# Recommendation function
def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]

    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommended_movies = []
    recommended_movies_posters = []

    for i in movies_list:
        movie_id = movies.iloc[i[0]]['id']

        recommended_movies.append(movies.iloc[i[0]]['title'])
        recommended_movies_posters.append(fetch_poster(movie_id))

    return recommended_movies, recommended_movies_posters

# Select movie
selected_movie = st.selectbox(
    "Select a movie",
    movies['title'].values
)

# Button
if st.button("Recommend"):
    recommended_movie_names, recommended_movie_posters = recommend(selected_movie)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(recommended_movie_names[0])
        st.image(recommended_movie_posters[0])
    with col2:
        st.text(recommended_movie_names[1])
        st.image(recommended_movie_posters[1])

    with col3:
        st.text(recommended_movie_names[2])
        st.image(recommended_movie_posters[2])
    with col4:
        st.text(recommended_movie_names[3])
        st.image(recommended_movie_posters[3])
    with col5:
        st.text(recommended_movie_names[4])
        st.image(recommended_movie_posters[4])




    
