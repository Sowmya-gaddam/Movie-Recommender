import streamlit as st
import pandas as pd
import requests
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# 🎨 UI STYLE
# -------------------------------
st.markdown("""
<style>
.stApp {background-color: #141414; color: white;}
h1 {text-align:center; color:#e50914;}

div.stButton > button {
    background-color:#e50914;
    color:white;
    border-radius:6px;
    padding:10px 20px;
}

div.stButton > button:hover {
    background-color:#b20710;
}

div[data-baseweb="select"] {
    background-color:white !important;
    border-radius:6px;
}

div[data-baseweb="select"] > div {
    color:black !important;
}

.movie-card {
    text-align:center;
    padding:10px;
    border-radius:10px;
    transition:0.3s;
}
.movie-card:hover {transform:scale(1.05);}

.movie-title {font-size:14px; font-weight:bold;}
.movie-info {font-size:12px; color:lightgray;}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    movies = pd.read_csv('tmdb_5000_movies.csv')
    movies = movies[['title','overview','genres','keywords','release_date','vote_average']]
    movies['overview'] = movies['overview'].fillna('')
    return movies

movies = load_data()

# -------------------------------
# CONVERT JSON
# -------------------------------
def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return " ".join(L)

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

# -------------------------------
# TAGS
# -------------------------------
movies['tags'] = movies['overview'] + " " + movies['genres'] + " " + movies['keywords']

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()
similarity = cosine_similarity(vectors)

# -------------------------------
# API
# -------------------------------
API_KEY = "0862abeccba99a99ea1e939a92cb9ae1"

def fetch_movie_data(movie):
    try:
        url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={movie}"
        data = requests.get(url).json()

        if data['results']:
            result = data['results'][0]
            poster = result.get('poster_path')

            poster_url = "https://image.tmdb.org/t/p/w500/" + poster if poster else None

            # 🎬 improved trailer search
            trailer_url = f"https://www.youtube.com/results?search_query={movie}+official+trailer"

            return poster_url, trailer_url
    except:
        pass

    return None, None

# -------------------------------
# RECOMMEND
# -------------------------------
def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = similarity[index]

    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])

    results = []

    for i in movie_list[1:]:
        row = movies.iloc[i[0]]
        title = row.title

        poster, trailer = fetch_movie_data(title)

        if poster:
            results.append({
                "title": title,
                "poster": poster,
                "rating": round(row.vote_average,1),
                "year": str(row.release_date)[:4],
                "trailer": trailer
            })

        if len(results) == 5:
            break

    return results

# -------------------------------
# UI
# -------------------------------
st.set_page_config(layout="wide")

st.title("🎬 Movie Recommender")
st.markdown("### Discover movies you'll love ❤️")

selected_movie = st.selectbox("Choose a movie", movies['title'].values)

if st.button("Recommend"):
    with st.spinner("Finding best movies for you... 🎬"):
        results = recommend(selected_movie)

    cols = st.columns(5)

    for i, movie in enumerate(results):
        with cols[i]:
            st.markdown(f"""
                <div class="movie-card">
                    <img src="{movie['poster']}" width="150">
                    <div class="movie-title">{movie['title'][:20]}...</div>
                    <div class="movie-info">⭐ {movie['rating']} / 10 • {movie['year']}</div>
                </div>
            """, unsafe_allow_html=True)

            # 🎬 TRAILER BUTTON
            st.markdown(f"""
                <a href="{movie['trailer']}" target="_blank">
                    <button style="
                        background-color:#e50914;
                        color:white;
                        border:none;
                        padding:6px 12px;
                        margin-top:6px;
                        border-radius:5px;
                        cursor:pointer;">
                        ▶ Watch Trailer
                    </button>
                </a>
            """, unsafe_allow_html=True)
