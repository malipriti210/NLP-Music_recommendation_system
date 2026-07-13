import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Music Recommendation",
    page_icon="🎵",
    layout="centered"
)



# ---------------- ATTRACTIVE UI ----------------

st.markdown(
"""
<style>

/* Background */

.stApp {

background:
linear-gradient(
rgba(5,5,25,0.75),
rgba(5,5,25,0.85)
),
url("https://images.unsplash.com/photo-1511379938547-c1f69419868d");

background-size:cover;

background-attachment:fixed;

}


/* Title */

h1 {

text-align:center;

font-size:48px !important;

font-weight:900;

color:white !important;

text-shadow:
0 0 10px #ff00cc,
0 0 25px #00ffff;

}



/* Text */

h2,h3,p,label {

color:white !important;

font-weight:600;

}



/* Input box */

/* Text Input */

/* Text Input */

.stTextInput input {

    background: white !important;

    color: black !important;

    -webkit-text-fill-color: black !important;

    font-size: 18px !important;

    font-weight: normal !important;

    font-family: Arial, sans-serif !important;

    caret-color: black !important;

    border: 2px solid #00FFFF !important;

    border-radius: 20px !important;

}


/* Placeholder */

.stTextInput input::placeholder {

    color: gray !important;

    font-weight: normal !important;

}
/* Dropdown */

.stSelectbox div[data-baseweb="select"] {

background:rgba(255,255,255,0.15);

border-radius:20px;

border:2px solid #ff00cc;

}



/* Button */

.stButton button {

width:250px;

height:55px;

border-radius:30px;

background:

linear-gradient(
45deg,
#ff0080,
#7928ca
);

color:white;

font-size:20px;

font-weight:bold;

border:none;

box-shadow:
0 0 20px #ff0080;

}



.stButton button:hover {

transform:scale(1.05);

box-shadow:
0 0 30px #00ffff;

}



/* Recommendation cards */

div[data-testid="stMarkdownContainer"] {

background:rgba(255,255,255,0.12);

border-radius:20px;

padding:15px;

margin:10px;

border-left:5px solid #00ffff;

}



/* Remove menu */

#MainMenu {

visibility:hidden;

}

footer {

visibility:hidden;

}


</style>

""",
unsafe_allow_html=True
)



# ---------------- LOAD DATA ----------------


@st.cache_data
def load_data():
    hindi = pd.read_csv("Hindi_songs.csv")
    punjabi = pd.read_csv("Punjabi_songs.csv")
    marathi = pd.read_csv("Marathi_songs.csv")
    kannada = pd.read_csv("Kannada_songs.csv")
    gujarati = pd.read_csv("Gujarati_songs.csv")

    hindi["language"]="Hindi"
    punjabi["language"]="Punjabi"
    marathi["language"]="Marathi"
    kannada["language"]="Kannada"
    gujarati["language"]="Gujarati"



    df = pd.concat(
        [
            hindi,
            punjabi,
            marathi,
            kannada,
            gujarati
        ],
        ignore_index=True
    )


    return df



df = load_data()



# ---------------- DATA CHECK ----------------


if "song_name" not in df.columns:

    st.error(
        "song_name column not found"
    )

    st.write(df.columns)

    st.stop()



if "singer" not in df.columns:

    st.error(
        "singer column not found"
    )

    st.write(df.columns)

    st.stop()



# ---------------- CREATE TAGS ----------------


df["tags"] = (

    df["song_name"].astype(str)

    +" "

    +df["singer"].astype(str)

    +" "

    +df["language"].astype(str)

)



# ---------------- NLP MODEL ----------------


@st.cache_resource
def create_similarity(data):


    tfidf = TfidfVectorizer(
        stop_words="english"
    )


    vectors = tfidf.fit_transform(
        data["tags"]
    )


    similarity = cosine_similarity(
        vectors
    )


    return similarity



similarity = create_similarity(df)



# ---------------- TITLE ----------------


st.markdown(
"""
<h1>
🎵 Music Recommendation System 🎧
</h1>

<p style="
text-align:center;
font-size:22px;
">
Find songs according to your language and taste
</p>

""",
unsafe_allow_html=True
)



# ---------------- LANGUAGE SEARCH ----------------


st.subheader("🌐 Search Language")


language_search = st.text_input(
    "Type language"
)



languages = sorted(
    df["language"].unique()
)



if language_search:

    languages = [

        x for x in languages

        if language_search.lower()
        in x.lower()

    ]



language = st.selectbox(
    "Select Language",
    languages
)



# ---------------- SONG SEARCH ----------------


language_df = df[
    df["language"] == language
]



st.subheader("🎧 Search Song")


song_search = st.text_input(
    "Type song name"
)



songs = sorted(
    language_df["song_name"]
    .dropna()
    .unique()
)



if song_search:

    songs = [

        x for x in songs

        if song_search.lower()
        in x.lower()

    ]



if len(songs)>0:

    selected_song = st.selectbox(
        "Select Song",
        songs
    )


else:

    st.warning(
        "Song not found"
    )

    selected_song=None



# ---------------- RECOMMEND FUNCTION ----------------


def recommend(song):


    index = df[
        df["song_name"] == song
    ].index[0]



    scores = list(
        enumerate(
            similarity[index]
        )
    )



    scores = sorted(
        scores,
        key=lambda x:x[1],
        reverse=True
    )



    result=[]


    for i in scores[1:6]:

        result.append(
            df.iloc[i[0]]
        )


    return result



# ---------------- BUTTON ----------------


if st.button("🎶 Recommend Songs"):


    if selected_song:


        result = recommend(
            selected_song
        )


        st.subheader(
            "✨ Recommended Songs"
        )


        for i,song in enumerate(result,1):

            st.markdown(
            f"""
            ### 🎵 {i}. {song['song_name']}

            🎤 **Singer:** {song['singer']}

            🌐 **Language:** {song['language']}

            """
            )


    else:

        st.warning(
            "Please select a song"
        )
