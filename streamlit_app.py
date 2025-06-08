import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import re
from collections import Counter
import string
import ast
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Google Maps Reviews Dashboard", layout="wide")

# --- Load data ---
df = pd.read_parquet("data_lv2.parquet")

# --- Utility functions ---
def parse_relative_date(text):
    if pd.isna(text):
        return None
    if 'day' in text:
        n = int(re.search(r'\d+', text).group()) if re.search(r'\d+', text) else 1
        return pd.Timestamp.now().normalize() - pd.Timedelta(days=n)
    elif 'hour' in text:
        return pd.Timestamp.now().normalize()
    return None

def extract_note(prix):
    if isinstance(prix, str) and '€' in prix:
        euros = re.findall(r'\d+', prix)
        if euros:
            return sum(map(int, euros)) / len(euros)
    return None

def basic_clean_text(text):
    if pd.isna(text):
        return []
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.split()

def parse_dict_safe(x):
    try:
        return ast.literal_eval(x) if isinstance(x, str) and x.strip() != '' else {}
    except:
        return {}

nltk.download('stopwords')
stopwords_en = set(stopwords.words('english'))

def remove_stopwords(text):
    return ' '.join([w for w in text.split() if w.lower() not in stopwords_en])

# --- Initial cleaning ---
df["statut_commentaire"] = np.where(df["textTranslated"].notna(), "Commented", "No comment")
df['parsedDate'] = pd.to_datetime(df['publishedAtDate'], errors='coerce').dt.tz_localize(None)
df['note_estimee'] = df['stars']
df['nb_mots'] = df['text'].astype(str).apply(lambda x: len(x.split()))
df['cleaned_words_all'] = df['textTranslated'].astype(str).apply(remove_stopwords)

df['originalLanguage'] = df['originalLanguage'].fillna('Not specified')
top_langues = df['originalLanguage'].value_counts().nlargest(4).index
df['originalLanguage'] = df['originalLanguage'].where(df['originalLanguage'].isin(top_langues), 'Other')

# --- Sidebar ---
st.sidebar.markdown("""
<h2 style='text-align: center; color: #4B8BBE;'>🚀 Google Review Analysis</h2>
<p style='text-align: center; font-size: 90%; color: #666;'>Filter, explore, understand</p>
<hr style='margin-top: 0.5em; margin-bottom: 1em;'>
""", unsafe_allow_html=True)

# Date range filter
st.sidebar.markdown("### 🗓️ Analysis Period")
min_date = df['parsedDate'].min().date()
max_date = df['parsedDate'].max().date()
date_range = st.sidebar.slider(
    "Select date range",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date),
    format="YYYY-MM-DD"
)

# Rating range filter
st.sidebar.markdown("---")
st.sidebar.markdown("### ⭐ Rating filter")
rating_range = st.sidebar.slider(
    "Select rating range",
    min_value=1,
    max_value=5,
    value=(1, 5)
)

# Language filter
st.sidebar.markdown("---")
st.sidebar.markdown("### 🌄️ Review languages")
langues_disponibles = sorted(df['originalLanguage'].dropna().unique())
langues_choisies = st.sidebar.multiselect(
    "Selected language(s)", options=langues_disponibles, default=[]
)

# Place name filter
st.sidebar.markdown("---")
st.sidebar.markdown("### 🏢 Place name")
noms_disponibles = sorted(df['title'].dropna().unique())
nom_choisi = st.sidebar.selectbox("Filter by place name", options=["All"] + noms_disponibles)

# --- Apply filters ---
df_filtered = df.copy()
df_filtered = df_filtered[
    (df_filtered['parsedDate'].dt.date >= date_range[0]) &
    (df_filtered['parsedDate'].dt.date <= date_range[1])
]
df_filtered = df_filtered[
    df_filtered['note_estimee'].fillna(0).between(rating_range[0], rating_range[1])
]
if langues_choisies:
    df_filtered = df_filtered[df_filtered['originalLanguage'].isin(langues_choisies)]
if nom_choisi != "All":
    df_filtered = df_filtered[df_filtered['title'] == nom_choisi]

# Tabs
tabs = st.tabs(["\U0001F4CA Summary", "\U0001F4C8 Trends", "\U0001F4AC Review Quality", "\U0001F30E Language & Profiles", "\U0001F4DD Explore"])


with tabs[0]:
    st.markdown("## 📊 Key Metrics")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    note_moyenne = df_filtered['note_estimee'].mean()
    pct_guides = df_filtered['isLocalGuide'].mean() * 100
    col1.metric("⭐ Average Estimated Rating", f"{note_moyenne:.2f}" if not pd.isna(note_moyenne) else "N/A")
    col2.metric("📝 Total Reviews", len(df_filtered))
    col3.metric("🎯 % Local Guides", f"{pct_guides:.1f}%" if not pd.isna(pct_guides) else "N/A")

    st.markdown("")

    col4, col5, col6 = st.columns(3)
    dernier_mois = pd.Timestamp.now() - pd.Timedelta(days=30)
    nb_recents = len(df_filtered[df_filtered['parsedDate'] >= dernier_mois])
    col4.metric("📅 Reviews (Last 30 Days)", nb_recents)

    if not df_filtered['originalLanguage'].mode().empty:
        langue = df_filtered['originalLanguage'].mode().iloc[0]
    else:
        langue = "N/A"
    col5.metric("🌄 Main Language", langue)

    if df_filtered['parsedDate'].notna().sum() > 0:
        nb_mois = (df_filtered['parsedDate'].max() - df_filtered['parsedDate'].min()).days / 30
        moyenne_mensuelle = len(df_filtered) / nb_mois if nb_mois > 0 else len(df_filtered)
        col6.metric("📈 Reviews / Month (Estimated)", f"{moyenne_mensuelle:.1f}")
    else:
        col6.metric("📈 Reviews / Month (Estimated)", "N/A")

    st.markdown("---")

     # --- Carte des établissements ---
    st.markdown("## 🗺️ Map of Reviewed Locations")

    # Calcul des stats par établissement
    etablissements = (
        df_filtered
        .groupby(['title', 'location/lat', 'location/lng', 'url'], as_index=False)
        .agg(note_moyenne=('note_estimee', 'mean'),
             nb_avis=('note_estimee', 'count'))
    )

    # Nettoyage des valeurs manquantes
    etablissements = etablissements.dropna(subset=['location/lat', 'location/lng'])

    import folium
    from streamlit_folium import st_folium

    if not etablissements.empty:
        moyenne_lat = etablissements['location/lat'].mean()
        moyenne_lon = etablissements['location/lng'].mean()
        m = folium.Map(location=[moyenne_lat, moyenne_lon], zoom_start=13)

        def get_color(note):
            if note >= 4.5:
                return 'green'
            elif note >= 3.5:
                return 'orange'
            else:
                return 'red'

        for _, row in etablissements.iterrows():
            popup_html = f"""
            <b>{row['title']}</b><br>
            ⭐ Average rating : {row['note_moyenne']:.1f}<br>
            📝 Number of reviews : {row['nb_avis']}<br>
            🔗 <a href="{row['url']}" target="_blank">See on Google Maps</a>
            """
            folium.CircleMarker(
                location=[row['location/lat'], row['location/lng']],
                radius=6,
                color=get_color(row['note_moyenne']),
                fill=True,
                fill_opacity=0.7,
                popup=folium.Popup(popup_html, max_width=300)
            ).add_to(m)

        st_data = st_folium(m, width=900, height=600)
    else:
        st.warning("Aucune coordonnée disponible pour afficher la carte.")

with tabs[1]:
    st.header("📈 Time Analysis")

    if df_filtered['parsedDate'].notna().sum() == 0:
        st.warning("No valid dates found in the data.")
    else:
        df_filtered['month'] = df_filtered['parsedDate'].dt.to_period("M").astype(str)

        st.markdown("### 🎯 Filter Curves")
        col1, col2 = st.columns([2, 3])
        with col1:
            variable_cat = st.selectbox("Categorization variable:", options=[None, 'stars', 'originalLanguage', 'isLocalGuide'])

        st.markdown("---")
        st.markdown("### 📊 Number of Reviews Over Time (Monthly)")

        if variable_cat and variable_cat in df_filtered.columns:
            df_grouped = (
                df_filtered.groupby(['month', variable_cat], observed=True)
                .size()
                .reset_index(name="Number of Reviews")
            )
            fig_line_cat = px.line(
                df_grouped, x='month', y="Number of Reviews", color=variable_cat,
                title="Monthly Evolution of Reviews by Category",
                labels={'month': 'Month', 'Number of Reviews': "Number of Reviews"}
            )
            fig_line_cat.update_layout(legend_title_text=variable_cat, title_x=0.05)
            st.plotly_chart(fig_line_cat, use_container_width=True)
        else:
            df_grouped = (
                df_filtered.groupby('month', observed=True)
                .size()
                .reset_index(name="Number of Reviews")
            )
            fig_line = px.line(
                df_grouped, x='month', y="Number of Reviews",
                title="Monthly Evolution of Reviews",
                labels={'month': 'Month'}
            )
            fig_line.update_layout(title_x=0.05)
            st.plotly_chart(fig_line, use_container_width=True)

        st.markdown("### ⭐ Average Rating by Month")
        moyennes = df_filtered.groupby('month', observed=True)['note_estimee'].mean().reset_index()
        fig_moy = px.line(
            moyennes, x='month', y='note_estimee',
            title="Monthly Evolution of Average Rating",
            labels={'month': 'Month', 'note_estimee': 'Average Rating'}
        )
        fig_moy.update_layout(title_x=0.05)
        st.plotly_chart(fig_moy, use_container_width=True)

        st.markdown("### 🌍 Monthly Number of Reviews by Language")

        lang_monthly = (
        df_filtered.groupby(['month', 'originalLanguage'], observed=True)
        .size()
        .reset_index(name="Number of Reviews")
        )

        fig_lang = px.line(
        lang_monthly, x="month", y="Number of Reviews", color="originalLanguage",
        title="Monthly Reviews by Language",
        labels={"month": "Month", "Number of Reviews": "Number of Reviews"}
        )

        fig_lang.update_layout(
        title_x=0.05,
        legend_title_text="Language",
        legend=dict(
        x=0.01,
        y=0.99,
        xanchor='left',
        yanchor='top',
        bgcolor='rgba(255,255,255,0.6)',
        bordercolor='black',
        borderwidth=1))

        st.plotly_chart(fig_lang, use_container_width=True)


with tabs[2]:
    st.header("💬 Review Quality")

    # Histogramme 1 — distribution des notes estimées
    fig_note = px.histogram(
        df_filtered,
        x='note_estimee',
        color='statut_commentaire',
        nbins=5,
        title="Distribution of Estimated Ratings by Status",
        category_orders={'note_estimee': [1, 2, 3, 4, 5]}
    )
    fig_note.update_layout(
        xaxis=dict(tickmode='linear', tick0=1, dtick=1, title="Google Rating (stars)"),
        yaxis_title="Number of Reviews",
        bargap=0.2
    )
    st.plotly_chart(fig_note, use_container_width=True)

    # Histogramme 2 — longueur des avis avec couleur selon note
    fig_mots = px.histogram(
        df_filtered,
        x='nb_mots',
        color='note_estimee',
        nbins=30,
        title="Review Length (in words)"
    )
    fig_mots.update_layout(bargap=0.2)
    st.plotly_chart(fig_mots, use_container_width=True)

    mots_moyens_par_note_statut = (
        df_filtered.groupby(["note_estimee", "isLocalGuide"], observed=True)["nb_mots"]
        .mean()
        .reset_index(name="Longueur Moyenne")
    )

    fig_mots_moyens = px.bar(
        mots_moyens_par_note_statut,
        x="note_estimee",
        y="Longueur Moyenne",
        color="isLocalGuide",
        barmode="group",
        text="Longueur Moyenne",
        title="Average Review Length by Estimated Rating and Comment Status"
    )
    
    fig_mots_moyens.update_traces(texttemplate='%{text:.1f}', textposition='outside')
    fig_mots_moyens.update_layout(
        xaxis_title="Estimated Rating",
        yaxis_title="Average Number of Words",
        yaxis=dict(tickformat="d"),
        bargap=0.2
    )
    st.plotly_chart(fig_mots_moyens, use_container_width=True)

    # Statistiques textuelles
    st.markdown(f"**% of very short reviews (<10 words)**: {100 * (df_filtered['nb_mots'] < 10).mean():.1f}%")
    st.markdown(f"**% of very long reviews (>100 words)**: {100 * (df_filtered['nb_mots'] > 100).mean():.1f}%")

    st.markdown("---")
    st.markdown("### ☁️ WordCloud by Review Tone")

    mots_exclus = {"none", "louis", "vuitton", "store", "Paris", "came", "come", " Store"}

    text_pos = ' '.join(
        word for line in df_filtered[df_filtered['stars'] >= 4]['cleaned_words_all'].dropna()
        for word in line.split()
        if len(word) > 3 and word.lower() not in mots_exclus
    )

    text_neg = ' '.join(
        word for line in df_filtered[df_filtered['stars'] < 4]['cleaned_words_all'].dropna()
        for word in line.split()
        if len(word) > 3 and word.lower() not in mots_exclus
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🟢 Words in **Positive Reviews** (⭐ ≥ 4)")
        wc_pos = WordCloud(width=400, height=300, background_color='white').generate(text_pos)
        fig, ax = plt.subplots()
        ax.imshow(wc_pos, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)

    with col2:
        st.markdown("#### 🔴 Words in **Negative Reviews** (⭐ < 4)")
        wc_neg = WordCloud(width=400, height=300, background_color='white', colormap='Reds').generate(text_neg)
        fig, ax = plt.subplots()
        ax.imshow(wc_neg, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)
        
        
with tabs[3]:
    st.header("🌍 Languages & Profiles")

    col1, col2 = st.columns(2)

    with col1:
        fig_lang = px.pie(df_filtered, names='originalLanguage', title="Language Distribution")
        st.plotly_chart(fig_lang, use_container_width=True)

    with col2:
        fig_guide = px.pie(df_filtered, names='isLocalGuide', title="Proportion of Local Guides")
        st.plotly_chart(fig_guide, use_container_width=True)

    df_filtered['stars_arrondi'] = df_filtered['stars'].round(0).astype("Int64")

    fig_stars_lang = px.histogram(
        df_filtered,
        x='stars_arrondi',
        color='originalLanguage',
        barmode='stack',
        title="Rating Distribution by Language",
        labels={"stars_arrondi": "Rounded Rating", "count": "Number of Reviews"},
        category_orders={"stars_arrondi": [1, 2, 3, 4, 5]},
        height=500
    )
    fig_stars_lang.update_layout(
        bargap=0.2,
        xaxis_title="Stars",
        yaxis_title="Number of Reviews",
        legend_title="Language"
    )
    st.plotly_chart(fig_stars_lang, use_container_width=True)

    moyennes_par_langue = (
        df_filtered.groupby('originalLanguage')['stars']
        .mean()
        .reset_index()
        .sort_values(by='stars', ascending=False)
    )

    moyennes_par_langue['stars'] = moyennes_par_langue['stars'].round(1)
    moyennes_par_langue['langue_ordonnee'] = pd.Categorical(
        moyennes_par_langue['originalLanguage'],
        categories=moyennes_par_langue['originalLanguage'],
        ordered=True
    )

    fig_bar = px.bar(
        moyennes_par_langue,
        x='langue_ordonnee',
        y='stars',
        color='originalLanguage',
        text='stars',
        text_auto=True,
        title="Average Rating by Language (Descending Order)",
        labels={"stars": "Average Rating", "langue_ordonnee": "Language"}
    )
    st.plotly_chart(fig_bar, use_container_width=True)


with tabs[4]:
    st.header("📝 Explore Reviews")

    # Sélecteur langue et minimum mots
    langue_sel = st.selectbox(
        "Filter by Language",
        options=["All"] + sorted(df['originalLanguage'].dropna().unique().tolist()),
        key="langue_sel_explore"
    )

    min_mots = st.slider(
        "Minimum Review Length (in words)", 
        min_value=0, max_value=500, 
        value=20,
        key="min_mots_explore"
    )

    # Champ de recherche — HORS colonnes
    mot_cle = st.text_input(
        "Search for a word in comments (case-insensitive)",
        key="mot_cle_input_explore"
    )

    # Copie DataFrame
    df_exploration = df_filtered.copy()

    # Filtres
    if langue_sel != "All":
        df_exploration = df_exploration[df_exploration['originalLanguage'] == langue_sel]

    df_exploration = df_exploration[df_exploration['nb_mots'] >= min_mots]

    if mot_cle:
        df_exploration = df_exploration[df_exploration['text'].str.contains(mot_cle, case=False, na=False)]

    st.markdown(f"### {len(df_exploration)} review(s) found")
    st.markdown("---")

    for _, row in df_exploration.head(30).iterrows():
        st.markdown(f"**{row['name']}** ({'Local Guide' if row['isLocalGuide'] else 'Visitor'})")
        st.markdown(f"⭐ {row['note_estimee']} | 🕒 {row['publishAt']}")
        st.markdown(f"💬 {row['text']}")
        st.markdown("---")


