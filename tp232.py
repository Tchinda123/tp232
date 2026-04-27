import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Configuration
st.set_page_config(page_title="INF232 - Analyse de données", layout="centered")

st.title("📊 Application de collecte et d'analyse des données")
st.write("Cette application permet de collecter, analyser et visualiser des données.")

# Initialisation des données
if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=["Nom", "Age", "Score"])

# ------------------------
# FORMULAIRE
# ------------------------
st.header("📝 Collecte des données")

with st.form("formulaire"):
    nom = st.text_input("Nom")
    age = st.number_input("Age", min_value=0, max_value=120, step=1)
    score = st.number_input("Score", min_value=0.0, max_value=100.0, step=0.1)

    submit = st.form_submit_button("Ajouter")

    if submit:
        if nom == "":
            st.warning("Veuillez entrer un nom")
        else:
            new_row = pd.DataFrame([[nom, age, score]], columns=["Nom", "Age", "Score"])
            st.session_state.data = pd.concat([st.session_state.data, new_row], ignore_index=True)
            st.success("✅ Donnée ajoutée avec succès")

# ------------------------
# AFFICHAGE DES DONNÉES
# ------------------------
st.header("📋 Données collectées")

df = st.session_state.data

if len(df) > 0:
    st.dataframe(df)

    # ------------------------
    # ANALYSE DESCRIPTIVE
    # ------------------------
    st.header("📈 Analyse descriptive")

    st.write("🔢 Nombre total :", len(df))
    st.write("📊 Age moyen :", round(df["Age"].mean(), 2))
    st.write("📊 Score moyen :", round(df["Score"].mean(), 2))
    st.write("⬆️ Age max :", df["Age"].max())
    st.write("⬆️ Score max :", df["Score"].max())
    st.write("⬇️ Age min :", df["Age"].min())
    st.write("⬇️ Score min :", df["Score"].min())

    # ------------------------
    # VISUALISATION
    # ------------------------
    st.header("📊 Visualisation")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribution des âges")
        fig1, ax1 = plt.subplots()
        ax1.hist(df["Age"])
        ax1.set_xlabel("Age")
        ax1.set_ylabel("Fréquence")
        st.pyplot(fig1)

    with col2:
        st.subheader("Distribution des scores")
        fig2, ax2 = plt.subplots()
        ax2.hist(df["Score"])
        ax2.set_xlabel("Score")
        ax2.set_ylabel("Fréquence")
        st.pyplot(fig2)

    # ------------------------
    # RÉGRESSION LINÉAIRE
    # ------------------------
    st.header("📉 Régression linéaire simple (Age → Score)")

    if len(df) >= 2:
        X = df[["Age"]].values
        y = df["Score"].values

        model = LinearRegression()
        model.fit(X, y)

        y_pred = model.predict(X)

        st.write("📌 Equation : Score = a * Age + b")
        st.write(f"a = {round(model.coef_[0], 2)}")
        st.write(f"b = {round(model.intercept_, 2)}")

        # Graphique
        fig3, ax3 = plt.subplots()
        ax3.scatter(X, y)
        ax3.plot(X, y_pred)
        ax3.set_xlabel("Age")
        ax3.set_ylabel("Score")
        st.pyplot(fig3)

    else:
        st.info("Ajoutez au moins 2 données pour la régression")

    # ------------------------
    # TÉLÉCHARGEMENT
    # ------------------------
    st.header("💾 Export des données")

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Télécharger en CSV", csv, "donnees.csv", "text/csv")

else:
    st.info("Aucune donnée disponible pour le moment")

# ------------------------
# FOOTER
# ------------------------
st.write("---")
st.write("👨‍💻 Projet INF232 - Analyse de données")