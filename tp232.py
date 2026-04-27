import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="MediData INF232 EC2", page_icon="hospital", layout="wide")

# ── SECTEUR : SANTE ──────────────────────────────────────────
# Application axee sur les donnees de sante des patients

MALADIES = {'Diabete':1,'Hypertension':2,'Paludisme':3,'Tuberculose':4,'Anemie':5,'Autre':6}
HOPITAUX = ['CHU Yaounde','Hopital Central','Laquintinie Douala','Hopital de District','Clinique Privee']
COULEURS = ['#e74c3c','#3498db','#2ecc71','#f39c12','#9b59b6','#1abc9c']

if 'db' not in st.session_state:
    st.session_state.db = pd.DataFrame(columns=[
        'Patient','Age','Sexe','Hopital','Maladie','Maladie_num',
        'Duree_Sejour','Cout_Traitement','Satisfaction','Ville'
    ])

st.sidebar.title("MediData - Sante")
st.sidebar.markdown("**INF 232 EC2** - Analyse de donnees medicales")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigation", [
    "Accueil",
    "Collecte de donnees",
    "Base de donnees",
    "1 - Regression Lineaire Simple",
    "2 - Regression Lineaire Multiple",
    "3 - Reduction PCA",
    "4 - Classification Supervisee",
    "5 - Clustering KMeans"
])

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Patients enregistres :** {len(st.session_state.db)}")
if len(st.session_state.db) >= 3:
    st.sidebar.success("Donnees suffisantes")
else:
    st.sidebar.warning(f"Minimum 3 patients requis ({len(st.session_state.db)}/3)")

if st.sidebar.button("Charger donnees demo"):
    noms = ['Patient_001','Patient_002','Patient_003','Patient_004','Patient_005',
            'Patient_006','Patient_007','Patient_008','Patient_009','Patient_010',
            'Patient_011','Patient_012','Patient_013','Patient_014','Patient_015',
            'Patient_016','Patient_017','Patient_018','Patient_019','Patient_020']
    mal_list = list(MALADIES.keys())
    rows = []
    np.random.seed(10)
    for i, nom in enumerate(noms):
        duree = np.random.randint(1, 30)
        mal_nom = mal_list[np.random.randint(0, 6)]
        mal_num = MALADIES[mal_nom]
        rows.append({
            'Patient': nom,
            'Age': np.random.randint(18, 75),
            'Sexe': 'M' if i % 2 == 0 else 'F',
            'Hopital': HOPITAUX[i % len(HOPITAUX)],
            'Maladie': mal_nom,
            'Maladie_num': mal_num,
            'Duree_Sejour': duree,
            'Cout_Traitement': int((20000 + duree*15000 + mal_num*10000 + np.random.randint(-10000,10000))/1000)*1000,
            'Satisfaction': np.random.randint(3, 11),
            'Ville': ['Douala','Yaounde','Bafoussam','Garoua','Kribi'][i%5]
        })
    st.session_state.db = pd.DataFrame(rows)
    st.sidebar.success("20 patients demo charges!")
    st.rerun()


def check_data(min_rows=5):
    if len(st.session_state.db) < min_rows:
        st.warning(f"Minimum {min_rows} patients requis. Vous avez {len(st.session_state.db)}.")
        st.info("Allez dans Collecte de donnees ou chargez les donnees demo.")
        return False
    return True


if page == "Accueil":
    st.title("MediData - INF 232 EC2")
    st.markdown("### Application de Collecte et Analyse des Donnees de Sante")
    st.markdown("---")
    df = st.session_state.db
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Patients enregistres", len(df))
    col2.metric("Age moyen", f"{df['Age'].mean():.0f} ans" if len(df) else "---")
    col3.metric("Cout moyen", f"{df['Cout_Traitement'].mean()/1000:.0f}k FCFA" if len(df) else "---")
    col4.metric("Satisfaction", f"{df['Satisfaction'].mean():.1f}/10" if len(df) else "---")
    st.markdown("---")
    st.markdown("## Programme EC2 - Secteur Sante")
    col1, col2 = st.columns(2)
    with col1:
        st.info("**Collecte** : Donnees patients (age, maladie, duree sejour, cout, satisfaction)")
        st.success("**1 - Regression Simple** : Duree sejour -> Cout traitement")
        st.success("**2 - Regression Multiple** : Age + Duree + Maladie -> Cout")
    with col2:
        st.warning("**3 - PCA** : Compression des variables medicales")
        st.warning("**4 - KNN** : Predire le type de maladie")
        st.error("**5 - KMeans** : Segmenter les patients par profil")
    st.markdown("---")
    st.markdown("Utilisez le menu a gauche. Cliquez **Charger donnees demo** pour commencer.")


elif page == "Collecte de donnees":
    st.title("Formulaire Patient - Sante")
    st.markdown("---")
    with st.form("form_sante", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            patient = st.text_input("Identifiant patient", placeholder="Ex : Patient_021")
            age = st.number_input("Age du patient", min_value=0, max_value=110, value=35)
            sexe = st.selectbox("Sexe", ["Masculin","Feminin"])
            hopital = st.selectbox("Hopital / Structure", HOPITAUX)
        with col2:
            maladie = st.selectbox("Maladie diagnostiquee", list(MALADIES.keys()))
            duree = st.number_input("Duree du sejour (jours)", min_value=1, max_value=365, value=5)
            cout = st.number_input("Cout du traitement (FCFA)", min_value=0, max_value=5000000, value=50000, step=5000)
            ville = st.text_input("Ville", placeholder="Ex : Yaounde")
        satisfaction = st.slider("Satisfaction du patient (1 a 10)", 1, 10, 7)
        submitted = st.form_submit_button("Enregistrer le patient", use_container_width=True)
        if submitted:
            if not patient.strip():
                st.error("Veuillez saisir un identifiant patient.")
            else:
                new_row = pd.DataFrame([{
                    'Patient': patient.strip(), 'Age': age, 'Sexe': sexe,
                    'Hopital': hopital, 'Maladie': maladie, 'Maladie_num': MALADIES[maladie],
                    'Duree_Sejour': duree, 'Cout_Traitement': cout,
                    'Satisfaction': satisfaction, 'Ville': ville
                }])
                st.session_state.db = pd.concat([st.session_state.db, new_row], ignore_index=True)
                st.success(f"Patient {patient} enregistre ! Total : {len(st.session_state.db)}")


elif page == "Base de donnees":
    st.title("Base de Donnees - Patients")
    df = st.session_state.db
    col1, col2, col3 = st.columns(3)
    col1.metric("Total patients", len(df))
    with col2:
        if len(df):
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Telecharger CSV", csv, "medidata_export.csv", "text/csv")
    with col3:
        if st.button("Vider les donnees"):
            st.session_state.db = pd.DataFrame(columns=df.columns)
            st.rerun()
    st.markdown("---")
    if df.empty:
        st.warning("Aucun patient. Allez dans Collecte de donnees.")
    else:
        st.dataframe(df, use_container_width=True)
        st.markdown("### Statistiques descriptives")
        st.dataframe(df[['Age','Duree_Sejour','Cout_Traitement','Satisfaction']].describe().round(2), use_container_width=True)
        st.markdown("### Distributions")
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes[0,0].hist(df['Age'], bins=10, color='#e74c3c', edgecolor='white')
        axes[0,0].set_title('Distribution des Ages')
        axes[0,1].hist(df['Cout_Traitement'], bins=10, color='#3498db', edgecolor='white')
        axes[0,1].set_title('Cout du Traitement')
        mc = df['Maladie'].value_counts()
        axes[1,0].bar(mc.index, mc.values, color=COULEURS[:len(mc)], edgecolor='white')
        axes[1,0].set_title('Repartition par Maladie')
        axes[1,0].tick_params(axis='x', rotation=30)
        hc = df['Hopital'].value_counts()
        axes[1,1].pie(hc.values, labels=hc.index, colors=COULEURS[:len(hc)], autopct='%1.0f%%')
        axes[1,1].set_title('Repartition par Hopital')
        plt.tight_layout()
        st.pyplot(fig); plt.close()


elif page == "1 - Regression Lineaire Simple":
    st.title("1 - Regression Lineaire Simple")
    st.markdown("**Objectif :** Relation entre **Duree du sejour (X)** et **Cout du traitement (Y)**.")
    st.markdown("**Equation :** Cout = a x Duree + b")
    st.markdown("---")
    if not check_data(3): st.stop()
    df = st.session_state.db.copy()
    X = df[['Duree_Sejour']].values
    y = df['Cout_Traitement'].values
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    a = model.coef_[0]; b = model.intercept_
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R2", f"{r2:.4f}")
    col2.metric("R Pearson", f"{r2**0.5:.4f}")
    col3.metric("RMSE", f"{rmse/1000:.1f}k FCFA")
    col4.metric("Patients", len(df))
    st.markdown("---")
    st.info(f"Cout = {a:.0f} x Duree_Sejour + {b:.0f}")
    st.markdown(f"- Chaque jour supplementaire coute en moyenne **+{a:.0f} FCFA**")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(X, y, color='#e74c3c', edgecolors='black', s=70, label='Patients')
        x_line = np.linspace(X.min(), X.max(), 100).reshape(-1,1)
        ax.plot(x_line, model.predict(x_line), color='blue', linewidth=2.5, label=f'Droite')
        ax.set_xlabel("Duree Sejour (jours)")
        ax.set_ylabel("Cout Traitement (FCFA)")
        ax.set_title(f"Regression Simple (R2={r2:.3f})")
        ax.legend(); ax.grid(True, alpha=0.3)
        st.pyplot(fig); plt.close()
    with col2:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(y, y_pred, color='#3498db', edgecolors='black', s=70)
        lims = [y.min(), y.max()]
        ax.plot(lims, lims, 'r--', linewidth=1.5, label='Parfait')
        ax.set_xlabel("Reel (FCFA)"); ax.set_ylabel("Predit (FCFA)")
        ax.set_title("Reel vs Predit"); ax.legend(); ax.grid(True, alpha=0.3)
        st.pyplot(fig); plt.close()
    st.markdown("---")
    duree_val = st.slider("Duree du sejour (jours)", 1, 60, 10)
    pred = model.predict([[duree_val]])[0]
    st.success(f"Pour {duree_val} jours de sejour -> Cout predit : **{pred:,.0f} FCFA**")


elif page == "2 - Regression Lineaire Multiple":
    st.title("2 - Regression Lineaire Multiple")
    st.markdown("**Objectif :** Predire le **Cout** avec Age + Duree + Maladie + Satisfaction.")
    st.markdown("---")
    if not check_data(5): st.stop()
    df = st.session_state.db.copy()
    features = ['Age','Duree_Sejour','Maladie_num','Satisfaction']
    labels_f = ['Age','Duree Sejour','Type Maladie','Satisfaction']
    X = df[features].values; y = df['Cout_Traitement'].values
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    model = LinearRegression()
    model.fit(X_sc, y)
    y_pred = model.predict(X_sc)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R2", f"{r2:.4f}"); col2.metric("R multiple", f"{max(0,r2)**0.5:.4f}")
    col3.metric("RMSE", f"{rmse/1000:.1f}k FCFA"); col4.metric("Variables", 4)
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        coef_df = pd.DataFrame({'Variable': labels_f, 'Coefficient': [f"{c:+.0f}" for c in model.coef_],
                                 'Effet': ['Positif' if c>0 else 'Negatif' for c in model.coef_]})
        st.dataframe(coef_df, use_container_width=True, hide_index=True)
        corr = df[features].rename(columns=dict(zip(features,labels_f))).corr()
        fig, ax = plt.subplots(figsize=(5,4))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax)
        ax.set_title('Correlations'); st.pyplot(fig); plt.close()
    with col2:
        fig, ax = plt.subplots(figsize=(6,4))
        colors = ['#e74c3c' if c>0 else '#3498db' for c in model.coef_]
        ax.barh(labels_f, model.coef_, color=colors, edgecolor='white')
        ax.set_title("Impact sur le Cout"); ax.axvline(0,color='black',linewidth=0.8)
        ax.grid(True,axis='x',alpha=0.3); st.pyplot(fig); plt.close()
        fig, ax = plt.subplots(figsize=(6,4))
        ax.scatter(y, y_pred, color='#2ecc71', edgecolors='black', s=60)
        lims = [y.min(), y.max()]
        ax.plot(lims, lims, 'r--', linewidth=1.5)
        ax.set_xlabel("Reel"); ax.set_ylabel("Predit"); ax.set_title(f"R2={r2:.3f}")
        ax.grid(True, alpha=0.3); st.pyplot(fig); plt.close()


elif page == "3 -  Technique de Reduction des dimentionalites des donnees":
    st.title("3 - Reduction de Dimensionnalite ")
    st.markdown("**Objectif :** Compresser les variables medicales en 2D.")
    st.markdown("---")
    if not check_data(5): st.stop()
    df = st.session_state.db.copy()
    features = ['Age','Duree_Sejour','Maladie_num','Cout_Traitement','Satisfaction']
    labels_f = ['Age','Duree','Maladie','Cout','Satisfaction']
    X = df[features].values
    scaler = StandardScaler(); X_sc = scaler.fit_transform(X)
    pca_full = PCA(); pca_full.fit(X_sc)
    var_exp = pca_full.explained_variance_ratio_; var_cum = np.cumsum(var_exp)
    pca2 = PCA(n_components=2); X_pca = pca2.fit_transform(X_sc)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Variance PC1", f"{var_exp[0]:.1%}"); col2.metric("Variance PC2", f"{var_exp[1]:.1%}")
    col3.metric("Cumul PC1+PC2", f"{(var_exp[0]+var_exp[1]):.1%}"); col4.metric("Variables", 5)
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6,4))
        ax.bar(range(1,len(var_exp)+1), var_exp*100, color='#e74c3c', edgecolor='white', label='Par composante')
        ax.plot(range(1,len(var_exp)+1), var_cum*100, 'o-', color='#3498db', linewidth=2, label='Cumule')
        ax.axhline(y=95, color='green', linestyle='--', label='95%')
        ax.set_xlabel("Composante"); ax.set_ylabel("Variance (%)"); ax.set_title("Scree Plot")
        ax.legend(); ax.grid(True, alpha=0.3); st.pyplot(fig); plt.close()
        st.dataframe(pd.DataFrame(pca2.components_.T, index=labels_f, columns=['PC1','PC2']).round(3), use_container_width=True)
    with col2:
        fig, ax = plt.subplots(figsize=(6,5))
        maladies = df['Maladie'].unique()
        for i, mal in enumerate(maladies):
            mask = df['Maladie'] == mal
            ax.scatter(X_pca[mask,0], X_pca[mask,1], color=COULEURS[i%len(COULEURS)],
                       label=mal, edgecolors='black', s=70, alpha=0.9)
        ax.set_xlabel(f"PC1 ({var_exp[0]:.1%})"); ax.set_ylabel(f"PC2 ({var_exp[1]:.1%})")
        ax.set_title("PCA 2D (par Maladie)"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
        st.pyplot(fig); plt.close()


elif page == "4 - Technique de Classification Supervisee":
    st.title("4 - Classification Supervisee ")
    st.markdown("**Objectif :** Predire le **type de maladie** a partir de Age, Cout, Duree, Satisfaction.")
    st.markdown("---")
    if not check_data(6): st.stop()
    df = st.session_state.db.copy()
    features = ['Age','Duree_Sejour','Cout_Traitement','Satisfaction']
    X = df[features].values; y = df['Maladie_num'].values
    scaler = StandardScaler(); X_sc = scaler.fit_transform(X)
    k = min(5, len(df)-1)
    model_knn = KNeighborsClassifier(n_neighbors=k)
    model_knn.fit(X_sc, y); y_pred = model_knn.predict(X_sc)
    acc = accuracy_score(y, y_pred)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{acc:.1%}"); col2.metric("K voisins", k)
    col3.metric("Maladies", len(np.unique(y))); col4.metric("Variables", 4)
    st.markdown("---")
    inv_mal = {v: k for k, v in MALADIES.items()}
    col1, col2 = st.columns(2)
    with col1:
        uniq = np.unique(y)
        mal_names = [inv_mal.get(n, str(n)) for n in uniq]
        prec = [np.mean(y_pred[np.where(y==niv)[0]]==niv) if len(np.where(y==niv)[0]) else 0 for niv in uniq]
        fig, ax = plt.subplots(figsize=(6,4))
        ax.bar(mal_names, [p*100 for p in prec], color=COULEURS[:len(uniq)], edgecolor='white')
        ax.set_ylabel("Precision (%)"); ax.set_title("Precision par Maladie")
        ax.tick_params(axis='x', rotation=20); ax.set_ylim(0,115); ax.grid(True, axis='y', alpha=0.3)
        st.pyplot(fig); plt.close()
    with col2:
        cm = confusion_matrix(y, y_pred)
        fig, ax = plt.subplots(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=ax, xticklabels=mal_names, yticklabels=mal_names)
        ax.set_xlabel("Predit"); ax.set_ylabel("Reel"); ax.set_title("Matrice de Confusion")
        ax.tick_params(axis='x', rotation=20); st.pyplot(fig); plt.close()
        st.markdown(f"### Accuracy : **{acc:.1%}**"); st.progress(acc)
    st.markdown("---")
    st.markdown("### Predire la maladie d'un nouveau patient")
    c1,c2,c3,c4 = st.columns(4)
    pa=c1.number_input("Age",0,110,40,key='k_age')
    pd_=c2.number_input("Duree sejour",1,365,7,key='k_dur')
    pc=c3.number_input("Cout (FCFA)",0,2000000,80000,step=5000,key='k_cout')
    ps=c4.slider("Satisfaction",1,10,6,key='k_sat')
    if st.button("Predire la maladie"):
        inp = scaler.transform([[pa,pd_,pc,ps]])
        pred = model_knn.predict(inp)[0]
        st.success(f"Maladie predite : **{inv_mal.get(pred,pred)}**")


elif page == "5 - Technique de classification non-supervisee":
    st.title("5 - Classification Non-Supervisee ")
    st.markdown("**Objectif :** Segmenter les patients en groupes homogenes selon Cout et Duree.")
    st.markdown("---")
    if not check_data(5): st.stop()
    df = st.session_state.db.copy()
    K_max = min(6, len(df)-1)
    K = st.slider("Nombre de clusters (K)", 2, K_max, min(3, K_max))
    X = df[['Cout_Traitement','Duree_Sejour']].values
    scaler = StandardScaler(); X_sc = scaler.fit_transform(X)
    km = KMeans(n_clusters=K, random_state=42, n_init=10)
    df['Cluster'] = km.fit_predict(X_sc)
    col1,col2,col3,col4 = st.columns(4)
    col1.metric("Clusters",K); col2.metric("Inertie",f"{km.inertia_:.1f}")
    col3.metric("Patients",len(df)); col4.metric("Variables",2)
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6,5))
        for c in range(K):
            mask = df['Cluster']==c
            ax.scatter(df.loc[mask,'Cout_Traitement'], df.loc[mask,'Duree_Sejour'],
                       color=COULEURS[c%len(COULEURS)], label=f'Groupe {c}', edgecolors='black', s=80)
        centers_orig = scaler.inverse_transform(km.cluster_centers_)
        ax.scatter(centers_orig[:,0], centers_orig[:,1], marker='X', s=200, c='black', zorder=5, label='Centres')
        ax.set_xlabel("Cout Traitement (FCFA)"); ax.set_ylabel("Duree Sejour (jours)")
        ax.set_title(f"K-Means Patients (K={K})"); ax.legend(fontsize=8); ax.grid(True,alpha=0.3)
        st.pyplot(fig); plt.close()
        profile = df.groupby('Cluster')[['Age','Duree_Sejour','Cout_Traitement','Satisfaction']].mean().round(1)
        profile.index = [f"Groupe {i}" for i in profile.index]
        st.dataframe(profile, use_container_width=True)
    with col2:
        ks = range(2, K_max+1)
        inertias = [KMeans(n_clusters=k,random_state=42,n_init=10).fit(X_sc).inertia_ for k in ks]
        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(list(ks), inertias, 'o-', color='#e74c3c', linewidth=2.5, markersize=8)
        ax.axvline(x=K, color='blue', linestyle='--', label=f'K={K}')
        ax.set_xlabel("K"); ax.set_ylabel("Inertie"); ax.set_title("Methode du Coude")
        ax.legend(); ax.grid(True,alpha=0.3); st.pyplot(fig); plt.close()
        sizes = df['Cluster'].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(6,4))
        ax.bar([f'Groupe {i}' for i in sizes.index], sizes.values,
               color=[COULEURS[i%len(COULEURS)] for i in sizes.index], edgecolor='white')
        ax.set_ylabel("Nombre de patients"); ax.set_title("Effectif par Groupe")
        ax.grid(True,axis='y',alpha=0.3); st.pyplot(fig); plt.close()
    st.markdown("---")
    col1, col2 = st.columns(2)
    col1.info("**Supervise (KNN)** : On connait la maladie -> On predit pour un nouveau patient")
    col2.warning("**Non-Supervise (KMeans)** : On ne connait pas les groupes -> L'algorithme les trouve seul")
