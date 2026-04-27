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

st.set_page_config(page_title="AgroData INF232 EC2", page_icon="seedling", layout="wide")

# ── SECTEUR : AGRICULTURE ─────────────────────────────────────
# Application axee sur les donnees agricoles des exploitants

CULTURES = {'Mais':1,'Cacao':2,'Cafe':3,'Manioc':4,'Plantain':5,'Arachide':6}
REGIONS = ['Centre','Littoral','Ouest','Sud','Nord','Est']
COULEURS = ['#27ae60','#f39c12','#8b4513','#e74c3c','#f1c40f','#16a085']

if 'db' not in st.session_state:
    st.session_state.db = pd.DataFrame(columns=[
        'Exploitant','Age','Sexe','Region','Culture','Culture_num',
        'Superficie','Production','Satisfaction','Ville'
    ])

st.sidebar.title("AgroData - Agriculture")
st.sidebar.markdown("**INF 232 EC2** - Analyse de donnees agricoles")
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
st.sidebar.markdown(f"**Exploitants enregistres :** {len(st.session_state.db)}")
if len(st.session_state.db) >= 3:
    st.sidebar.success("Donnees suffisantes")
else:
    st.sidebar.warning(f"Minimum 3 enregistrements requis ({len(st.session_state.db)}/3)")

if st.sidebar.button("Charger donnees demo"):
    noms = ['Nkoa Jean','Fon Marie','Bello Ali','Manga Rose','Tabi Pierre',
            'Ekedi Claire','Zanga Paul','Moto Lucie','Ndi Thomas','Baka Judith',
            'Kana Eric','Toko Sara','Mbida Victor','Neba Alice','Foko Gaetan',
            'Simo Rose','Abega Claude','Talla Herve','Kotto Estelle','Bissa Franck']
    cult_list = list(CULTURES.keys())
    rows = []
    np.random.seed(5)
    for i, nom in enumerate(noms):
        superficie = np.random.randint(1, 20)
        cult_nom = cult_list[np.random.randint(0, 6)]
        cult_num = CULTURES[cult_nom]
        rows.append({
            'Exploitant': nom,
            'Age': np.random.randint(20, 65),
            'Sexe': 'M' if i % 2 == 0 else 'F',
            'Region': REGIONS[i % len(REGIONS)],
            'Culture': cult_nom,
            'Culture_num': cult_num,
            'Superficie': superficie,
            'Production': int((500 + superficie*300 + cult_num*100 + np.random.randint(-200,200))),
            'Satisfaction': np.random.randint(3, 11),
            'Ville': ['Yaounde','Douala','Bafoussam','Ebolowa','Bertoua'][i%5]
        })
    st.session_state.db = pd.DataFrame(rows)
    st.sidebar.success("20 exploitants demo charges!")
    st.rerun()


def check_data(min_rows=5):
    if len(st.session_state.db) < min_rows:
        st.warning(f"Minimum {min_rows} enregistrements requis. Vous avez {len(st.session_state.db)}.")
        st.info("Allez dans Collecte de donnees ou chargez les donnees demo.")
        return False
    return True


if page == "Accueil":
    st.title("AgroData - INF 232 EC2")
    st.markdown("### Application de Collecte et Analyse des Donnees Agricoles")
    st.markdown("---")
    df = st.session_state.db
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Exploitants enregistres", len(df))
    col2.metric("Age moyen", f"{df['Age'].mean():.0f} ans" if len(df) else "---")
    col3.metric("Production moyenne", f"{df['Production'].mean():.0f} kg" if len(df) else "---")
    col4.metric("Satisfaction", f"{df['Satisfaction'].mean():.1f}/10" if len(df) else "---")
    st.markdown("---")
    st.markdown("## Programme EC2 - Secteur Agriculture")
    col1, col2 = st.columns(2)
    with col1:
        st.info("**Collecte** : Donnees exploitants (age, culture, superficie, production, satisfaction)")
        st.success("**1 - Regression Simple** : Superficie -> Production agricole")
        st.success("**2 - Regression Multiple** : Age + Superficie + Culture -> Production")
    with col2:
        st.warning("**3 - PCA** : Compression des variables agricoles")
        st.warning("**4 - KNN** : Predire le type de culture")
        st.error("**5 - KMeans** : Segmenter les exploitants par profil")
    st.markdown("---")
    st.markdown("Utilisez le menu a gauche. Cliquez **Charger donnees demo** pour commencer.")


elif page == "Collecte de donnees":
    st.title("Formulaire Exploitant Agricole")
    st.markdown("---")
    with st.form("form_agro", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            exploitant = st.text_input("Nom de l'exploitant", placeholder="Ex : Nkoa Jean")
            age = st.number_input("Age", min_value=15, max_value=80, value=35)
            sexe = st.selectbox("Sexe", ["Masculin","Feminin"])
            region = st.selectbox("Region", REGIONS)
        with col2:
            culture = st.selectbox("Type de culture", list(CULTURES.keys()))
            superficie = st.number_input("Superficie (hectares)", min_value=0.5, max_value=100.0, value=2.0, step=0.5)
            production = st.number_input("Production (kg)", min_value=0, max_value=100000, value=1000, step=100)
            ville = st.text_input("Ville / Village", placeholder="Ex : Bafia")
        satisfaction = st.slider("Satisfaction de l'exploitant (1 a 10)", 1, 10, 6)
        submitted = st.form_submit_button("Enregistrer l'exploitant", use_container_width=True)
        if submitted:
            if not exploitant.strip():
                st.error("Veuillez saisir un nom.")
            else:
                new_row = pd.DataFrame([{
                    'Exploitant': exploitant.strip(), 'Age': age, 'Sexe': sexe,
                    'Region': region, 'Culture': culture, 'Culture_num': CULTURES[culture],
                    'Superficie': superficie, 'Production': production,
                    'Satisfaction': satisfaction, 'Ville': ville
                }])
                st.session_state.db = pd.concat([st.session_state.db, new_row], ignore_index=True)
                st.success(f"Exploitant {exploitant} enregistre ! Total : {len(st.session_state.db)}")


elif page == "Base de donnees":
    st.title("Base de Donnees - Exploitants Agricoles")
    df = st.session_state.db
    col1, col2, col3 = st.columns(3)
    col1.metric("Total exploitants", len(df))
    with col2:
        if len(df):
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Telecharger CSV", csv, "agrodata_export.csv", "text/csv")
    with col3:
        if st.button("Vider les donnees"):
            st.session_state.db = pd.DataFrame(columns=df.columns)
            st.rerun()
    st.markdown("---")
    if df.empty:
        st.warning("Aucun exploitant. Allez dans Collecte de donnees.")
    else:
        st.dataframe(df, use_container_width=True)
        st.markdown("### Statistiques descriptives")
        st.dataframe(df[['Age','Superficie','Production','Satisfaction']].describe().round(2), use_container_width=True)
        st.markdown("### Distributions")
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes[0,0].hist(df['Age'], bins=10, color='#27ae60', edgecolor='white')
        axes[0,0].set_title('Distribution des Ages')
        axes[0,1].hist(df['Production'], bins=10, color='#f39c12', edgecolor='white')
        axes[0,1].set_title('Production (kg)')
        cc = df['Culture'].value_counts()
        axes[1,0].bar(cc.index, cc.values, color=COULEURS[:len(cc)], edgecolor='white')
        axes[1,0].set_title('Repartition par Culture')
        axes[1,0].tick_params(axis='x', rotation=30)
        rc = df['Region'].value_counts()
        axes[1,1].pie(rc.values, labels=rc.index, colors=COULEURS[:len(rc)], autopct='%1.0f%%')
        axes[1,1].set_title('Repartition par Region')
        plt.tight_layout()
        st.pyplot(fig); plt.close()


elif page == "1 - Regression Lineaire Simple":
    st.title("1 - Regression Lineaire Simple")
    st.markdown("**Objectif :** Relation entre **Superficie (X)** et **Production (Y)**.")
    st.markdown("**Equation :** Production = a x Superficie + b")
    st.markdown("---")
    if not check_data(3): st.stop()
    df = st.session_state.db.copy()
    X = df[['Superficie']].values; y = df['Production'].values
    model = LinearRegression(); model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred); rmse = np.sqrt(mean_squared_error(y, y_pred))
    a = model.coef_[0]; b = model.intercept_
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R2", f"{r2:.4f}"); col2.metric("R Pearson", f"{r2**0.5:.4f}")
    col3.metric("RMSE", f"{rmse:.1f} kg"); col4.metric("Exploitants", len(df))
    st.markdown("---")
    st.info(f"Production = {a:.0f} x Superficie + {b:.0f}")
    st.markdown(f"- Chaque hectare supplementaire produit en moyenne **+{a:.0f} kg**")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(7,5))
        ax.scatter(X, y, color='#27ae60', edgecolors='black', s=70, label='Exploitants')
        x_line = np.linspace(X.min(), X.max(), 100).reshape(-1,1)
        ax.plot(x_line, model.predict(x_line), color='#e74c3c', linewidth=2.5, label='Droite')
        ax.set_xlabel("Superficie (ha)"); ax.set_ylabel("Production (kg)")
        ax.set_title(f"Regression Simple (R2={r2:.3f})")
        ax.legend(); ax.grid(True,alpha=0.3); st.pyplot(fig); plt.close()
    with col2:
        fig, ax = plt.subplots(figsize=(7,5))
        ax.scatter(y, y_pred, color='#f39c12', edgecolors='black', s=70)
        lims = [y.min(), y.max()]; ax.plot(lims, lims, 'r--', linewidth=1.5)
        ax.set_xlabel("Reel (kg)"); ax.set_ylabel("Predit (kg)"); ax.set_title("Reel vs Predit")
        ax.grid(True,alpha=0.3); st.pyplot(fig); plt.close()
    st.markdown("---")
    sup_val = st.slider("Superficie (hectares)", 1, 50, 5)
    pred = model.predict([[sup_val]])[0]
    st.success(f"Pour {sup_val} hectares -> Production predite : **{pred:,.0f} kg**")


elif page == "2 - Regression Lineaire Multiple":
    st.title("2 - Regression Lineaire Multiple")
    st.markdown("**Objectif :** Predire la **Production** avec Age + Superficie + Culture + Satisfaction.")
    st.markdown("---")
    if not check_data(5): st.stop()
    df = st.session_state.db.copy()
    features = ['Age','Superficie','Culture_num','Satisfaction']
    labels_f = ['Age','Superficie','Type Culture','Satisfaction']
    X = df[features].values; y = df['Production'].values
    scaler = StandardScaler(); X_sc = scaler.fit_transform(X)
    model = LinearRegression(); model.fit(X_sc, y)
    y_pred = model.predict(X_sc)
    r2 = r2_score(y, y_pred); rmse = np.sqrt(mean_squared_error(y, y_pred))
    col1,col2,col3,col4 = st.columns(4)
    col1.metric("R2",f"{r2:.4f}"); col2.metric("R multiple",f"{max(0,r2)**0.5:.4f}")
    col3.metric("RMSE",f"{rmse:.1f} kg"); col4.metric("Variables",4)
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        coef_df = pd.DataFrame({'Variable':labels_f,'Coefficient':[f"{c:+.0f}" for c in model.coef_],
                                 'Effet':['Positif' if c>0 else 'Negatif' for c in model.coef_]})
        st.dataframe(coef_df, use_container_width=True, hide_index=True)
        corr = df[features].rename(columns=dict(zip(features,labels_f))).corr()
        fig, ax = plt.subplots(figsize=(5,4))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='Greens', ax=ax)
        ax.set_title('Correlations'); st.pyplot(fig); plt.close()
    with col2:
        fig, ax = plt.subplots(figsize=(6,4))
        colors = ['#27ae60' if c>0 else '#e74c3c' for c in model.coef_]
        ax.barh(labels_f, model.coef_, color=colors, edgecolor='white')
        ax.set_title("Impact sur la Production"); ax.axvline(0,color='black',linewidth=0.8)
        ax.grid(True,axis='x',alpha=0.3); st.pyplot(fig); plt.close()
        fig, ax = plt.subplots(figsize=(6,4))
        ax.scatter(y, y_pred, color='#27ae60', edgecolors='black', s=60)
        lims=[y.min(),y.max()]; ax.plot(lims,lims,'r--',linewidth=1.5)
        ax.set_xlabel("Reel (kg)"); ax.set_ylabel("Predit (kg)"); ax.set_title(f"R2={r2:.3f}")
        ax.grid(True,alpha=0.3); st.pyplot(fig); plt.close()


elif page == "3 - Reduction PCA":
    st.title("3 - Reduction de Dimensionnalite (PCA)")
    st.markdown("**Objectif :** Compresser les variables agricoles en 2D pour visualiser les structures.")
    st.markdown("---")
    if not check_data(5): st.stop()
    df = st.session_state.db.copy()
    features = ['Age','Superficie','Culture_num','Production','Satisfaction']
    labels_f = ['Age','Superficie','Culture','Production','Satisfaction']
    X = df[features].values
    scaler = StandardScaler(); X_sc = scaler.fit_transform(X)
    pca_full = PCA(); pca_full.fit(X_sc)
    var_exp = pca_full.explained_variance_ratio_; var_cum = np.cumsum(var_exp)
    pca2 = PCA(n_components=2); X_pca = pca2.fit_transform(X_sc)
    col1,col2,col3,col4 = st.columns(4)
    col1.metric("Variance PC1",f"{var_exp[0]:.1%}"); col2.metric("Variance PC2",f"{var_exp[1]:.1%}")
    col3.metric("Cumul PC1+PC2",f"{(var_exp[0]+var_exp[1]):.1%}"); col4.metric("Variables",5)
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6,4))
        ax.bar(range(1,len(var_exp)+1), var_exp*100, color='#27ae60', edgecolor='white', label='Par composante')
        ax.plot(range(1,len(var_exp)+1), var_cum*100, 'o-', color='#f39c12', linewidth=2, label='Cumule')
        ax.axhline(y=95, color='red', linestyle='--', label='95%')
        ax.set_xlabel("Composante"); ax.set_ylabel("Variance (%)"); ax.set_title("Scree Plot")
        ax.legend(); ax.grid(True,alpha=0.3); st.pyplot(fig); plt.close()
        st.dataframe(pd.DataFrame(pca2.components_.T,index=labels_f,columns=['PC1','PC2']).round(3), use_container_width=True)
    with col2:
        fig, ax = plt.subplots(figsize=(6,5))
        regions = df['Region'].unique()
        for i, reg in enumerate(regions):
            mask = df['Region']==reg
            ax.scatter(X_pca[mask,0], X_pca[mask,1], color=COULEURS[i%len(COULEURS)],
                       label=reg, edgecolors='black', s=70, alpha=0.9)
        ax.set_xlabel(f"PC1 ({var_exp[0]:.1%})"); ax.set_ylabel(f"PC2 ({var_exp[1]:.1%})")
        ax.set_title("PCA 2D (par Region)"); ax.legend(fontsize=7); ax.grid(True,alpha=0.3)
        st.pyplot(fig); plt.close()


elif page == "4 - Classification Supervisee":
    st.title("4 - Classification Supervisee (KNN)")
    st.markdown("**Objectif :** Predire le **type de culture** a partir de Age, Superficie, Production, Satisfaction.")
    st.markdown("---")
    if not check_data(6): st.stop()
    df = st.session_state.db.copy()
    features = ['Age','Superficie','Production','Satisfaction']
    X = df[features].values; y = df['Culture_num'].values
    scaler = StandardScaler(); X_sc = scaler.fit_transform(X)
    k = min(5, len(df)-1)
    model_knn = KNeighborsClassifier(n_neighbors=k)
    model_knn.fit(X_sc, y); y_pred = model_knn.predict(X_sc)
    acc = accuracy_score(y, y_pred)
    col1,col2,col3,col4 = st.columns(4)
    col1.metric("Accuracy",f"{acc:.1%}"); col2.metric("K voisins",k)
    col3.metric("Cultures",len(np.unique(y))); col4.metric("Variables",4)
    st.markdown("---")
    inv_cult = {v: k for k, v in CULTURES.items()}
    col1, col2 = st.columns(2)
    with col1:
        uniq = np.unique(y)
        cult_names = [inv_cult.get(n,str(n)) for n in uniq]
        prec = [np.mean(y_pred[np.where(y==c)[0]]==c) if len(np.where(y==c)[0]) else 0 for c in uniq]
        fig, ax = plt.subplots(figsize=(6,4))
        ax.bar(cult_names, [p*100 for p in prec], color=COULEURS[:len(uniq)], edgecolor='white')
        ax.set_ylabel("Precision (%)"); ax.set_title("Precision par Culture")
        ax.tick_params(axis='x',rotation=20); ax.set_ylim(0,115); ax.grid(True,axis='y',alpha=0.3)
        st.pyplot(fig); plt.close()
    with col2:
        cm = confusion_matrix(y, y_pred)
        fig, ax = plt.subplots(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax, xticklabels=cult_names, yticklabels=cult_names)
        ax.set_xlabel("Predit"); ax.set_ylabel("Reel"); ax.set_title("Matrice de Confusion")
        ax.tick_params(axis='x',rotation=20); st.pyplot(fig); plt.close()
        st.markdown(f"### Accuracy : **{acc:.1%}**"); st.progress(acc)
    st.markdown("---")
    st.markdown("### Predire la culture d'un nouvel exploitant")
    c1,c2,c3,c4 = st.columns(4)
    pa=c1.number_input("Age",15,80,35,key='k_age')
    ps_=c2.number_input("Superficie (ha)",0.5,100.0,3.0,step=0.5,key='k_sup')
    pp=c3.number_input("Production (kg)",0,100000,1500,step=100,key='k_prod')
    psat=c4.slider("Satisfaction",1,10,6,key='k_sat')
    if st.button("Predire la culture"):
        inp = scaler.transform([[pa,ps_,pp,psat]])
        pred = model_knn.predict(inp)[0]
        st.success(f"Culture predite : **{inv_cult.get(pred,pred)}**")


elif page == "5 - Clustering KMeans":
    st.title("5 - Classification Non-Supervisee (K-Means)")
    st.markdown("**Objectif :** Segmenter les exploitants en groupes selon Superficie et Production.")
    st.markdown("---")
    if not check_data(5): st.stop()
    df = st.session_state.db.copy()
    K_max = min(6, len(df)-1)
    K = st.slider("Nombre de clusters (K)", 2, K_max, min(3,K_max))
    X = df[['Superficie','Production']].values
    scaler = StandardScaler(); X_sc = scaler.fit_transform(X)
    km = KMeans(n_clusters=K, random_state=42, n_init=10)
    df['Cluster'] = km.fit_predict(X_sc)
    col1,col2,col3,col4 = st.columns(4)
    col1.metric("Clusters",K); col2.metric("Inertie",f"{km.inertia_:.1f}")
    col3.metric("Exploitants",len(df)); col4.metric("Variables",2)
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6,5))
        for c in range(K):
            mask = df['Cluster']==c
            ax.scatter(df.loc[mask,'Superficie'], df.loc[mask,'Production'],
                       color=COULEURS[c%len(COULEURS)], label=f'Groupe {c}', edgecolors='black', s=80)
        centers_orig = scaler.inverse_transform(km.cluster_centers_)
        ax.scatter(centers_orig[:,0], centers_orig[:,1], marker='X', s=200, c='black', zorder=5, label='Centres')
        ax.set_xlabel("Superficie (ha)"); ax.set_ylabel("Production (kg)")
        ax.set_title(f"K-Means Exploitants (K={K})"); ax.legend(fontsize=8); ax.grid(True,alpha=0.3)
        st.pyplot(fig); plt.close()
        profile = df.groupby('Cluster')[['Age','Superficie','Production','Satisfaction']].mean().round(1)
        profile.index = [f"Groupe {i}" for i in profile.index]
        st.dataframe(profile, use_container_width=True)
    with col2:
        ks = range(2,K_max+1)
        inertias = [KMeans(n_clusters=k,random_state=42,n_init=10).fit(X_sc).inertia_ for k in ks]
        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(list(ks), inertias, 'o-', color='#27ae60', linewidth=2.5, markersize=8)
        ax.axvline(x=K, color='red', linestyle='--', label=f'K={K}')
        ax.set_xlabel("K"); ax.set_ylabel("Inertie"); ax.set_title("Methode du Coude")
        ax.legend(); ax.grid(True,alpha=0.3); st.pyplot(fig); plt.close()
        sizes = df['Cluster'].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(6,4))
        ax.bar([f'Groupe {i}' for i in sizes.index], sizes.values,
               color=[COULEURS[i%len(COULEURS)] for i in sizes.index], edgecolor='white')
        ax.set_ylabel("Nombre d'exploitants"); ax.set_title("Effectif par Groupe")
        ax.grid(True,axis='y',alpha=0.3); st.pyplot(fig); plt.close()
    st.markdown("---")
    col1, col2 = st.columns(2)
    col1.info("**Supervise (KNN)** : On connait la culture -> On predit pour un nouvel exploitant")
    col2.warning("**Non-Supervise (KMeans)** : L'algorithme regroupe seul les exploitants similaires")
