from pathlib import Path

import streamlit as st
import joblib
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import normalize

# Configuration de la page
st.set_page_config(page_title="Segmentation - Examen ML", page_icon=":material/scatter_plot:", layout="centered")

# Dossier racine du projet
ROOT = Path(__file__).parent

# Style : en-tête de la page et cartes
st.markdown("""
<style>
@import url("https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,400,0,0");
.block-container {padding-top: 2.5rem;}
.hero {
    background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
    color: white; border-radius: 16px; padding: 1.6rem 1.8rem; margin-bottom: 1.2rem;
}
.hero .icon {font-family: 'Material Symbols Rounded'; font-size: 2.4rem; line-height: 1;
    font-weight: normal; font-style: normal; display: inline-block; -webkit-font-smoothing: antialiased;}
.hero h1 {color: white; font-size: 1.7rem; margin: .5rem 0 .3rem 0; padding: 0;}
.hero p {color: rgba(255,255,255,.88); margin: 0; font-size: .97rem;}
.hero .tag {
    display: inline-block; background: rgba(255,255,255,.2); border-radius: 999px;
    padding: .15rem .7rem; font-size: .78rem; font-weight: 600; letter-spacing: .02em;
}
.result-label {color: #64748b; font-size: .85rem; margin-bottom: .2rem;}
.result-value {font-size: 2.1rem; font-weight: 700; line-height: 1.2;}
</style>
""", unsafe_allow_html=True)


# =====================================================================
# OUTILS COMMUNS
# =====================================================================

# Charger les objets sauvegardés d'un dossier (une seule fois grâce au cache)
@st.cache_resource
def load_objects(dossier):
    path = ROOT / dossier
    colonnes = joblib.load(path / 'colonnes.joblib')
    centroides = joblib.load(path / 'centroides.joblib')
    noms_classes = joblib.load(path / 'noms_classes.joblib')
    uniques = joblib.load(path / 'uniques.joblib') if (path / 'uniques.joblib').exists() else None
    return colonnes, centroides, noms_classes, uniques


# Charger le jeu de données d'origine (pour les valeurs par défaut et l'exemple de CSV)
@st.cache_data
def load_dataset(dossier, csv_name, colonnes):
    df = pd.read_csv(ROOT / dossier / csv_name)
    return df[colonnes]


# Construire la fonction de prédiction à partir des objets sauvegardés
def make_pred_func(centroides, noms_classes, colonnes, uniques=None, cat_col=None):
    cat_idx = colonnes.index(cat_col) if cat_col else None

    def Pred_func(*args):
        valeurs = list(args)
        # Encoder la variable catégorielle (même ordre que le LabelEncoder)
        if cat_idx is not None and isinstance(valeurs[cat_idx], str):
            valeurs[cat_idx] = uniques.index(valeurs[cat_idx].strip())
        # Vecteur des valeurs numériques
        x_new = np.array(valeurs, dtype=float).reshape(1, -1)
        # Normaliser
        x_new = normalize(x_new)
        # Prédire puis décoder la classe prédite
        classe = int(cdist(x_new, centroides).argmin())
        return noms_classes[classe]

    return Pred_func


# En-tête de la page
def header(icon, title, tag, description):
    st.markdown(f"""
    <div class="hero">
        <div class="icon">{icon}</div>
        <h1>{title}</h1>
        <p><span class="tag">{tag}</span></p>
        <p style="margin-top:.6rem">{description}</p>
    </div>
    """, unsafe_allow_html=True)


# Carte de résultat
def show_value(label, value):
    with st.container(border=True):
        st.markdown(f'<div class="result-label">{label}</div>'
                    f'<div class="result-value">{value}</div>', unsafe_allow_html=True)


# Onglets de la page
def tabs():
    return st.tabs([':material/edit_note: Simple Prédiction', ':material/upload_file: Prédiction multiple'])


# Formulaire de prédiction simple, généré à partir de la configuration des champs
# groupes = [(titre, [(colonne, libellé, type), ...]), ...]
#   type : 'int', 'float' ou une liste d'options
def simple_tab(key, groupes, colonnes, defaults, pred_func, result_label, prefix=""):
    valeurs = {}
    with st.form(f'form_{key}'):
        for titre, champs in groupes:
            st.markdown(f"**{titre}**")
            for i in range(0, len(champs), 3):
                ligne = champs[i:i + 3]
                cols = st.columns(len(ligne))
                for col, (nom, label, kind) in zip(cols, ligne):
                    d = defaults[nom]
                    if isinstance(kind, list):
                        idx = kind.index(d) if d in kind else 0
                        valeurs[nom] = col.selectbox(label, options=kind, index=idx, key=f'{key}_{nom}')
                    elif kind == 'int':
                        valeurs[nom] = col.number_input(label, min_value=0, value=int(d), step=1,
                                                        key=f'{key}_{nom}')
                    else:
                        valeurs[nom] = col.number_input(label, min_value=0.0, value=float(d),
                                                        format="%.4f", key=f'{key}_{nom}')
        submit = st.form_submit_button('Prédire', type='primary', width='stretch', icon=":material/bolt:")
    if submit:
        try:
            classe = pred_func(*[valeurs[c] for c in colonnes])
            show_value(result_label, f"{prefix}{classe}")
        except ValueError as e:
            st.error(f"Impossible de prédire : {e}")


# Fonction de prédiction multiple
def Pred_func_csv(df, pred_func, colonnes, target_name):
    # Utiliser les colonnes par leur nom si elles existent, sinon par position
    if all(c in df.columns for c in colonnes):
        X = df[colonnes]
    else:
        X = df.iloc[:, :len(colonnes)]
    predictions = [pred_func(*row) for row in X.itertuples(index=False)]
    df = df.copy()
    df[target_name] = predictions
    return df


# Onglet de prédiction multiple
def multiple_tab(key, colonnes, sample_df, pred_func, target_name):
    n_features = len(colonnes)

    # Étape 1 : format attendu
    with st.container(border=True):
        st.markdown("**1. Préparer le fichier**")
        st.caption("Le fichier CSV doit contenir ces colonnes (par leur nom, ou dans cet ordre) :")
        st.markdown(" ".join(f"`{c}`" for c in colonnes))
        st.download_button("Télécharger un exemple de CSV",
                           data=sample_df.head().to_csv(index=False).encode('utf-8'),
                           file_name=f'exemple_{key}.csv', mime='text/csv', icon=":material/download:",
                           on_click='ignore', key=f'sample_{key}')

    # Étape 2 : chargement
    with st.container(border=True):
        st.markdown("**2. Charger le fichier**")
        file = st.file_uploader('Charger un fichier CSV', type=['csv'], key=f'file_{key}',
                                label_visibility='collapsed')

    if file is None:
        st.session_state.pop(f'pred_{key}', None)
        return

    # Lire le fichier csv
    df = pd.read_csv(file)
    if df.shape[1] < n_features:
        st.error(f"Le fichier contient {df.shape[1]} colonnes, il en faut au moins {n_features}.")
        return

    with st.expander(f"Aperçu des données chargées ({len(df)} lignes)"):
        st.dataframe(df.head(), hide_index=True)

    if st.button('Prédire', key=f'btn_{key}', type='primary', width='stretch', icon=":material/bolt:"):
        try:
            st.session_state[f'pred_{key}'] = Pred_func_csv(df, pred_func, colonnes, target_name)
        except (ValueError, TypeError) as e:
            st.error(f"Une valeur du fichier n'est pas reconnue par le modèle : {e}")

    # Étape 3 : résultats (gardés en mémoire entre deux actions)
    df_pred = st.session_state.get(f'pred_{key}')
    if df_pred is not None:
        st.markdown("**3. Résultats**")
        counts = df_pred[target_name].value_counts()
        st.metric("Lignes prédites", len(df_pred), border=True)
        cols = st.columns(len(counts))
        for col, (classe, n) in zip(cols, counts.items()):
            col.metric(classe, n, border=True)
        st.dataframe(df_pred, hide_index=True,
                     column_config={target_name: st.column_config.Column(target_name)})
        # Télécharger le fichier des prédictions
        st.download_button(label='Télécharger le fichier CSV',
                           data=df_pred.to_csv(index=False).encode('utf-8'),
                           file_name=f'predictions_{key}.csv',
                           mime='text/csv', type='primary', width='stretch',
                           icon=":material/download:", on_click='ignore', key=f'dl_{key}')


# Construire une page complète à partir de sa configuration
def build_page(key, dossier, csv_name, icon, title, tag, description, groupes,
               result_label, target_name, prefix="", cat_col=None):
    colonnes, centroides, noms_classes, uniques = load_objects(dossier)
    data = load_dataset(dossier, csv_name, colonnes)
    defaults = data.iloc[0].to_dict()
    pred_func = make_pred_func(centroides, noms_classes, colonnes, uniques, cat_col)

    header(icon, title, tag, description)
    with st.expander(f"Segments possibles ({len(noms_classes)})"):
        st.markdown("\n".join(f"- **{v}**" for v in noms_classes.values()))

    tab1, tab2 = tabs()
    with tab1:
        simple_tab(key, groupes, colonnes, defaults, pred_func, result_label, prefix)
    with tab2:
        multiple_tab(key, colonnes, data, pred_func, target_name)


# =====================================================================
# PAGE 1 : segmentation des clients (télécommunications)
# =====================================================================
def page_clients():
    build_page(
        key='clients', dossier='Cusmtomer_Segmentation', csv_name='Customer_dataset.csv',
        icon="group", title="Segmentation des clients", tag="Clustering · CAH",
        description="""Affecte un client d'une entreprise de télécommunications à son segment
        (Fidele ou Non Fidele) à partir de son utilisation, de son abonnement et de son âge.""",
        groupes=[
            ("Utilisation", [
                ('Call  Failure', 'Appels échoués', 'int'),
                ('Seconds of Use', "Secondes d'utilisation", 'int'),
                ('Frequency of use', "Fréquence d'utilisation", 'int'),
                ('Frequency of SMS', 'Fréquence des SMS', 'int'),
                ('Distinct Called Numbers', 'Numéros distincts appelés', 'int'),
            ]),
            ("Abonnement", [
                ('Subscription  Length', "Durée d'abonnement (mois)", 'int'),
                ('Charge  Amount', 'Montant facturé', list(range(11))),
                ('Customer Value', 'Valeur client', 'float'),
                ('Complains', 'Plaintes', [0, 1]),
                ('Status', 'Statut', [1, 2]),
            ]),
            ("Client", [
                ('Age Group', "Groupe d'âge", [1, 2, 3, 4, 5]),
                ('Age', 'Âge', 'int'),
            ]),
        ],
        result_label="Segment prédit", target_name='Segment', prefix="Client ",
    )


# =====================================================================
# PAGE 2 : clients d'un grossiste
# =====================================================================
def page_wholesale():
    build_page(
        key='wholesale', dossier='Wholesale_Customers', csv_name='Wholesale_customers_data.csv',
        icon="storefront", title="Clients d'un grossiste", tag="Clustering · CAH",
        description="""Affecte un client d'un grossiste à son segment à partir de son canal,
        de sa région et de ses dépenses annuelles par catégorie de produits.""",
        groupes=[
            ("Client", [
                ('Channel', 'Canal (1 = Horeca, 2 = Détail)', [1, 2]),
                ('Region', 'Région (1 = Lisbonne, 2 = Porto, 3 = Autre)', [1, 2, 3]),
            ]),
            ("Dépenses annuelles", [
                ('Fresh', 'Produits frais', 'int'),
                ('Milk', 'Produits laitiers', 'int'),
                ('Grocery', 'Épicerie', 'int'),
                ('Frozen', 'Surgelés', 'int'),
                ('Detergents_Paper', 'Détergents & papier', 'int'),
                ('Delicassen', 'Traiteur', 'int'),
            ]),
        ],
        result_label="Segment prédit", target_name='Segment',
    )


# =====================================================================
# PAGE 3 : publications Facebook Live
# =====================================================================
def page_facebook():
    _, _, _, uniques = load_objects('Facebook_Live_Sellers')
    build_page(
        key='facebook', dossier='Facebook_Live_Sellers', csv_name='Live_20210128.csv',
        icon="live_tv", title="Publications Facebook Live", tag="Clustering · CAH",
        description="""Affecte une publication d'un vendeur Facebook Live à son profil d'engagement
        à partir de son type et de ses réactions.""",
        groupes=[
            ("Publication", [
                ('status_type', 'Type de publication', list(uniques)),
            ]),
            ("Engagement", [
                ('num_reactions', 'Réactions', 'int'),
                ('num_comments', 'Commentaires', 'int'),
                ('num_shares', 'Partages', 'int'),
            ]),
            ("Détail des réactions", [
                ('num_likes', "J'aime", 'int'),
                ('num_loves', "J'adore", 'int'),
                ('num_wows', 'Wouah', 'int'),
                ('num_hahas', 'Haha', 'int'),
                ('num_sads', 'Triste', 'int'),
                ('num_angrys', 'Grrr', 'int'),
            ]),
        ],
        result_label="Profil prédit", target_name='Profil', cat_col='status_type',
    )


# =====================================================================
# PAGE 4 : graines de blé
# =====================================================================
def page_wheat():
    build_page(
        key='wheat', dossier='wheat_seeds', csv_name='wheat_seeds_dataset.csv',
        icon="grass", title="Graines de blé", tag="Clustering · CAH",
        description="""Affecte une graine de blé à sa classe (petites ou grandes graines)
        à partir de ses mesures géométriques.""",
        groupes=[
            ("Mesures de la graine", [
                ('area A', 'Surface', 'float'),
                ('perimeter', 'Périmètre', 'float'),
                ('compactness', 'Compacité', 'float'),
                ('length of kernel', 'Longueur du grain', 'float'),
                ('width of kernel', 'Largeur du grain', 'float'),
                ('asymmetry coefficient', "Coefficient d'asymétrie", 'float'),
                ('length of kernel groove', 'Longueur du sillon', 'float'),
            ]),
        ],
        result_label="Classe prédite", target_name='Classe',
    )


# =====================================================================
# PAGE D'ACCUEIL : présentation du projet
# =====================================================================
# Résumé de chaque jeu de données (affiché sur la page d'accueil)
PROJETS = [
    ('Cusmtomer_Segmentation', 'Customer_dataset.csv', "group", "Clients télécom",
     "Identifier les clients fidèles d'un opérateur de télécommunications."),
    ('Wholesale_Customers', 'Wholesale_customers_data.csv', "storefront", "Clients grossiste",
     "Regrouper les clients d'un grossiste selon leurs dépenses annuelles."),
    ('Facebook_Live_Sellers', 'Live_20210128.csv', "live_tv", "Facebook Live",
     "Classer les publications de vendeurs selon leur engagement."),
    ('wheat_seeds', 'wheat_seeds_dataset.csv', "grass", "Graines de blé",
     "Distinguer les graines de blé selon leurs mesures géométriques."),
]


def page_accueil():
    header("scatter_plot", "Segmentation par clustering", "Examen Machine Learning",
           """Quatre jeux de données, deux algorithmes non supervisés (K-Means et CAH) et une
           application pour affecter de nouvelles observations à leur segment.""")

    # Chiffres clés calculés à partir des objets sauvegardés
    infos = []
    for dossier, csv_name, icon, titre, desc in PROJETS:
        colonnes, _, noms_classes, _ = load_objects(dossier)
        n_obs = len(load_dataset(dossier, csv_name, colonnes))
        infos.append((dossier, icon, titre, desc, colonnes, noms_classes, n_obs))

    c1, c2, c3 = st.columns(3)
    c1.metric("Jeux de données", len(infos), border=True)
    c2.metric("Observations", f"{sum(i[6] for i in infos):,}".replace(",", " "), border=True)
    c3.metric("Segments au total", sum(len(i[5]) for i in infos), border=True)

    # Cartes des jeux de données
    st.subheader("Les jeux de données")
    for row in range(0, len(infos), 2):
        cols = st.columns(2)
        for col, (dossier, icon, titre, desc, colonnes, noms_classes, n_obs) in zip(cols, infos[row:row + 2]):
            with col.container(border=True, height="stretch"):
                st.markdown(f"#### :material/{icon}: {titre}")
                st.caption(desc)
                st.markdown(f"**{n_obs}** observations · **{len(colonnes)}** variables · "
                            f"**{len(noms_classes)}** segments")
                st.markdown(" ".join(f"`{v}`" for v in noms_classes.values()))
                st.page_link(PAGES[dossier], label="Ouvrir la page", icon=":material/arrow_forward:")

    # Démarche suivie dans les notebooks
    st.subheader("Démarche")
    etapes = [
        ("Préparation", "Exploration, nettoyage et encodage des données."),
        ("Normalisation", "Chaque observation est normalisée (`normalize`)."),
        ("K-Means", "Nombre de classes choisi par la méthode du coude."),
        ("CAH", "Liaison *average*, nombre de classes lu sur le dendrogramme."),
        ("Comparaison", "Le score de silhouette départage les deux modèles."),
        ("Déploiement", "Les centroïdes de la CAH sont sauvegardés en `.joblib`."),
    ]
    for row in range(0, len(etapes), 3):
        cols = st.columns(3)
        for i, (col, (titre, texte)) in enumerate(zip(cols, etapes[row:row + 3]), start=row + 1):
            with col.container(border=True, height="stretch"):
                st.markdown(f"**{i}. {titre}**")
                st.caption(texte)

    # Fonctionnement de la prédiction
    st.subheader("Comment fonctionne la prédiction ?")
    st.markdown("""Une CAH ne sait pas classer directement une nouvelle observation. L'application
    **normalise l'observation**, puis lui attribue **la classe du centroïde le plus proche**
    (distance euclidienne), et affiche le nom du segment.""")
    st.info("Choisis un jeu de données dans le menu de gauche. Chaque page permet une prédiction "
            "simple (formulaire) ou multiple (fichier CSV).", icon=":material/info:")


# =====================================================================
# NAVIGATION
# =====================================================================
PAGES = {
    'Cusmtomer_Segmentation': st.Page(page_clients, title="Clients télécom", icon=":material/group:", url_path="clients"),
    'Wholesale_Customers': st.Page(page_wholesale, title="Clients grossiste", icon=":material/storefront:", url_path="grossiste"),
    'Facebook_Live_Sellers': st.Page(page_facebook, title="Facebook Live", icon=":material/live_tv:", url_path="facebook"),
    'wheat_seeds': st.Page(page_wheat, title="Graines de blé", icon=":material/grass:", url_path="graines"),
}
pages = st.navigation({
    "Projet": [st.Page(page_accueil, title="Accueil", icon=":material/home:", url_path="accueil", default=True)],
    "Modèles": list(PAGES.values()),
})
pages.run()
