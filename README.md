# Examen ML : segmentation par clustering

Ce projet applique le **clustering** (K-Means et CAH) à quatre jeux de données. Les modèles retenus (CAH) sont déployés dans une **application Streamlit** : on y affecte une nouvelle observation à son segment, une à une ou à partir d'un fichier CSV.

## Jeux de données

| Dossier | Données | Nombre de classes (CAH) | Segments |
|---|---|---|---|
| `Cusmtomer_Segmentation/` | Clients d'un opérateur télécom | 2 | Fidele, Non Fidele |
| `Wholesale_Customers/` | Clients d'un grossiste | 3 | Gros acheteurs de produits frais, Gros acheteurs d'épicerie, Client atypique |
| `Facebook_Live_Sellers/` | Publications de vendeurs Facebook Live | 4 | Sans engagement, Lives très engageants, Viraux par partage, Photos likées |
| `wheat_seeds/` | Mesures de graines de blé | 2 | Petites graines, Grandes graines |

## Structure du projet

```
EXAMEN ML/
├── app.py                  # Application Streamlit (4 pages)
├── requirements.txt        # Dépendances Python
├── README.md
└── <jeu_de_données>/       # Un dossier par jeu de données
    ├── *.ipynb             # Notebook : exploration, K-Means, CAH, comparaison
    ├── *.csv               # Données brutes
    ├── colonnes.joblib     # Liste des variables, dans l'ordre utilisé par le modèle
    ├── centroides.joblib   # Centroïdes des classes de la CAH (données normalisées)
    ├── noms_classes.joblib # Dictionnaire numéro de classe → nom du segment
    └── uniques.joblib      # (Facebook seulement) modalités de status_type
```

## Méthode

Chaque notebook suit les mêmes étapes :

1. Exploration et nettoyage des données. Pour Facebook : suppression des colonnes vides, des doublons, de l'identifiant et de la date, puis encodage de `status_type` avec `LabelEncoder`.
2. Normalisation des observations avec `sklearn.preprocessing.normalize`.
3. Réduction à 2 composantes avec une ACP, pour la visualisation.
4. **K-Means**, avec un nombre de classes choisi par la méthode du coude.
5. **CAH** (`AgglomerativeClustering`, liaison *average*), avec un nombre de classes choisi sur le dendrogramme.
6. Comparaison des deux modèles avec le **score de silhouette**.
7. Calcul des centroïdes des classes de la CAH, interprétation des profils moyens et sauvegarde des objets en `.joblib`.

Une CAH ne sait pas prédire directement la classe d'une nouvelle observation. L'application **normalise donc l'observation, puis lui attribue la classe du centroïde le plus proche** (distance euclidienne).

## Installation

Il faut Python 3.10 ou plus récent.

```bash
cd "EXAMEN ML"
python3 -m venv .venv            # créer l'environnement virtuel
source .venv/bin/activate        # l'activer (Windows : .venv\Scripts\activate)
pip install -r requirements.txt  # installer les dépendances
```

## Lancer l'application

```bash
source .venv/bin/activate
streamlit run app.py
```

L'application s'ouvre sur http://localhost:8501.

## Utilisation

On choisit un jeu de données dans le menu latéral. Chaque page propose deux onglets :

- **Simple Prédiction** : on remplit le formulaire (pré-rempli avec la première ligne du jeu de données), puis on clique sur **Prédire** pour afficher le segment.
- **Prédiction multiple** :
  1. Télécharger l'exemple de CSV pour voir le format attendu.
  2. Charger son propre fichier. Les colonnes sont reconnues par leur nom, ou à défaut par leur ordre. Les CSV d'origine sont donc acceptés tels quels, y compris celui de Facebook avec `status_type` en texte.
  3. Cliquer sur **Prédire**. L'application affiche le nombre d'observations par segment et propose de télécharger le CSV avec les prédictions.

## Technologies

Python · pandas · NumPy · scikit-learn · SciPy · joblib · Streamlit · Plotly et Yellowbrick (dans les notebooks)
