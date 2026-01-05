import streamlit as st
import pandas as pd
import os
import json
from google.cloud import firestore
from google.oauth2 import service_account
from datetime import datetime
from sections.menu.menu import custom_sidebar_menu
from sections.constantes import cols_cumul_sum as initial_cols_cumul_sum
from sections.gps.pipeline import recuperer_all_files_gps, recuperer_all_files_gps_match, add_columns_session_rpe
from sections.gps.norme import calculer_toutes_references, calculer_prescription_groupe_auto

# ----------------------------------------------------------------------
# 1. CONFIGURATION DE LA PAGE
# ----------------------------------------------------------------------
# Doit être le premier appel Streamlit
st.set_page_config(
    page_title="AMS RCL F",
    page_icon="⚽", # J'ai ajouté une icône
    layout="wide",
    initial_sidebar_state="expanded",
    # Cette partie masque les éléments du menu hamburger (trois points)
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# ----------------------------------------------------------------------
# 2. INJECTION CSS POUR MASQUER LA NAVIGATION NATIVE
# ----------------------------------------------------------------------
# Ceci masque les éléments 'Pages' natifs de Streamlit
hide_streamlit_pages_css = """
<style>
/* Masque la navigation de la page native (Home, GPS groupe, etc.) */
section[data-testid="stSidebar"] ul:first-child {
    display: none;
}
/* Masque le titre "Pages" ou "Accueil" juste au-dessus de la liste */
section[data-testid="stSidebar"] div[data-testid="stSidebarNav"] > div:first-child {
    display: none;
}
</style>
"""
st.markdown(hide_streamlit_pages_css, unsafe_allow_html=True)


custom_sidebar_menu()
st.sidebar.caption("Fichiers")

# Etat général
if 'all_gps_session' not in st.session_state:
    df_gps = recuperer_all_files_gps()
    st.session_state['all_gps_session'] = df_gps #stocker la clef
else:
    df_gps = st.session_state['all_gps_session']
calculer_toutes_references(df_gps)


if 'all_gps_match' not in st.session_state:
    df_gps_match = recuperer_all_files_gps_match()
    st.session_state['all_gps_match'] = df_gps_match
else:
    df_gps_match = st.session_state['all_gps_match']

if 'df_groupe_final' in st.session_state:
    df_groupe_final = st.session_state['df_groupe_final']

if 'df_best_match_all_players' in st.session_state:
    
    df_meilleur_match = st.session_state.get('df_best_match_all_players')
else:
    st.sidebar.write("DataFrame non encore calculé.")

# ----------------------------------------------------------------------
# 🚨 MODIFICATION 1 : CHARGEMENT ET PRÉPARATION DU FICHIER ID (Format Catapult)
# ----------------------------------------------------------------------
try:
    # Fonction d'aide pour normaliser les noms pour la RECHERCHE interne.
    # Nous utilisons PRÉNOM NOM en MAJUSCULES pour la recherche car cela matche
    # le format probable de vos données GPS/RPE.
    def normalize_catapult_name_for_lookup(name):
        """Convertit 'NOM, Prénom' en 'PRÉNOM NOM' et met tout en MAJUSCULES pour la RECHERCHE."""
        name = str(name).strip()
        if pd.isna(name):
            return ""
            
        name_upper = name.upper()
        
        # Tente de convertir le format 'NOM, Prénom'
        if ',' in name_upper:
            try:
                # Sépare en utilisant la casse MAJUSCULES
                parts = [p.strip() for p in name_upper.split(',', 1)]
                last = parts[0]
                first = parts[1]
                # Format de RECHERCHE : PRÉNOM NOM en MAJ
                return f"{first} {last}".strip()
            except:
                return name_upper
        else:
            # Si le format est déjà PRÉNOM NOM, retourne juste en MAJUSCULES
            return name_upper

    df_id_joueuses = pd.read_csv('data/ID_joueuses.csv') 
    df_id_joueuses = df_id_joueuses[['Athlete ID', 'Athlete Name', 'Position']].copy()
    
    if 'Athlete Name' in df_id_joueuses.columns:
        # 🚨 NOUVEAU : Crée une colonne pour la RECHERCHE 🚨
        df_id_joueuses['Normalized Name'] = df_id_joueuses['Athlete Name'].apply(normalize_catapult_name_for_lookup)
        
        # Le format original 'Athlete Name' (NOM, Prénom) est conservé pour la SORTIE (format Catapult)
    
    st.sidebar.success("Fichier ID Catapult chargé et noms normalisés pour la recherche.")

except FileNotFoundError:
    st.error("Erreur : Le fichier d'identification 'data/ID_joueuses.csv' n'a pas été trouvé. Veuillez vérifier le chemin.")
    df_id_joueuses = pd.DataFrame() 

###### TEST
# Initialisation du DataFrame complet si ce n'est pas déjà fait
FILE_PATH = "data/rpe.csv"
if not os.path.exists("data"):
    os.makedirs("data")

@st.cache_data
def load_and_merge_rpe(df_base):
    try:
        df_rpe = pd.read_csv(FILE_PATH)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        df_rpe = pd.DataFrame(columns=['Name', 'Date', 'Jour semaine', 'Cardio', 'Muscu', 'Presence'])
    
    # Conversion de la colonne 'Date' dans les deux DataFrames pour éviter l'erreur de fusion
    df_base['Date'] = pd.to_datetime(df_base['Date'], dayfirst=True)
    df_rpe['Date'] = pd.to_datetime(df_rpe['Date'], dayfirst=True, errors='coerce')
    
    df_merged = pd.merge(df_base, df_rpe, on=['Name', 'Date', 'Jour semaine'], how='left')
    df_merged['Presence'] = df_merged['Presence'].fillna('C')
    return df_merged

DF_REFERENCES_MOYENNES = {
    'J-1': st.session_state.get('ref_j1_cumule', pd.DataFrame()),
    'J-2': st.session_state.get('ref_j2_cumule', pd.DataFrame()),
    'J-3': st.session_state.get('ref_j3_cumule', pd.DataFrame()),
    'J-4': st.session_state.get('ref_j4_cumule', pd.DataFrame()),
}

if 'DF_REFERENCES_MOYENNES' not in st.session_state:
    st.session_state['DF_REFERENCES_MOYENNES'] = DF_REFERENCES_MOYENNES

if 'df_joueuse_rpe_complete' not in st.session_state:
    st.session_state['df_joueuse_rpe_complete'] = load_and_merge_rpe(df_gps.copy())

filtered_df = st.session_state['df_joueuse_rpe_complete']
filtered_df = add_columns_session_rpe(df=filtered_df)
filtered_df['Date'] = pd.to_datetime(filtered_df['Date'],format='%Y-%m-%d')
filtered_df['Date'] = filtered_df['Date'].dt.strftime('%Y-%m-%d')

# ----------------------------------------------------------------------
# 🚨 MODIFICATION 2 : NETTOYAGE DES COLONNES INDÉSIRABLES
# ----------------------------------------------------------------------
cols_a_retirer_finales = ['Total Time', 'S. Cardio', 'S. Muscu']

filtered_df = filtered_df.drop(
    columns=[col for col in cols_a_retirer_finales if col in filtered_df.columns], 
    errors='ignore'
)

# ----------------------------------------------------------------------
# 🚨 MODIFICATION 3 : DÉFINITION DE LA LISTE DE COLONNES FINALES (cols_cumul_sum)
# ----------------------------------------------------------------------
# BUT : Retirer uniquement les colonnes inutiles (Total Time, S. Cardio, S. Muscu)
# ET GARANTIR QUE 'High Speed Distance' RESTE POUR ÊTRE UTILISÉE ET RECALCULÉE.
cols_a_retirer_temporaires = ['Total Time', 'S. Cardio', 'S. Muscu', 'Sprint(m)'] # On enlève 'Sprint(m)' au cas où il existerait
cols_cumul_sum = [
    col for col in initial_cols_cumul_sum 
    if col not in cols_a_retirer_temporaires
]

# Note: Si 'High Speed Distance' était dans la liste initiale, il y reste. C'est ce que nous voulons.
# Nous n'avons plus besoin de la ligne qui ajoutait 'Sprint(m)' car nous utilisons HSD.


if not filtered_df.empty:
    # Correction: On s'assure que toutes les colonnes numériques sont correctement typées AVANT tout calcul
    # On itère sur toutes les colonnes de cols_cumul_sum (qui inclut HSD) plus les colonnes SPR et SPR+ pour être sûr
    for col in cols_cumul_sum + ['SPR', 'SPR+']: 
        if col in filtered_df.columns:
            filtered_df[col] = pd.to_numeric(filtered_df[col], errors='coerce')

st.dataframe(filtered_df)

# --- 🎯 SÉLECTEUR DE DATE ET DÉFINITION DE LA CIBLE 🎯 ---

# 1. Définir la date du jour comme valeur par défaut
date_defaut = datetime.now().date()

# On assure que la colonne est de type datetime pour le .max()
if 'Date' in filtered_df.columns:
     # S'assurer que le format est bien datetime (déjà fait dans load_and_merge_rpe, mais sécurisé ici)
     filtered_df['Date'] = pd.to_datetime(filtered_df['Date'], errors='coerce') 
     filtered_df.dropna(subset=['Date'], inplace=True)


# --- 🎯 SÉLECTEUR DE DATE ET DÉFINITION DE LA CIBLE 🎯 ---

# 2. Trouver la date la plus récente dans les données pour l'adapter (Maintenant que c'est un type datetime)
if not filtered_df.empty:
    # La colonne est maintenant de type datetime, on peut utiliser .date() sans erreur
    date_max = filtered_df['Date'].max().date() 
    
    # Si la date max est passée, utiliser la date max comme défaut
    # Note: On compare l'objet date_max (date) à date_defaut (date)
    if date_max > date_defaut: 
         date_defaut = date_max

date_selectionnee = st.date_input(
    "Sélectionnez la **date d'analyse** du microcycle :",
    value=date_defaut
    # Si vous voulez analyser le futur, retirez max_value=datetime.now().date()
)

# 3. Convertir la date sélectionnée en format datetime pour la comparaison
date_analyse_cible = pd.to_datetime(date_selectionnee)

# ----------------------------------------------------------------------
# 🚨 MODIFICATION 5 : APPEL DE LA FONCTION DE GROUPE AVEC LE FICHIER ID
# ----------------------------------------------------------------------
st.subheader("--- Débogage des prescriptions (Les avertissements ci-dessus indiquent les échecs) ---")
df_prescriptions_groupe, jour_reference_calc, jour_a_ajuster_calc = calculer_prescription_groupe_auto(
    df_rpe_complet=filtered_df, # Votre DataFrame GPS/RPE complet
    DF_REFERENCES_MOYENNES=DF_REFERENCES_MOYENNES, # Votre dictionnaire de référence
    cols_cumul_sum=cols_cumul_sum, # Votre liste de colonnes métriques mises à jour
    date_analyse_cible=date_analyse_cible,
    df_id_joueuses=df_id_joueuses # 👈 NOUVEAU PARAMÈTRE CRITIQUE
)

# ----------------------------------------------------------------------
# 🚨 MODIFICATION 6 : AFFICHAGE FINAL (Format Catapult)
# ----------------------------------------------------------------------
if not df_prescriptions_groupe.empty:
    st.header("📋 Prescription de Groupe : Séance Corrigée")
    st.subheader(f"Charge Brute Corrigée Recommandée pour **{jour_a_ajuster_calc}** (Évaluation basée sur **{jour_reference_calc}**)")
    
    # L'affichage utilise le DataFrame formaté Catapult qui a déjà les bonnes colonnes
    st.dataframe(df_prescriptions_groupe.style
            .format(precision=2),
        use_container_width=True, # Important pour remplir la colonne
        hide_index=True
    )

    df_prescriptions_seance = df_prescriptions_groupe.drop(['Athlete ID', 'Position'], axis=1)
    
    cols_to_avg = [col for col in df_prescriptions_seance.columns 
        if col not in ['Athlete Name']]
    
    df_prescriptions_seance = df_prescriptions_seance[cols_to_avg].mean().to_frame().T
    st.dataframe(df_prescriptions_seance.style
            .format(precision=2),
        use_container_width=True, # Important pour remplir la colonne
        hide_index=True
    )

else:
    st.error("❌ Échec de la prescription : Aucune joueuse n'a pu être calculée. Veuillez vérifier les avertissements ci-dessus.")

#st.dataframe(df_meilleur_match)

def init_connection():
    """Initialise la connexion à Firestore en utilisant les Secrets de Streamlit."""
    try:
        # On récupère les secrets au format dictionnaire
        key_dict = dict(st.secrets["firestore"])
        
        # On crée les credentials
        creds = service_account.Credentials.from_service_account_info(key_dict)
        
        # On initialise le client Firestore
        return firestore.Client(credentials=creds, project=key_dict['project_id'])
    except Exception as e:
        st.error(f"Erreur d'initialisation : {e}")
        return None

st.title("🚀 Test de connexion Firestore")

db = init_connection()

if db:
    st.success("Connexion réussie aux Secrets et à Google Cloud !")
    
    # Test de lecture/écriture simple
    st.subheader("Vérification des données")
    
    # Bouton pour tester l'écriture
    if st.button("Écrire un document de test"):
        doc_ref = db.collection("test_collection").document("test_doc")
        doc_ref.set({
            "status": "Connecté !",
            "message": "Bravo, tes secrets fonctionnent parfaitement.",
            "timestamp": firestore.SERVER_TIMESTAMP
        })
        st.info("Document 'test_doc' créé dans la collection 'test_collection'.")

    # Affichage des documents existants (si disponibles)
    try:
        docs = db.collection("test_collection").stream()
        st.write("Documents trouvés dans 'test_collection' :")
        for doc in docs:
            st.json(doc.to_dict())
    except Exception as e:
        st.warning(f"Impossible de lire la collection (elle est peut-être vide) : {e}")
else:
    st.error("La connexion a échoué. Vérifie le format de ta 'private_key' dans les Secrets.")
    st.info("Astuce : La clé doit commencer par `-----BEGIN PRIVATE KEY-----` et finir par `-----END PRIVATE KEY-----\\n`.")