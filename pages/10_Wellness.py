import streamlit as st
import pandas as pd
from sections.menu.menu import custom_sidebar_menu
from datetime import date
from firebase_admin import firestore
import firebase_admin
import os # Nécessaire pour gérer les chemins de fichiers

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

# Les variables globales __app_id et __firebase_config sont nécessaires
try:
    APP_ID = st.secrets["__app_id"]
    # Tente d'initialiser Firebase si ce n'est pas déjà fait
    if not firebase_admin._apps:
        firebase_config = st.secrets["firebase_config"]
        cred = firebase_admin.credentials.Certificate(firebase_config)
        firebase_admin.initialize_app(cred, name=APP_ID)
    
    db = firestore.client(APP_ID)
    
    # Collection de stockage : nous utilisons 'public_user' comme ID utilisateur générique
    # car le formulaire est public et non authentifié.
    USER_ID_PUBLIC = 'public_user' 
    WELLNESS_COLLECTION_PATH = f"artifacts/{APP_ID}/users/{USER_ID_PUBLIC}/wellness_entries"

except Exception as e:
    st.error(f"Erreur de configuration Firebase. Vérifiez la disponibilité des 'st.secrets'.")
    st.stop()


# --- Fonction de chargement des Joueuses ---
@st.cache_data
def load_players_from_csv():
    """Charge la liste des noms complets des joueuses depuis identite.csv."""
    try:
        # Assurez-vous que le chemin vers identite.csv est correct
        # Dans Streamlit, il est souvent préférable d'utiliser le chemin relatif ou absolu
        if os.path.exists("data/identite.csv"):
            df = pd.read_csv("data/identite.csv")
        elif os.path.exists("./identite.csv"):
            df = pd.read_csv("./identite.csv")
        else:
            st.error("Fichier 'identite.csv' non trouvé. Veuillez vous assurer qu'il est au même niveau que votre script Streamlit.")
            return []
            
        # Créer la colonne Nom Complet (Prénom Nom)
        if 'Prénom' in df.columns and 'Nom' in df.columns:
            df['Nom Complet'] = df['Prénom'] + ' ' + df['Nom']
            # Retourne la liste des noms, triée pour plus de clarté
            return sorted(df['Nom Complet'].unique().tolist())
        else:
            st.error("Le fichier 'identite.csv' doit contenir les colonnes 'Prénom' et 'Nom'.")
            return []
            
    except Exception as e:
        st.error(f"Erreur lors du chargement ou du traitement de 'identite.csv': {e}")
        return []


# --- Fonction d'enregistrement dans Firestore ---
def save_wellness_entry(data):
    """Enregistre l'entrée de wellness dans Firestore."""
    
    if not data['Joueur ID'] or not data['Date']:
        return False

    # Création du document ID
    doc_id = f"{data['Joueur ID'].replace(' ', '_')}_{data['Date']}"
    doc_ref = db.collection(WELLNESS_COLLECTION_PATH).document(doc_id)
    
    try:
        doc_ref.set(data) 
        return True
    except Exception as e:
        st.error(f"Erreur lors de l'enregistrement dans Firestore: {e}")
        return False


# ----------------------------------------------------------------------
# Interface Streamlit (Formulaire Joueuse)
# ----------------------------------------------------------------------

st.title("Questionnaire Wellness Quotidien")
st.markdown(
    """
    Veuillez sélectionner votre nom et remplir votre état de wellness pour aujourd'hui.
    Les scores sont sur une échelle de **1 (Très Bien/Faible)** à **5 (Très Mal/Fort)**.
    """
)
st.markdown("---")

# Charger la liste des joueuses
joueuses_disponibles = load_players_from_csv()
joueuses_options = ["-- Choisir son nom --"] + joueuses_disponibles


# --- FORMULAIRE ---
with st.form(key='wellness_joueuse_form'):
    
    # 1. Menu Déroulant de sélection de la joueuse (OBLIGATOIRE)
    joueuse_selectionnee = st.selectbox(
        "Sélectionnez votre nom :",
        options=joueuses_options,
        index=0,
        key='joueur_id_select'
    )
    
    st.info(f"Date de la saisie : **{date.today().strftime('%A %d %B %Y')}**")

    st.markdown("---")
    st.subheader("Évaluations (1 = Très Bien/Faible, 5 = Très Mal/Fort)")

    # 2. Métriques de Wellness (Sliders)
    col_sliders1, col_sliders2 = st.columns(2)
    
    with col_sliders1:
        sommeil = st.slider(
            "1. Qualité du Sommeil", 
            min_value=1, max_value=5, value=3, step=1, key='sommeil_j'
        )
        fatigue = st.slider(
            "2. Niveau de Fatigue", 
            min_value=1, max_value=5, value=3, step=1, key='fatigue_j'
        )
        douleur = st.slider(
            "3. Niveau de Douleur/Courbatures", 
            min_value=1, max_value=5, value=1, step=1, key='douleur_j'
        )
        
    with col_sliders2:
        stress = st.slider(
            "4. Niveau de Stress", 
            min_value=1, max_value=5, value=3, step=1, key='stress_j'
        )
        humeur = st.slider(
            "5. Humeur Générale", 
            min_value=1, max_value=5, value=3, step=1, key='humeur_j'
        )
        st.empty() 

    # 3. Commentaires
    commentaires = st.text_area(
        "Commentaires / Notes (entraînement, blessure, sommeil spécial...) :", 
        key='commentaires_j',
        height=100
    )
    
    st.markdown("---")
    
    submitted = st.form_submit_button("✅ Envoyer ma Saisie Wellness")


# ----------------------------------------------------------------------
# Logique de Soumission
# ----------------------------------------------------------------------
if submitted:
    if joueuse_selectionnee == "-- Choisir son nom --":
        st.warning("⚠️ Veuillez sélectionner votre nom dans le menu déroulant.")
    else:
        # Préparation des données pour Firestore
        wellness_data = {
            'Joueur ID': joueuse_selectionnee, 
            'Date': date.today().strftime("%Y-%m-%d"), 
            'Sommeil': sommeil,
            'Fatigue': fatigue,
            'Douleur': douleur,
            'Stress': stress,
            'Humeur': humeur,
            'Commentaires': commentaires if commentaires else "RAS",
            'Timestamp Saisie': firestore.SERVER_TIMESTAMP,
            'UserID Saisie': USER_ID_PUBLIC 
        }
        
        # Enregistrement
        if save_wellness_entry(wellness_data):
            # Affichage du prénom uniquement pour un message plus personnel
            prenom = joueuse_selectionnee.split(' ')[0]
            st.success(f"Merci {prenom}, votre entrée a été enregistrée pour aujourd'hui. Vous pouvez fermer cette page.")
            st.balloons()
            st.stop()
        else:
            st.error("Un problème est survenu lors de l'enregistrement. Veuillez réessayer.")
