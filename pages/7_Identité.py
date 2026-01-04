import streamlit as st
import pandas as pd
import os
import numpy as np
from sections.menu.menu import custom_sidebar_menu
from datetime import date 
import base64 # Import nécessaire pour la gestion des images
import io 

# --- CONFIGURATION ET SETUP ---
# Les variables Firebase globales (non utilisées dans ce fichier mais gardées pour référence)
app_id = 'default-app-id'
firebase_config = {}

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

# Le fichier sera stocké dans un sous-dossier 'data' et nommé 'identite.csv'
FICHIER_ID_JOUEUSES = 'data/identite.csv' 
DOSSIER_DATA = 'data'

st.title("👥 Identité et Profils de l'Équipe")
st.markdown("---")

# ----------------------------------------------------------------------
# 1. FONCTION DE CHARGEMENT OU CRÉATION
# ----------------------------------------------------------------------

@st.cache_data(show_spinner="Chargement ou création du tableau d'identité...")
def charger_ou_creer_donnees_joueuses():
    """Charge les données du CSV ou crée un DataFrame vide si le fichier n'existe pas."""
    
    colonnes_base = [ 
        'Prénom', 
        'NOM',
        'N°',
        'Sexe',
        'Date de naissance',
        'Age',
        'Latéralité',
        '1er Poste', 
        '2nd Poste',
        'Statut',
        'Photo URL'  
    ]
    
    if os.path.exists(FICHIER_ID_JOUEUSES):
        try:
            dtype_force = {
                'Prénom': str,
                'NOM': str,
                '1er Poste': str,
                '2nd Poste': str,
                'Photo URL': str
            }
            
            
            df = pd.read_csv(FICHIER_ID_JOUEUSES, encoding='utf-8', dtype=dtype_force)
            st.success(f"Fichier '{FICHIER_ID_JOUEUSES}' chargé avec succès.")
            
            # S'assurer que le DataFrame chargé contient les colonnes de base
            for col in colonnes_base:
                if col not in df.columns:
                    df[col] = pd.NA
                    
        except Exception as e:
            st.error(f"Erreur lors du chargement de {FICHIER_ID_JOUEUSES}. Création d'un tableau vierge. Erreur: {e}")
            df = pd.DataFrame(columns=colonnes_base)
            
    else:
        st.info(f"Fichier '{FICHIER_ID_JOUEUSES}' non trouvé. Création d'un tableau d'identité vierge.")
        df = pd.DataFrame(columns=colonnes_base)

    # --- 🚨 ÉTAPE DE CONVERSION CRUCIALE 🚨 ---
    if 'Date de naissance' in df.columns:
        # La fonction to_datetime convertit les chaînes de caractères en objets date.
        # errors='coerce' remplace les valeurs non-date (y compris les NaN/None) par NaT (Not a Time),
        # ce qui est le format attendu pour les lignes vides ou nouvelles.
        df['Date de naissance'] = pd.to_datetime(
            df['Date de naissance'], 
            format='mixed', # Utiliser 'mixed' pour deviner le format ou spécifier 'DD/MM/YYYY' si vous avez un format strict
            errors='coerce'
        )

    # --- 💡 CALCUL AUTOMATIQUE DE L'ÂGE 💡 ---
    if 'Date de naissance' in df.columns and 'Age' in df.columns:
        # 1. Définir la date de référence (aujourd'hui)
        TODAY = pd.to_datetime(date.today())        
        age = (TODAY - df['Date de naissance']).dt.days / 365.25        
        df['Age'] = age.apply(lambda x: round(x, 1) if pd.notna(x) else np.nan)

    # Nettoyage pour les sélections
    if '1er Poste' in df.columns:
        df['1er Poste'] = df['1er Poste'].fillna('À définir')

    # 💡 AJOUT pour remplir les valeurs manquantes avec une chaîne vide si ce sont des NaT
    # Cela garantit que la SelectboxColumn trouve une chaîne et non un NaN/None
    if '2nd Poste' in df.columns:
        df['2nd Poste'] = df['2nd Poste'].fillna('')

    if 'Photo URL' in df.columns:
        df['Photo URL'] = df['Photo URL'].fillna('')
    
    if 'Age' in df.columns:
        # 'coerce' va mettre NaN si la valeur n'est pas un nombre (comme 'Attaquant')
        df['Age'] = pd.to_numeric(df['Age'], errors='coerce') 

    return df

# ----------------------------------------------------------------------
# 2. FONCTION DE SAUVEGARDE (CORRIGÉE)
# ----------------------------------------------------------------------
def sauvegarder_donnees_joueuses(df_modifie):
    """Sauvegarde le DataFrame modifié dans le nouveau fichier CSV."""
    
    # Créer le dossier 'data' s'il n'existe pas
    if not os.path.exists(DOSSIER_DATA):
        os.makedirs(DOSSIER_DATA)
        st.info(f"Dossier '{DOSSIER_DATA}' créé.")
    
    try:
        # 🚨 CORRECTION POUR LE 2nd POSTE 🚨
        # Remplacer toutes les valeurs manquantes (NaN) dans la colonne '2nd Poste'
        # par une chaîne vide ('') avant de sauvegarder.
        if '2nd Poste' in df_modifie.columns:
            df_modifie['2nd Poste'] = df_modifie['2nd Poste'].fillna('')
            
        # Nettoyage des lignes sans Prénom/NOM
        df_modifie_clean = df_modifie.dropna(subset=['Prénom', 'NOM'], how='all')
        
        df_modifie_clean.to_csv(FICHIER_ID_JOUEUSES, index=False, encoding='utf-8')
        st.session_state['data_saved_success'] = True
    except Exception as e:
        st.session_state['data_saved_error'] = f"Erreur lors de la sauvegarde : {e}"


# ----------------------------------------------------------------------
# CHARGEMENT DES DONNÉES
# ----------------------------------------------------------------------
df_identite_initial = charger_ou_creer_donnees_joueuses()

# Initialiser le DataFrame en session state pour la modification de l'URL de la photo
if 'df_identite_current' not in st.session_state:
    st.session_state['df_identite_current'] = df_identite_initial.copy()

# ----------------------------------------------------------------------
# 3. ÉDITEUR DE DONNÉES (CORRIGÉ)
# ----------------------------------------------------------------------

st.subheader("Modifier ou Ajouter les Profils des Joueuses")

# --- Définir les configurations de colonne ---
postes_connus = ['Attaquant', 'Sentinelle', 'Milieu', 'Milieu off', 'Milieu def', 'Défenseur Central', 'Piston', 'Latéral', 'Gardien']
lateralite_options = ['Droit', 'Gauche', 'Ambidextre']
statut_options = ['Titulaire', 'Remplaçant', 'En devenir']

config_colonnes = {
    'Prénom': st.column_config.TextColumn("Prénom", required=True),
    'NOM': st.column_config.TextColumn("NOM", required=True),
    'N°': st.column_config.NumberColumn("N° Maillot", format="%d", min_value=1, max_value=99),
    'Sexe': st.column_config.SelectboxColumn("Sexe", options=['F', 'H']),
    'Date de naissance': st.column_config.DateColumn("Date de naissance", format="DD/MM/YYYY"),
    'Latéralité': st.column_config.SelectboxColumn("Latéralité", options=lateralite_options),
    '1er Poste': st.column_config.SelectboxColumn("1er Poste", options=postes_connus, required=True),
    '2nd Poste': st.column_config.SelectboxColumn("2nd Poste", options=postes_connus, required=False),
    'Statut': st.column_config.SelectboxColumn("Statut", options=statut_options),
    'Photo URL': st.column_config.ImageColumn("Photo", help="Image du profil de la joueuse", width="small"), # L'URL sera gérée par la section ci-dessous

    # 'Age' est généralement calculé, pas saisi.
    'Age': st.column_config.NumberColumn("Age", disabled=True, format="%.1f"), 
}

# 🚨 ARGUMENTS CLÉS POUR L'ÉDITION 🚨
df_identite_modifie = st.data_editor(
    df_identite_initial,
    column_config=config_colonnes,
    hide_index=True,
    num_rows="dynamic", # <-- PERMET L'AJOUT DE LIGNES
    key="editor_identite"
)

# Mettre à jour le DataFrame du state après l'édition
st.session_state['df_identite_current'] = df_identite_modifie.copy()

# ----------------------------------------------------------------------
# 4. SAUVEGARDE ET FEEDBACK
# ----------------------------------------------------------------------
df_identite_modifie_clean = df_identite_modifie.dropna(subset=['Prénom', 'NOM'], how='all')

# Comparer la version modifiée (nettoyée) avec la version initiale (nettoyée)
initial_clean = df_identite_initial.dropna(subset=['Prénom', 'NOM'], how='all')
is_data_changed = (df_identite_modifie_clean.shape[0] != initial_clean.shape[0]) or (not df_identite_modifie_clean.equals(initial_clean))


if is_data_changed:
    st.markdown("---")
    # Vérifier si toutes les lignes ont au moins Prénom et NOM
    # Utiliser le DataFrame avant le nettoyage pour vérifier les lignes qui ont des NaN
    incomplete_rows = df_identite_modifie[df_identite_modifie['Prénom'].isna() | df_identite_modifie['NOM'].isna()]
    lignes_incompletes = incomplete_rows.shape[0]
  
    if lignes_incompletes > 0:
        st.error(f"❌ {lignes_incompletes} ligne(s) incomplète(s) détectée(s). Les champs Nom ou Prénom sont obligatoires.")
    else:
        st.warning(f"⚠️ Modifications détectées. {df_identite_modifie_clean.shape[0]} ligne(s) à sauvegarder.")
    
        if st.button("💾 SAUVEGARDER LE TABLEAU D'IDENTITÉ", key="btn_save_table"):
            sauvegarder_donnees_joueuses(df_identite_modifie)
else:
    st.markdown("---")
    st.info("Aucune modification en attente.")

# ======================================================================
# 5. GESTION DES PHOTOS DE PROFIL
# ======================================================================

st.markdown("---")
st.header("📸 Gestion des Photos de Profil")
st.markdown("Associez une photo à une joueuse. L'image sera encodée et sauvegardée dans le fichier CSV local. Vous devriez maintenant voir l'image directement dans le tableau ci-dessus après l'association.")


# --- FONCTIONS UTILES ---

def get_base64_image_url(uploaded_file):
    """Crée une URL de données temporaire à partir d'un fichier uploadé."""
    try:
        file_bytes = uploaded_file.getvalue()
        base64_data = base64.b64encode(file_bytes).decode('utf-8')
        mime_type = uploaded_file.type
        return f"data:{mime_type};base64,{base64_data}"
    except Exception as e:
        st.error(f"Erreur lors de l'encodage de l'image: {e}")
        return None


def sauvegarder_photo_joueuse(player_name, uploaded_file):
    """
    Met à jour l'URL de la photo dans le DataFrame en session state et force le ré-affichage.
    """
    
    photo_url_base64 = get_base64_image_url(uploaded_file)
    
    if photo_url_base64 is None:
        return

    # 1. Obtenir le DataFrame depuis l'état de la session (la source de vérité)
    df = st.session_state['df_identite_current'].copy() 
    
    # Trouver l'index de la joueuse par son nom/prénom
    player_id = df['Prénom'] + ' ' + df['NOM']
    index = df[player_id == player_name].index
    
    if not index.empty:
        # 2. Mettre à jour la colonne 'Photo URL'
        df.loc[index[0], 'Photo URL'] = photo_url_base64
        
        # 3. Ré-assigner la nouvelle version du DataFrame à l'état de la session
        st.session_state['df_identite_current'] = df
        
        st.success(f"L'image de **{player_name}** a été associée au profil. Le tableau ci-dessus est mis à jour. Veuillez cliquer sur **SAUVEGARDER LE TABLEAU D'IDENTITÉ** pour la rendre permanente !")
        
        # 4. Forcer Streamlit à ré-exécuter le script, ce qui redessinera l'éditeur avec le nouvel état.
        st.rerun() 
    else:
        st.error(f"Joueuse **{player_name}** introuvable dans le tableau.")


# --- UI GESTION PHOTO ---

# On utilise la version du state pour déterminer les joueuses valides
df_valid_players = st.session_state['df_identite_current'].dropna(subset=['Prénom', 'NOM'])
player_list = (df_valid_players['Prénom'] + ' ' + df_valid_players['NOM']).tolist()

selected_player_photo = st.selectbox(
    "Sélectionnez la joueuse à mettre à jour:", 
    options=[""] + player_list, 
    key='selected_player_photo'
)

if selected_player_photo:
    # Récupérer l'URL de la joueuse sélectionnée dans le DataFrame du state
    current_row = df_valid_players[(df_valid_players['Prénom'] + ' ' + df_valid_players['NOM']) == selected_player_photo]
    
    if not current_row.empty:
        current_photo_url = current_row['Photo URL'].iloc[0]
        
        col_view, col_upload = st.columns([1, 2])

        with col_view:
            st.subheader("Photo Actuelle")
            if current_photo_url:
                # Affichage de l'image Base64
                st.image(current_photo_url, caption=f"Photo de {selected_player_photo}", width=150)
            else:
                st.warning("Aucune photo enregistrée.")

        with col_upload:
            st.subheader(f"Télécharger une nouvelle photo pour {selected_player_photo}")
            
            uploaded_file = st.file_uploader(
                "Choisir une image (JPG, PNG, max 1MB)", 
                type=['png', 'jpg', 'jpeg'], 
                key='player_photo_uploader'
            )

            if uploaded_file is not None:
                st.image(uploaded_file, caption="Nouvelle photo sélectionnée", width=150)
                
                if st.button(f"Associer et Enregistrer l'URL pour {selected_player_photo}", type="primary", key="btn_associate_photo"):
                    # L'appel à cette fonction met à jour l'état et force le rerun,
                    # ce qui garantit la mise à jour du tableau ci-dessus.
                    sauvegarder_photo_joueuse(selected_player_photo, uploaded_file)