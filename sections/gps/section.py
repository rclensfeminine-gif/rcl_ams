import streamlit as st
import pandas as pd
from pathlib import Path
from sections.gps.pipeline import recuperer_all_files_gps
from sections.constantes import ONGLET_GPS_SAISON

df_gps = recuperer_all_files_gps()

def gps_files_upload() -> list:
    gps_files = st.sidebar.file_uploader(
        "Choisir un fichier CSV séance",
        accept_multiple_files=True,
        type=['.csv']
    )
    return gps_files

def gps_files_upload_match() -> list:
    gps_files = st.sidebar.file_uploader(
        "Choisir un fichier CSV match",
        accept_multiple_files=True,
        type=['.csv']
    )
    return gps_files

# --- Constantes de Session State ---
# Drapeau pour signaler un rerun post-sauvegarde
RERUN_KEY = 'gps_uploader_rerun_in_progress'
# Nom du fichier à écraser (pour le prochain rerun)
OVERWRITE_FILE_KEY = 'gps_overwrite_file_name' 
# Ensemble des noms de fichiers que l'utilisateur a choisi d'ignorer (MÉMOIRE PERSISTANTE)
IGNORE_FILES_KEY = 'gps_ignore_files_set'

def gps_files_uploader(type_de_donnees: str):
    """
    Gère le téléchargement des fichiers GPS (séances ou matchs) et leur sauvegarde locale.
    
    Permet à l'utilisateur d'écraser ou d'ignorer les fichiers existants.
    
    :param type_de_donnees: 'séance' ou 'match'.
    """
    if type_de_donnees not in ['séance', 'match']:
        st.sidebar.error("Type de données non supporté. Utiliser 'séance' ou 'match'.")
        return

    # Initialiser les clés de session state si elles n'existent pas
    if RERUN_KEY not in st.session_state:
        st.session_state[RERUN_KEY] = False
    if OVERWRITE_FILE_KEY not in st.session_state:
        st.session_state[OVERWRITE_FILE_KEY] = None
    if IGNORE_FILES_KEY not in st.session_state:
        # L'ensemble (set) est idéal pour stocker les noms de fichiers à ignorer
        st.session_state[IGNORE_FILES_KEY] = set() 

    # Récupérer l'état du rerun et réinitialiser le drapeau pour ce run
    is_post_save_rerun = st.session_state[RERUN_KEY]
    st.session_state[RERUN_KEY] = False

    # Définir et créer le répertoire de sauvegarde
    save_directory = Path('data') / type_de_donnees
    save_directory.mkdir(parents=True, exist_ok=True) 

    label = f"Choisir un fichier CSV {type_de_donnees.capitalize()}"
    key = f'file_uploader_{type_de_donnees}'
    
    uploaded_files = st.sidebar.file_uploader(
        label,
        accept_multiple_files=True,
        type=['csv'],
        key=key
    )

    if uploaded_files:
        nouveaux_fichiers_sauvegardes = 0
        
        # Filtrer les fichiers qui ont été marqués pour être ignorés lors des runs précédents
        files_to_process = [
            f for f in uploaded_files 
            if f.name not in st.session_state[IGNORE_FILES_KEY]
        ]
        
        
        # --- Boucle principale de traitement des fichiers ---
        for uploaded_file in files_to_process:
            file_name = uploaded_file.name
            file_path = save_directory / file_name
            
            # 1. Vérification de l'existence du fichier sur le disque
            if file_path.exists():
                
                # 2. Vérifier si l'utilisateur a cliqué sur "Remplacer" (stocké via OVERWRITE_FILE_KEY)
                overwrite_requested = (st.session_state[OVERWRITE_FILE_KEY] == file_name)
                
                if overwrite_requested:
                    # CAS A: Écrasement demandé explicitement.
                    st.toast(f"Fichier '{file_name}' remplacé.", icon='✅')
                    st.session_state[OVERWRITE_FILE_KEY] = None # On nettoie la requête
                    # S'il était dans le set d'ignorance, on le retire
                    if file_name in st.session_state[IGNORE_FILES_KEY]:
                         st.session_state[IGNORE_FILES_KEY].remove(file_name)
                    # La boucle continue pour passer à la sauvegarde
                    
                elif is_post_save_rerun:
                    # CAS B: Rerun automatique post-sauvegarde. On ignore silencieusement.
                    continue
                else:
                    # CAS C: Fichier existant, attente de l'action utilisateur. Afficher l'alerte et les boutons.
                    st.sidebar.warning(f"Le fichier '{file_name}' existe déjà.")
                    
                    # --- Fonctions de callback ---
                    def handle_overwrite_click(file_to_overwrite):
                        st.session_state[OVERWRITE_FILE_KEY] = file_to_overwrite
                        # st.rerun() # L'avertissement vient d'ici, on le retire. Le run suivant sera déclenché par une autre interaction OU le script arrivera à sa fin si c'est le dernier élément.
                        
                    def handle_ignore_click(file_to_ignore):
                        st.session_state[IGNORE_FILES_KEY].add(file_to_ignore)
                        st.session_state[OVERWRITE_FILE_KEY] = None 
                        # st.rerun() # L'avertissement vient d'ici, on le retire.
                    
                    # Affichage des boutons dans la barre latérale
                    col1, col2 = st.sidebar.columns(2)
                    
                    # Ajout d'une clé de session state temporaire pour forcer le rerun APRES le callback
                    # C'est la meilleure technique pour éviter l'avertissement tout en forçant le rerun.
                    temp_rerun_key = f'trigger_rerun_for_{file_name}'
                    if temp_rerun_key not in st.session_state:
                        st.session_state[temp_rerun_key] = False

                    def handle_overwrite_click_and_rerun(file_to_overwrite):
                        st.session_state[OVERWRITE_FILE_KEY] = file_to_overwrite
                        st.session_state[temp_rerun_key] = True 
                        # PAS de st.rerun() ici

                    def handle_ignore_click_and_rerun(file_to_ignore):
                        st.session_state[IGNORE_FILES_KEY].add(file_to_ignore)
                        st.session_state[OVERWRITE_FILE_KEY] = None 
                        st.session_state[temp_rerun_key] = True 
                        # PAS de st.rerun() ici


                    with col1:
                         st.button(
                            "Remplacer",
                            on_click=handle_overwrite_click_and_rerun,
                            args=(file_name,),
                            key=f'overwrite_btn_{file_name}',
                            use_container_width=True
                        )
                    with col2:
                         st.button(
                            "Ignorer",
                            on_click=handle_ignore_click_and_rerun,
                            args=(file_name,),
                            key=f'ignore_btn_{file_name}',
                            use_container_width=True
                        )
                    
                    # Arrêter le traitement de ce fichier en attente de l'action
                    # Et vérifier si un rerun est en attente
                    if st.session_state[temp_rerun_key]:
                        st.session_state[temp_rerun_key] = False # Réinitialisation
                        st.rerun() # Le rerun est appelé EN DEHORS du callback

                    continue 

            # --- SAUVEGARDE PHYSIQUE DU FICHIER ---
            # Exécutée si le fichier est nouveau OU si l'écrasement a été demandé.
            try:
                uploaded_file.seek(0)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                    
                nouveaux_fichiers_sauvegardes += 1
                
            except Exception as e:
                st.sidebar.error(f"Erreur critique lors de la sauvegarde de {file_name}: {e}")
                continue
                
        # --- POST-TRAITEMENT ET RERUN ---
        
        # Déclencher le rerun UNIQUEMENT s'il y a eu des sauvegardes effectives
        if nouveaux_fichiers_sauvegardes > 0:
            # 1. Définir le drapeau de session state pour le prochain run (le rerun)
            st.session_state[RERUN_KEY] = True 
            
            # 2. Afficher le succès général
            st.sidebar.success(f"{nouveaux_fichiers_sauvegardes} fichier(s) sauvegardé(s) dans {save_directory.name}/")
            
            # 3. Suppression des clés de Session State (cache) pour forcer le rechargement des données
            if type_de_donnees == 'séance':
                if 'all_gps_session' in st.session_state:
                     del st.session_state['all_gps_session']
                if 'DF_REFERENCES_CUMULEES' in st.session_state:
                     del st.session_state['DF_REFERENCES_CUMULEES']
                         
            # 4. Déclencher le rerun
            st.rerun()

#Fonction séance
def onglet_seance(df: pd.DataFrame) -> pd.DataFrame:
    dates = sorted(df['Date'].unique(), reverse=True)
    selected_date = st.selectbox('Date de la séance', options=dates)
    df_filtered = df[df['Date'] == selected_date]
    st.dataframe(df_filtered, hide_index=True)

#Fonction joueuse
def onglet_joueuse(df: pd.DataFrame) -> pd.DataFrame:
    joueuses = sorted(pd.Series(df['Name'].unique()).dropna(), reverse=True)
    # 2. Créez des onglets dynamiques pour chaque joueuse
    tabs = st.tabs(joueuses)
    for i, joueuse_name in enumerate(joueuses):
        with tabs[i]:
            st.header(f"Données individuelles pour {joueuse_name}")
            # 3. Filtrer le DataFrame pour la joueuse sélectionnée
            df_joueuse = df[df['Name'] == joueuse_name]
            # 4. Afficher le tableau de données
            st.dataframe(df_joueuse, hide_index=True)