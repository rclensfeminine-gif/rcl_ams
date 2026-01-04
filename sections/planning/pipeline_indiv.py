import streamlit as st
import pandas as pd
import os
import json
import uuid
from datetime import datetime, timedelta
from sections.planning.pipeline import get_match_color

# --- Configuration des chemins ---
PLANNING_DIR = "data/planning"
WEEK_NAMES_FILE = os.path.join(PLANNING_DIR, "week_names.json")
FILE_PATH_PLANNING = "data/planning/"
FILE_PATH_PLANNING_INDIV = "data/planning_indiv/"
FILE_PATH_SEMAINE = "data/planning/week_names.json" 
FILE_PATH_IDENTITE = "data/identite.csv"

# --- CONSTANTES ---
JOURS_SEMAINE = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
PERIODES_JOURNEE_INDIV = ["Matin (8h-12h)", "Midi (12h-14h)", "Après-midi (14h-20h)"] 
PERIODES_KEYS = [p.split('(')[0].strip() for p in PERIODES_JOURNEE_INDIV] 
PERIODES_JOURNEE_SHORT = [p.split('(')[0].strip() for p in PERIODES_JOURNEE_INDIV] 
TYPES_RDV = ["Soin", "Récup", "RDV Doc", "RDV Extérieur", "Séance Réa", "Séance Supplémentaire", "Muscu", "Vidéo", "Testing", "Diététicien", "Entretien", "Prépa Mental", "Podologue", "OP Joueuse", "Autre"]

DEFAULT_APPOINTMENTS_DATA = pd.DataFrame(
    columns=[
        "Date",
        "Heure Début",
        "Heure Fin",
        "Joueur",
        "Activité",
        "Lieu",
        "Commentaires",
        "ID"
    ]
)

# --- FONCTIONS DE CHARGEMENT DES DONNEES ---
@st.cache_data
def load_weekly_plannings():
    """Charge les plannings collectifs depuis les fichiers CSV."""
    plannings = {}
    if not os.path.exists(FILE_PATH_PLANNING):
        return plannings
    
    for filename in os.listdir(FILE_PATH_PLANNING):
        if filename.endswith(".csv"):
            date_key = filename.replace("planning_", "").replace(".csv", "")
            try:
                # La première colonne (Créneau Horaire) est l'index
                df = pd.read_csv(
                    os.path.join(FILE_PATH_PLANNING, filename), 
                    index_col=0, 
                    encoding='utf-8'
                )
                plannings[date_key] = df
            except Exception as e:
                st.error(f"Erreur de chargement du planning {filename}: {e}")
    return plannings

@st.cache_data
def load_player_data():
    """
    Charge les données des joueuses depuis le fichier CSV d'identité.
    
    Retourne un objet pandas DataFrame contenant les données.
    """
    # Vérifie si le fichier existe
    if os.path.exists(FILE_PATH_IDENTITE):
        try:
            # === CORRECTION ICI : Utilisation de pandas.read_csv pour les fichiers CSV ===
            # Détecte automatiquement l'encodage et les délimiteurs
            df_players = pd.read_csv(FILE_PATH_IDENTITE)
            
            # Pour s'assurer que 'N°' est bien un entier si nécessaire (même si .0 est courant en CSV)
            if 'N°' in df_players.columns:
                # Convertir en entier, gérer les NaN s'il y en a
                df_players['N°'] = df_players['N°'].fillna(0).astype(int) 
                
            return df_players
            
        except FileNotFoundError:
            # Ce bloc est techniquement redondant avec os.path.exists, mais bonne pratique
            st.error(f"Erreur : Le fichier CSV {FILE_PATH_IDENTITE} est introuvable.")
            return pd.DataFrame()
            
        except Exception as e:
            # Gestion des erreurs de lecture CSV (mauvais format, encodage, etc.)
            st.error(f"Erreur lors du chargement des données des joueuses depuis {FILE_PATH_IDENTITE} : {e}")
            return pd.DataFrame()
    else:
        st.warning(f"Avertissement : Le fichier d'identité des joueuses ({FILE_PATH_IDENTITE}) n'existe pas.")
        return pd.DataFrame() # Retourne un DataFrame vide si le fichier n'existe pas

def initialize_session_state():
    """Initialise l'état de session Streamlit pour les clés de formulaire et les données d'application."""
    
    # 1. Initialisation de la clé de formulaire (votre logique existante)
    if 'rdv_form_key' not in st.session_state:
        st.session_state.rdv_form_key = 0

    # 2. Correction de l'erreur : Initialisation des données de rendez-vous
    if "appointments_data" not in st.session_state:
        # Initialise avec une copie du DataFrame vide par défaut
        st.session_state.appointments_data = DEFAULT_APPOINTMENTS_DATA.copy()
        st.info("Initialisation de la structure des données de rendez-vous pour la session.")

    # 3. Initialisation de la date sélectionnée
    if "selected_individual_date" not in st.session_state:
        st.session_state.selected_individual_date = datetime.now().date()

    # 4. CORRECTION DE L'ERREUR : Initialisation de la joueuse sélectionnée
    if "selected_joueuse" not in st.session_state:
        # On initialise à None. La valeur sera définie plus tard avec le premier nom chargé.
        st.session_state.selected_joueuse = None

def get_indiv_filepath(date_key: str):
    """Retourne le chemin complet du fichier JSON des RDV individuels pour une semaine donnée."""
    return os.path.join(FILE_PATH_PLANNING_INDIV, f"rdv_indiv_{date_key}.json")

@st.cache_data(show_spinner=False)
def load_individual_appointments(date_key: str):
    """
    Charge les RDV persistants depuis un fichier JSON spécifique à la semaine.
    Utilise le cache Streamlit.
    """
    filepath = get_indiv_filepath(date_key)

    if not os.path.exists(filepath):
        # Initialisation si le fichier n'existe pas
        return []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # S'assurer que les données sont bien une liste
            return data if isinstance(data, list) else []
    except json.JSONDecodeError as e:
        st.error(f"Erreur de lecture/décodage JSON pour le fichier {filepath}: {e}")
        return []
    except Exception as e:
        st.error(f"Erreur de chargement du fichier {filepath}: {e}")
        return []

def save_appointment_persistent(new_rdv: dict, date_key: str) -> bool:
    """Sauvegarde un nouveau rendez-vous de manière persistante (FICHIER)."""
    filepath = get_indiv_filepath(date_key)
    os.makedirs(FILE_PATH_PLANNING_INDIV, exist_ok=True)
    
    # Chargement de la liste existante via la fonction cacheable
    all_appointments = load_individual_appointments(date_key)
    
    # 1. Ajout des données manquantes
    # NOTE: L'ID est idéalement généré dans le formulaire, mais on le sécurise ici si manquant
    if 'id' not in new_rdv:
         new_rdv["id"] = str(uuid.uuid4())
    new_rdv["created_at"] = datetime.now().isoformat()
    # Mettre à jour la clé 'Joueuse' pour la cohérence
    new_rdv['Joueuse'] = new_rdv.get('player_name', new_rdv.get('Joueuse', 'Inconnue'))
    
    all_appointments.append(new_rdv)
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            # S'assurer que ensure_ascii=False est utilisé pour l'UTF-8
            json.dump(all_appointments, f, indent=4, ensure_ascii=False)
            
        # 2. Invalidation spécifique : C'EST LA CLEF DE LA CORRECTION
        load_individual_appointments.clear() 
        
        # Le formulaire sera réinitialisé par st.rerun() dans la fonction appelante
        print(f"RDV sauvegardé pour {new_rdv['Joueuse']}")
        return True
        
    except Exception as e:
        # st.error(f"Erreur lors de l'écriture du fichier de rendez-vous : {e}")
        # On log l'erreur pour ne pas bloquer l'interface appelante
        print(f"Erreur lors de l'écriture du fichier de rendez-vous : {e}")
        return False






##### Chargement des semaines et du lundi en cours
def load_json_data(filepath, default_value=None):
    """Charge les données JSON à partir d'un fichier. Gère l'absence de fichier et les erreurs de décodage."""
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            # Si le fichier est corrompu ou vide, retourne la valeur par défaut
            return default_value if default_value is not None else {}
    return default_value if default_value is not None else {}

@st.cache_data
def get_week_names():
    """Charge les noms de semaines personnalisés depuis le fichier JSON."""
    return load_json_data(WEEK_NAMES_FILE)

def get_available_weeks():
    """
    Récupère les dates de début de semaine disponibles (clés dans week_names.json).
    Si week_names est vide, génère une liste par défaut pour la démo.
    """
    week_names = get_week_names()
    if week_names:
        # Trié par date pour un affichage cohérent
        return sorted(week_names.keys())
    
    # Génération d'un ensemble de dates de semaines par défaut pour la démo si le fichier n'est pas prêt
    today = datetime.now().date()
    start_of_current_week = today - timedelta(days=today.weekday()) # Lundi (0)
    
    weeks = []
    for i in range(-4, 10): # 4 semaines passées et 10 futures
        week_start = start_of_current_week + timedelta(weeks=i)
        weeks.append(week_start.strftime('%Y-%m-%d'))
    return sorted(weeks)

def get_current_week_monday_str():
    """Calcule la date du lundi de la semaine courante au format YYYY-MM-DD."""
    today = datetime.now().date()
    # today.weekday() retourne 0 pour Lundi, 6 pour Dimanche
    start_of_current_week = today - timedelta(days=today.weekday())
    return start_of_current_week.strftime('%Y-%m-%d')





#####################
##### Gestion des RDV
#####################
def get_appointments_df(appointments_list):
    """
    Convertit la liste des RDV en DataFrame et garantit un index unique
    et la présence des colonnes critiques.

    *** Mise à jour pour gérer 'player_name' si 'Joueuse' est absent ***
    """
    # Colonnes critiques attendues dans l'ordre final
    COLUMNS = ['id', 'Joueuse', 'Jour', 'Heure', 'Type', 'Détails']

    if not appointments_list:
        return pd.DataFrame(columns=COLUMNS)
    
    df = pd.DataFrame(appointments_list)
    
    # 1. Gérer la clé du nom de la joueuse (Joueuse vs player_name)
    if 'Joueuse' not in df.columns:
        if 'player_name' in df.columns:
            # Si le backend utilise 'player_name' (comme suggéré par l'erreur), on le renomme en 'Joueuse'
            df = df.rename(columns={'player_name': 'Joueuse'})
        else:
            # Si aucune clé n'est trouvée pour le nom, on ajoute la colonne 'Joueuse' vide
            df['Joueuse'] = None

    # 2. GARANTIE DE LA PRÉSENCE DES AUTRES COLONNES CRITIQUES:
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = None 
            
    # S'assurer que seules les colonnes pertinentes sont conservées et dans l'ordre attendu
    # (En excluant 'player_name' s'il n'a pas été renommé en 'Joueuse')
    df = df.reindex(columns=COLUMNS)
    
    return df.reset_index(drop=True)

def write_appointments_to_file(appointments_list: list, date_key: str) -> bool:
    """
    Fonction utilitaire pour écrire la liste complète des RDV dans le fichier JSON.
    Gère la persistance réelle sur le disque.
    """
    filepath = get_indiv_filepath(date_key)
    os.makedirs(FILE_PATH_PLANNING_INDIV, exist_ok=True)
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            # S'assurer que ensure_ascii=False est utilisé pour l'UTF-8
            json.dump(appointments_list, f, indent=4, ensure_ascii=False)
            
        # Mise à jour de l'état de session après l'écriture réussie (pour les opérations suivantes)
        st.session_state.rdv_data_store = appointments_list
        
        # Invalide le cache après l'écriture sur le disque
        load_individual_appointments.clear()
        return True
    except Exception as e:
        st.error(f"Erreur lors de l'écriture du fichier de rendez-vous {filepath}: {e}")
        print(f"Erreur lors de l'écriture du fichier de rendez-vous : {e}")
        return False

def save_appointment_persistent(new_rdv: dict, date_key: str) -> bool:
    """Sauvegarde un nouveau rendez-vous de manière persistante (FICHIER)."""
    
    # 1. Chargement de la liste existante via la fonction cacheable
    # NOTE: L'état de session est mis à jour dans write_appointments_to_file
    all_appointments = load_individual_appointments(date_key) 
    
    # 2. Ajout des données manquantes
    if 'id' not in new_rdv:
        new_rdv["id"] = str(uuid.uuid4())
    new_rdv["created_at"] = datetime.now().isoformat()
    new_rdv['Joueuse'] = new_rdv.get('player_name', new_rdv.get('Joueuse', 'Inconnue'))
    
    all_appointments.append(new_rdv)
    
    # 3. Écriture sur le disque (Fonction refactorisée)
    if write_appointments_to_file(all_appointments, date_key):
        print(f"RDV sauvegardé pour {new_rdv['Joueuse']} sur le disque.")
        return True
    else:
        return False

def update_appointment_persistent(updated_rdv: dict, date_key: str) -> bool:
    """
    Met à jour un RDV existant et persiste les changements sur le disque.
    """
    # 1. Obtenir la liste actuelle des RDV depuis la source de vérité (fichier via cache)
    all_appointments = load_individual_appointments(date_key) 
    rdv_id_to_update = str(updated_rdv.get('id')) 
    
    if not rdv_id_to_update:
        print("Erreur: ID de RDV manquant pour la mise à jour.")
        return False 
    
    found = False
    new_appointments_list = []
    
    for rdv in all_appointments:
        if str(rdv.get('id')) == rdv_id_to_update:
            # 2. Effectuer la mise à jour
            player_name = updated_rdv.get('player_name', rdv.get('Joueuse', rdv.get('player_name')))
            
            updated_rdv_entry = rdv.copy()
            updated_rdv_entry.update({
                "player_name": player_name, 
                "Joueuse": player_name, 
                "Heure": updated_rdv.get('Heure', rdv.get('Heure')),
                "Type": updated_rdv.get('Type', rdv.get('Type')),
                "Détails": updated_rdv.get('Détails', rdv.get('Détails')),
                "Jour": updated_rdv.get('Jour', rdv.get('Jour')),
                # Maintenir les champs existants non modifiés dans le formulaire
            })
            new_appointments_list.append(updated_rdv_entry)
            found = True
        else:
            new_appointments_list.append(rdv)
            
    if found:
        # 3. Écrire la nouvelle liste complète sur le disque
        if write_appointments_to_file(new_appointments_list, date_key):
            return True
        else:
            return False
    else:
        print(f"Erreur: ID de RDV '{rdv_id_to_update}' non trouvé dans le store pour la mise à jour.")
        return False

def delete_appointment_persistent(rdv_id, date_key):
    """
    Supprime un RDV et persiste les changements sur le disque.
    """
    # 1. Obtenir la liste actuelle des RDV
    all_appointments = load_individual_appointments(date_key) 
    rdv_id_str = str(rdv_id) 
    initial_len = len(all_appointments)

    # 2. Créer la nouvelle liste sans l'élément à supprimer
    new_appointments_list = [rdv for rdv in all_appointments if str(rdv.get('id')) != rdv_id_str]
    
    if len(new_appointments_list) < initial_len:
        # 3. L'élément a été trouvé et supprimé en mémoire. Écrire la nouvelle liste.
        if write_appointments_to_file(new_appointments_list, date_key):
            return True
        else:
            return False
    
    print(f"Erreur: ID de RDV '{rdv_id_str}' non trouvé pour la suppression.")
    return False

def handle_edit_click(rdv_data):
    """Set l'état d'édition et relance l'application."""
    # S'assurer que les données pour l'édition utilisent la clé 'Joueuse' pour la compatibilité
    if 'player_name' in rdv_data and 'Joueuse' not in rdv_data:
        rdv_data['Joueuse'] = rdv_data['player_name']
        
    st.session_state.editing_rdv_id = rdv_data['id']
    st.session_state.editing_rdv_data = rdv_data
    st.rerun()

def display_edit_form(players_data, update_callback, date_key):
    """Affiche le formulaire de modification d'un RDV."""
    
    rdv_data = st.session_state.get('editing_rdv_data')
    if not rdv_data: return

    st.markdown("### 📝 Modifier le RDV")
    st.markdown(f"**RDV en cours de modification :** {rdv_data.get('Joueuse')} le {rdv_data.get('Jour')} à {rdv_data.get('Heure')}")

    # Récupérer la liste des noms de joueuses pour le selectbox
    player_display_names = players_data['JoueuseDisplayName'].tolist()
    current_player_name = rdv_data.get('Joueuse', player_display_names[0] if player_display_names else '')
    
    if current_player_name not in player_display_names and current_player_name:
        player_display_names.insert(0, current_player_name)
    
    current_player_index = player_display_names.index(current_player_name) if current_player_name in player_display_names else 0

    st.markdown("<div style='border: 1px solid #ffcc00; padding: 20px; border-radius: 8px; background-color: #fff9e6;'>", unsafe_allow_html=True)

    
    # --- DÉBUT DU FORMULAIRE (Contient les champs et le bouton submit) ---
    with st.form("edit_appointment_form", clear_on_submit=False):
        
        # Sélecteur de Joueuse
        new_player_name_display = st.selectbox(
            "Joueuse concernée",
            player_display_names,
            index=current_player_index,
            key="edit_player_select"
        )
        
        # Sélecteur de Jour (pour pouvoir déplacer le RDV)
        current_day_index = JOURS_SEMAINE.index(rdv_data.get('Jour', 'Lundi')) if rdv_data.get('Jour') in JOURS_SEMAINE else 0
        new_day = st.selectbox(
            "Jour du RDV",
            JOURS_SEMAINE,
            index=current_day_index,
            key="edit_day_select"
        )

        col_time, col_type = st.columns(2)
        
        with col_time:
            new_time = st.text_input("Heure (ex: 10:30)", value=rdv_data.get('Heure', ''), key="edit_time")
        with col_type:
            new_type = st.text_input("Type d'activité", value=rdv_data.get('Type', ''), key="edit_type")
        
        new_details = st.text_area("Détails", value=rdv_data.get('Détails', ''), key="edit_details")
        
        # CORRECTION DU PROBLÈME DE BOUTON MANQUANT:
        # On définit de nouvelles colonnes DANS le formulaire pour le bouton Submit
        # et on utilise un bouton standard HORS du formulaire pour l'annulation
        col_save_inner, col_spacer_inner = st.columns(2)
        
        # Le bouton de soumission DOIT être un st.form_submit_button et DANS le formulaire
        with col_save_inner:
            submitted = st.form_submit_button("💾 Sauvegarder les modifications", type="primary", use_container_width=True)

        if submitted:
            if not new_time or not new_type:
                st.error("Veuillez remplir l'heure et le type d'activité.")
            else:
                updated_rdv = {
                    "id": rdv_data['id'],
                    "player_name": new_player_name_display, 
                    "Jour": new_day,
                    "Heure": new_time,
                    "Type": new_type,
                    "Détails": new_details
                }
                
                if update_callback(updated_rdv, date_key):
                    st.success(f"RDV pour {new_player_name_display} mis à jour avec succès.")
                    st.session_state.editing_rdv_id = None
                    st.session_state.editing_rdv_data = None
                    st.rerun()
                else:
                    st.error("Échec de la mise à jour du rendez-vous.")
    
    # --- BOUTON ANNULER (DOIT ÊTRE HORS DU FORMULAIRE) ---
    # On définit de nouvelles colonnes HORS du formulaire, alignées à droite.
    col_spacer_after, col_cancel_after = st.columns(2) 

    with col_cancel_after:
        # st.button() est HORS du formulaire, cela fonctionne et annule l'édition
        if st.button("❌ Annuler", key="edit_cancel_btn_final", use_container_width=True):
            st.session_state.editing_rdv_id = None
            st.session_state.editing_rdv_data = None
            st.rerun()


    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("---")

def display_integrated_planning_management(df_planning: pd.DataFrame, players_data: pd.DataFrame, date_key: str, save_callback, delete_callback, update_callback):
    """
    Affiche le planning intégré (collectif + individuel) par jour
    et permet la gestion centralisée des RDV individuels pour toutes les joueuses.
    """
    
    if df_planning.empty:
        st.warning("Planning collectif non chargé.")
        return
    
    # 1. PRÉPARATION DES NOMS DE JOUEUSES
    filter_key = None 
    if 'Prénom' in players_data.columns:
        filter_key = 'Prénom'
    elif 'NOM' in players_data.columns: 
        filter_key = 'NOM'
    
    if not filter_key:
        st.error("Impossible de trouver la colonne des noms de joueuses ('Prénom' ou 'NOM').")
        return
        
    players_data['JoueuseDisplayName'] = players_data[filter_key]
    if 'Prénom' in players_data.columns and 'NOM' in players_data.columns:
        players_data['JoueuseDisplayName'] = players_data.apply(
            lambda row: f"{row['Prénom']} {row['NOM'].upper()}", axis=1
        )
    elif 'NOM' in players_data.columns:
        players_data['JoueuseDisplayName'] = players_data['NOM'].str.upper()

    player_display_names = sorted(players_data['JoueuseDisplayName'].tolist())
    
    if 'editing_rdv_id' not in st.session_state: st.session_state.editing_rdv_id = None
    if 'editing_rdv_data' not in st.session_state: st.session_state.editing_rdv_data = None
    
    # 2. AFFICHAGE DU FORMULAIRE DE MODIFICATION (PRIORITAIRE)
    if st.session_state.editing_rdv_id:
        # On passe le DataFrame mis à jour avec la colonne 'JoueuseDisplayName'
        display_edit_form(players_data, update_callback, date_key)
        # Sortir de la boucle jour par jour pour afficher uniquement le formulaire
        return

    # 3. AFFICHAGE DU PLANNING JOUR PAR JOUR
    
    # Utilise la fonction load_individual_appointments pour obtenir l'état actuel du fichier
    df_appointments = get_appointments_df(load_individual_appointments(date_key))
    day_column = 'Jour' 

    if df_appointments.empty:
        st.info("Aucun rendez-vous individuel chargé pour la semaine sélectionnée.")
    elif day_column not in df_appointments.columns:
        st.error(f"Erreur: La colonne '{day_column}' est manquante dans les RDV. (Vérifiez votre fonction de chargement).")
        return 
    
    df_collective_reindexed = df_planning.reindex(PERIODES_KEYS, fill_value="") 
    
    # Affichage en grille jour par jour
    for jour in JOURS_SEMAINE:
        st.subheader(f"🗓️ {jour}")
        
        col_collectif, col_individuel = st.columns([1, 2]) 

        # --- COLONNE 1: PLANNING COLLECTIF ---
        with col_collectif:
            st.markdown("**Planning Collectif (Base)**", help="Activités d'équipe, non modifiables.")
            st.markdown("---")

            for i, periode_key_short in enumerate(PERIODES_KEYS):
                periode_full = PERIODES_JOURNEE_INDIV[i] 
                
                cell_value = df_collective_reindexed.loc[periode_key_short, jour] \
                    if jour in df_collective_reindexed.columns and periode_key_short in df_collective_reindexed.index \
                    else ""
                
                display_value = format_cell(cell_value)
                color_style = get_match_color(cell_value)
                
                if display_value != "N/A" and display_value.strip() != "":
                    html_content_periode = f"""
                    <div style='margin-bottom: 5px; border-bottom: 1px dashed #cccccc; padding-bottom: 5px;'>
                        <p style='margin: 0; font-size: 0.8em; font-weight: bold; color: #555;'>{periode_full}:</p>
                        <div style='{color_style} padding: 5px; border-radius: 3px; font-size: 0.9em; min-height: 28px;'>
                            {display_value.replace('\n', '<br>')}
                        </div>
                    </div>
                    """
                    st.markdown(html_content_periode, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)
            
        # --- COLONNE 2: GESTION DES RDV INDIVIDUELS (CENTRALISÉE) ---
        with col_individuel:
            st.markdown(f"**RDV Individuels du Jour**")
            
            rdv_du_jour_list = []
            if not df_appointments.empty and day_column:
                rdv_du_jour = df_appointments[
                    (df_appointments[day_column] == jour)
                ].sort_values(by="Heure", ascending=True).reset_index(drop=True)
                
                rdv_du_jour_list = rdv_du_jour.to_dict('records') 

            # --- Affichage du header des RDV ---
            if rdv_du_jour_list:
                col_ratios = [1, 2, 2, 3.5, 1.25, 1.25]
                cols_header = st.columns(col_ratios)
                cols_header[0].markdown("**Heure**")
                cols_header[1].markdown("**Joueuse**") 
                cols_header[2].markdown("**Type**")
                cols_header[3].markdown("**Détails**")
                cols_header[4].markdown("`Action`")
                cols_header[5].markdown("`Action`")
                st.markdown("<hr style='margin: 0 0 5px 0;'>", unsafe_allow_html=True) 

                # --- Boucle pour afficher chaque RDV avec ses boutons ---
                for rdv in rdv_du_jour_list:
                    rdv_id = rdv.get('id')
                    rdv_time = rdv.get('Heure', 'N/A')
                    rdv_player = rdv.get('Joueuse', 'Inconnue') 
                    rdv_type = rdv.get('Type', 'N/A')
                    rdv_details = (rdv.get('Détails', 'Aucune description.')[:40] + '...') if len(rdv.get('Détails', '')) > 40 else rdv.get('Détails', 'Aucune description.')

                    if not rdv_id: continue

                    cols = st.columns(col_ratios, gap="small")
                    
                    # Données
                    cols[0].markdown(f"<span style='font-size: 0.9em;'>{rdv_time}</span>", unsafe_allow_html=True)
                    cols[1].markdown(f"<span style='font-size: 0.9em; font-weight: bold; color: #333;'>{rdv_player}</span>", unsafe_allow_html=True) 
                    cols[2].markdown(f"<span style='font-size: 0.9em;'>{rdv_type}</span>", unsafe_allow_html=True)
                    cols[3].markdown(f"<span style='font-size: 0.9em;'>{rdv_details}</span>", unsafe_allow_html=True)
                    
                    # Bouton Modifier
                    with cols[4]:
                        if st.button("✏️ Modif", key=f"edit_btn_{rdv_id}_{jour}", use_container_width=True):
                            handle_edit_click(rdv)
                    
                    # Bouton Supprimer
                    with cols[5]:
                        with st.popover("🗑️ Suppr"):
                            st.write(f"Supprimer le RDV de {rdv_player} à {rdv_time} ({jour}) ?")
                            if st.button("Confirmer la suppression", 
                                         key=f"confirm_del_btn_{rdv_id}_{jour}", 
                                         type="primary",
                                         use_container_width=True):
                                if delete_callback(rdv_id, date_key):
                                    st.toast("Rendez-vous supprimé avec succès !", icon='✅')
                                    st.rerun() 
                                else:
                                    st.error("La suppression a échoué.")
            else:
                st.info("Aucun RDV individuel prévu pour ce jour.")
                
            st.markdown("---")
            
            # --- FORMULAIRE D'AJOUT RAPIDE (Intégré à la colonne) ---
            st.markdown("### ➕ Ajouter un RDV")
            
            with st.form(f"add_appointment_form_{jour}", clear_on_submit=True):
                new_player_name_display = st.selectbox(
                    "Joueuse concernée",
                    player_display_names,
                    key=f"new_player_select_{jour}"
                )

                col_time, col_type = st.columns(2)
                
                with col_time:
                    new_time = st.text_input("Heure (ex: 10h30)", key=f"new_time_{jour}")
                with col_type:
                    new_type = st.selectbox(
                        "Type d'activité", 
                        TYPES_RDV, # Utilisation de la liste des types
                        key=f"new_type_select_{jour}")
                
                new_details = st.text_area("Détails", key=f"new_details_{jour}")
                
                submitted = st.form_submit_button("✅ Ajouter le RDV pour ce jour")
                
                if submitted:
                    if not new_player_name_display:
                         st.error("Veuillez sélectionner une joueuse.")
                    elif new_time and new_type:
                        new_rdv = {
                            "id": str(uuid.uuid4()), 
                            "player_name": new_player_name_display, 
                            "Jour": jour, 
                            "Heure": new_time,
                            "Type": new_type,
                            "Détails": new_details
                        }
                        
                        if save_callback(new_rdv, date_key):
                            st.success(f"Rendez-vous ajouté pour {new_player_name_display} ce {jour}.")
                            st.rerun()
                        else:
                            st.error("Échec de l'ajout du rendez-vous.")
                    else:
                        st.error("Veuillez remplir l'heure et le type d'activité.")

        st.markdown("<hr style='border: 4px solid #f0f0f0; margin: 20px 0;'>", unsafe_allow_html=True)






def format_cell(value):
    """Formate le contenu de la cellule pour l'affichage HTML."""
    if pd.isna(value) or value == "":
        return "DISPONIBLE"
    return str(value)