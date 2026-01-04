import streamlit as st
import pandas as pd
import os
import re
import json
from datetime import datetime, time
from pathlib import Path
from typing import Dict, List
from sections.menu.menu import custom_sidebar_menu
from sections.planning.pipeline import get_match_color
from sections.planning.pipeline_indiv import load_weekly_plannings, load_player_data, initialize_session_state, get_week_names, get_current_week_monday_str, save_appointment_persistent, update_appointment_persistent, delete_appointment_persistent, display_integrated_planning_management

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

# Sidebar
custom_sidebar_menu() 

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

COLLECTIVE_TIMES_MAP = {
    "Matin": "08:00", 
    "Midi": "12:00", 
    "Après-midi": "14:00"
}

initialize_session_state()

# --- GESTION DES DOSSIERS (Vérification et Création) ---
# Assurer l'existence du répertoire de données avant toute opération de lecture
if not os.path.exists(FILE_PATH_PLANNING):
  os.makedirs(FILE_PATH_PLANNING, exist_ok=True)
  st.warning(f"Le répertoire de données `{FILE_PATH_PLANNING}` a été créé. Veuillez y placer vos fichiers (nommés planning_AAAA_MM_JJ.csv) pour que l'application fonctionne.")

def get_indiv_filepath(date_key: str):
  """
  Retourne le chemin complet du fichier JSON des RDV individuels pour une semaine donnée,
  et assure que le répertoire existe.
  """
  Path(FILE_PATH_PLANNING_INDIV).mkdir(parents=True, exist_ok=True)
  return Path(FILE_PATH_PLANNING_INDIV) / f"rdv_indiv_{date_key}.json"

def read_appointments_from_file(date_key: str) -> list:
    """Lit les RDV persistants depuis le fichier JSON spécifique."""
    filepath = get_indiv_filepath(date_key)
    if filepath.exists():
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            # En cas d'erreur de lecture (fichier vide ou corrompu), on log et retourne une liste vide
            print(f"Erreur de décodage JSON dans {filepath}. Fichier ignoré.")
            return []
    return []

def write_appointments_to_file(date_key: str, appointments: list):
    """Écrit la liste des RDV dans le fichier JSON, et invalide le cache."""
    filepath = get_indiv_filepath(date_key)
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            # Utiliser ensure_ascii=False pour conserver les accents
            json.dump(appointments, f, indent=4, ensure_ascii=False)
        
        # INVALIDE LE CACHE : Essentiel pour que Streamlit recharge les données mises à jour
        # Invalide le cache de la fonction de chargement pour forcer Streamlit à recharger les données
        load_individual_json_appointments.clear()
        
    except Exception as e:
        print(f"Erreur lors de l'écriture du fichier {filepath}: {e}")
        st.error(f"Erreur lors de l'écriture des données: {e}")

def load_individual_json_appointments(date_key: str):
  """
  Charge les RDV persistants directement depuis le fichier JSON
  (Remplace la logique MOCK de session_state).
  """
  return read_appointments_from_file(date_key)

# La fonction de l'utilisateur avec le décorateur st.cache_data
@st.cache_data(show_spinner="Chargement des rendez-vous individuels...")
def load_individual_appointments(date_key: str):
  """
  Charge les RDV persistants depuis l'état de session (MOCK). 
  En production, le code fourni par l'utilisateur serait utilisé.
  """
  # En production, vous utiliseriez la logique de lecture JSON/fichier de l'utilisateur:
  # filepath = get_indiv_filepath(date_key)
  # ... lecture JSON ...
  
  # En mode MOCK, nous utilisons st.session_state
  if 'appointments_data' not in st.session_state:
    initialize_session_state()
    
  # Retourne une copie pour éviter les modifications directes au cache
    return st.session_state.appointments_data.get(date_key, []) # Utiliser la clé de date
    
# --- Fonction utilisateur pour DataFrame (Non modifiée, mais incluse) ---
def get_appointments_df(appointments_list: List[Dict[str, str]]) -> pd.DataFrame:
    """
    Convertit la liste des RDV en DataFrame et garantit un index unique
    et la présence des colonnes critiques.

    N.B. Utilise la fonction utilitaire 'clean_time_str' pour le tri.
    """
    COLUMNS = ['id', 'Joueuse', 'Jour', 'Heure', 'Type', 'Détails']

    if not appointments_list:
        return pd.DataFrame(columns=COLUMNS)
    
    df = pd.DataFrame(appointments_list)
    
    # 1. Renommage/Normalisation de la colonne Joueuse
    if 'Joueuse' not in df.columns:
        if 'player_name' in df.columns:
            df = df.rename(columns={'player_name': 'Joueuse'})
        else:
            df['Joueuse'] = None

    # 2. Garantie des colonnes
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = None 
        
    df = df.reindex(columns=COLUMNS)
    
    # 3. Tri par heure (CORRECTION: Utilise clean_time_str)
    try:
        # 3.1. Convertir '10h30' en '10:30'
        df['Heure_Clean'] = df['Heure'].apply(lambda x: clean_time_str(x) if isinstance(x, str) else '00:00')
        
        # 3.2. Convertir en objet time pour le tri
        df['Heure_Sort'] = df['Heure_Clean'].apply(lambda x: datetime.strptime(x, '%H:%M').time())
        
        # 3.3. Trier
        df.sort_values(by=['Jour', 'Heure_Sort'], inplace=True, kind='stable')
        
        # 3.4. Nettoyer les colonnes de tri temporaires
        df.drop(columns=['Heure_Sort', 'Heure_Clean'], inplace=True)
        
    except Exception:
        # En cas d'erreur de tri, on continue sans trier
        pass 
        
    return df.reset_index(drop=True)

# --- TAB 2: VUE GLOBALE FUSIONNÉE (Implémentation) ---
def charger_rdv_individuels():
    """
    Trouve tous les fichiers JSON dans le répertoire DATA_DIR et les agrège dans un DataFrame.
    """
    print("--- 2. Chargement et agrégation des données des RDV ---")
    
    # Correction: Utiliser le pattern de fichier "rdv_indiv_*.json"
    rdv_files = Path(FILE_PATH_PLANNING_INDIV).glob('rdv_indiv_*.json')
    
    # Liste pour stocker les DataFrames de chaque fichier
    all_dfs = []
    
    for file_path in rdv_files:
        try:
            # Charger le contenu du fichier JSON
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convertir la liste de dictionnaires en DataFrame
            df = pd.DataFrame(data)
            
            # Ajouter le nom du fichier comme colonne de source (utile pour l'analyse)
            df['Source_Fichier'] = file_path.name
            
            all_dfs.append(df)
            print(f"Chargement réussi du fichier : {file_path.name} ({len(df)} RDV)")

        except Exception as e:
            print(f"Erreur lors du chargement de {file_path.name}: {e}")
    
    
    if not all_dfs:
        print("Aucun fichier JSON de RDV trouvé. Le DataFrame final sera vide.")
        return pd.DataFrame() # Retourne un DataFrame vide si rien n'est trouvé

    # 2. Concaténer tous les DataFrames en un seul
    df_final = pd.concat(all_dfs, ignore_index=True)

    # 3. Nettoyage et conversion des types (ex: convertir la date de création)
    if 'created_at' in df_final.columns:
        try:
            df_final['created_at'] = pd.to_datetime(df_final['created_at'])
            print("\nLa colonne 'created_at' a été convertie au format datetime.")
        except Exception as e:
            print(f"Avertissement : Impossible de convertir la colonne 'created_at' en datetime. Erreur : {e}")

    print("-" * 50)
    print(f"Agrégation terminée. Nombre total de RDV : {len(df_final)}")
    
    return df_final

def get_indiv_color(rdv_type: str) -> str:
    """
    Retourne un style de fond basé sur le type de RDV individuel, 
    en utilisant les groupes définis par l'utilisateur.
    """
    rdv_type = rdv_type.strip().lower()

    # GR 1 : "Soin", "Récup", "RDV Doc" -> Light Teal (Soins & Santé)
    if rdv_type in ["soin", "récup", "rdv doc"]:
        return "background-color: #e0fff4; border-left: 3px solid #00b894;"
    
    # GR 2 : "Séance Réa", "Séance Supplémentaire", "Muscu", "Testing" -> Light Orange (Performance & Physique)
    if rdv_type in ["séance réa", "séance supplémentaire", "muscu", "testing"]:
        return "background-color: #fff5e0; border-left: 3px solid #ff7f50;"
    
    # GR 3 : "Diététicien", "Entretien", "Prépa Mental", "Podologue" -> Light Lavender (Consultation & Support)
    if rdv_type in ["diététicien", "entretien", "prépa mental", "podologue"]:
        return "background-color: #f5e6ff; border-left: 3px solid #a29bfe;"
    
    # GR 4 : "RDV Extérieur" -> Light Cyan (Logistique & Externe)
    if rdv_type == "rdv extérieur":
        return "background-color: #e6f7ff; border-left: 3px solid #007bff;"
    
    # GR 5 : "Vidéo" -> Light Rose (Analyse & Tactique)
    if rdv_type == "vidéo":
        return "background-color: #ffe6f0; border-left: 3px solid #ff4757;"
    
    # GR 6 : "OP Joueuse" -> Light Gray (Opérations & Média)
    if rdv_type == "op joueuse":
        return "background-color: #f0f0f0; border-left: 3px solid #a4b0be;"

    # Couleur par défaut si le type n'est pas reconnu (utilisé précédemment)
    return "background-color: #e6f7ff; border-left: 3px solid #007bff;"

def clean_time_str(time_str: str) -> str:
    """
    Normalise le format français 'HHhMM' ou 'HHh' vers 'HH:MM' pour la comparaison interne.
    Ex: '10h30' -> '10:30', '14h' -> '14:00'
    """
    if not time_str:
        return "08:00"
    
    time_str = time_str.lower().strip()
    
    # 1. Handle "HHh" (e.g., "10h") -> "10:00"
    if time_str.endswith('h'):
        return time_str.replace('h', ':00')
    
    # 2. Handle "HHhMM" (e.g., "10h30") -> "10:30"
    return time_str.replace('h', ':')


def map_time_to_collective_start(time_str: str) -> str:
    """
    Mappe une heure spécifique (en format français ou standard) à l'heure de début 
    de son bloc collectif (08:00, 12:00, ou 14:00) en utilisant la comparaison interne normalisée.
    """
    cleaned_time_str = clean_time_str(time_str)

    try:
        t = datetime.strptime(cleaned_time_str, '%H:%M').time()
    except ValueError:
        return "08:00" 

    # Définir les heures de début des blocs comme objets time
    time_14_00 = time(14, 0)
    time_12_00 = time(12, 0)

    if t >= time_14_00:
        return "14:00" # Bloc Après-midi (à partir de 14h00)
    elif t >= time_12_00:
        return "12:00" # Bloc Midi (à partir de 12h00)
    else:
        return "08:00" # Bloc Matin (à partir de 08:00)


# --- TAB 2: VUE GLOBALE FUSIONNÉE (Code utilisateur intégré) ---
def apply_styles_and_sort(cell_activities: list):
    """ Trie les activités par heure et applique les styles HTML pour chaque cellule. """
    if not cell_activities: return "—"
    
    # 3.1. TRIER par heure de l'activité
    def sort_key(activity):
        try:
            t_str_cleaned = clean_time_str(activity['time'])
            return datetime.strptime(t_str_cleaned, '%H:%M').time()
        except ValueError:
            return time(23, 59) 

    sorted_activities = sorted(cell_activities, key=sort_key)
    
    styled_activities_html = []
    
    # 3.2. CONSTRUIRE l'affichage HTML
    for activity in sorted_activities:
        content = activity['content']
        
        if activity['is_collective']:
            # On recherche la première ligne pour déterminer la couleur collective
            first_line = content.split('\n')[0]
            activity_name_match = re.search(r"-\s*(.*)", first_line)
            activity_name = activity_name_match.group(1).strip() if activity_name_match else first_line
            
            collective_style = get_match_color(activity_name)
            display_content = content.replace('\n', '<br>')
            
            styled_collective_block = f"<div style='{collective_style} padding: 4px 6px; border-radius: 3px; margin-bottom: 2px; font-weight: bold; overflow-wrap: break-word;'>{display_content}</div>"
            styled_activities_html.append(styled_collective_block)
        else:
            rdv_type = activity.get('type', 'Default')
            indiv_style = get_indiv_color(rdv_type)
            # Affichage de l'heure et du type pour l'individuel
            display_line = content
            
            styled_line = f"<div style='{indiv_style} padding: 4px 6px; border-radius: 3px; margin-bottom: 2px; font-weight: normal; overflow-wrap: break-word;'>{display_line.replace('\n', '<br>') }</div>"
            styled_activities_html.append(styled_line) 

    return "".join(styled_activities_html)


def display_html_table_styles():
    """Applique les styles CSS pour le rendu de la table fusionnée."""
    custom_css = """
    <style>
    .dataframe {
      width: 100%; border-collapse: collapse; font-family: Arial, sans-serif; font-size: 0.9em; text-align: left;
      border: 2px solid #333 !important;
    }
    .dataframe th, .dataframe td {
      padding: 4px; vertical-align: top; min-width: 120px; height: auto; border: none !important;
    }
    .dataframe thead th {
      background-color: #f2f2f2; font-weight: bold; text-align: center;
      border-bottom: 2px solid #333 !important; border-right: 2px solid #333 !important; border-top: none;
    }
    .dataframe thead th:last-child { border-right: none !important; }
    .dataframe tbody td, .dataframe tbody th { border-top: none !important; border-bottom: none !important; }
    .dataframe tbody td { border-right: 2px solid #333 !important; }
    .dataframe tbody td:last-child { border-right: none !important; }
    .dataframe th:first-child {
      width: 80px; background-color: #e6e6e6; font-weight: bold; text-align: center; vertical-align: middle; line-height: normal;
      border-right: 2px solid #333 !important;
    }
    .dataframe td div { white-space: normal; } 
    .dataframe td { line-height: 1.3; }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


# --- TAB 2: VUE GLOBALE FUSIONNÉE (Code utilisateur intégré) ---
def apply_styles_and_sort(cell_activities: list):
    """ Trie les activités par heure et applique les styles HTML pour chaque cellule. """
    if not cell_activities: return "—"
    
    # 3.1. TRIER par heure de l'activité
    def sort_key(activity):
        try:
            t_str_cleaned = clean_time_str(activity['time'])
            return datetime.strptime(t_str_cleaned, '%H:%M').time()
        except ValueError:
            return time(23, 59) 

    sorted_activities = sorted(cell_activities, key=sort_key)
    
    styled_activities_html = []
    
    # 3.2. CONSTRUIRE l'affichage HTML
    for activity in sorted_activities:
        content = activity['content']
        
        if activity['is_collective']:
            # On recherche la première ligne pour déterminer la couleur collective
            first_line = content.split('\n')[0]
            activity_name_match = re.search(r"-\s*(.*)", first_line)
            activity_name = activity_name_match.group(1).strip() if activity_name_match else first_line
            
            collective_style = get_match_color(activity_name)
            display_content = content.replace('\n', '<br>')
            
            styled_collective_block = f"<div style='{collective_style} padding: 4px 6px; border-radius: 3px; margin-bottom: 2px; font-weight: bold; overflow-wrap: break-word;'>{display_content}</div>"
            styled_activities_html.append(styled_collective_block)
        else:
            rdv_type = activity.get('type', 'Default')
            indiv_style = get_indiv_color(rdv_type)
            # Affichage de l'heure et du type pour l'individuel
            display_line = content
            
            styled_line = f"<div style='{indiv_style} padding: 4px 6px; border-radius: 3px; margin-bottom: 2px; font-weight: normal; overflow-wrap: break-word;'>{display_line.replace('\n', '<br>') }</div>"
            styled_activities_html.append(styled_line) 

    return "".join(styled_activities_html)


def display_html_table_styles():
    """Applique les styles CSS pour le rendu de la table fusionnée."""
    custom_css = """
    <style>
    .dataframe {
      width: 100%; border-collapse: collapse; font-family: Arial, sans-serif; font-size: 0.9em; text-align: left;
      border: 2px solid #333 !important;
    }
    .dataframe th, .dataframe td {
      padding: 4px; vertical-align: top; min-width: 120px; height: auto; border: none !important;
    }
    .dataframe thead th {
      background-color: #f2f2f2; font-weight: bold; text-align: center;
      border-bottom: 2px solid #333 !important; border-right: 2px solid #333 !important; border-top: none;
    }
    .dataframe thead th:last-child { border-right: none !important; }
    .dataframe tbody td, .dataframe tbody th { border-top: none !important; border-bottom: none !important; }
    .dataframe tbody td { border-right: 2px solid #333 !important; }
    .dataframe tbody td:last-child { border-right: none !important; }
    .dataframe th:first-child {
      width: 80px; background-color: #e6e6e6; font-weight: bold; text-align: center; vertical-align: middle; line-height: normal;
      border-right: 2px solid #333 !important;
    }
    .dataframe td div { white-space: normal; } 
    .dataframe td { line-height: 1.3; }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


# --- TAB 2: VUE GLOBALE FUSIONNÉE (Code utilisateur intégré) ---
def display_tab_globale(df_planning: pd.DataFrame, date_key: str):
    """
    Affiche un tableau fusionnant le planning collectif et tous les RDV individuels
    pour toutes les joueuses, en utilisant les périodes collectives comme index principal.
    """
    st.header("Vue Globale Fusionnée")

    # 1. PRÉPARATION DES DONNÉES
    df_appointments = get_appointments_df(load_individual_json_appointments(date_key))
    
    # Initialisation du DataFrame avec des listes vides
    df_fused = pd.DataFrame(index=PERIODES_KEYS, columns=JOURS_SEMAINE)
    df_fused = df_fused.apply(lambda col: [[] for _ in range(len(col))], axis=0) 

    # 2. POPULER LE DATAFRAME FUSIONNÉ
    df_collective_reindexed = df_planning.reindex(PERIODES_KEYS, fill_value="")

    for jour in JOURS_SEMAINE:
        
        # A. Ajouter les activités collectives
        for periode_key in PERIODES_KEYS:
            heure_start = COLLECTIVE_TIMES_MAP.get(periode_key, "00:00")
            
            if periode_key in df_fused.index and periode_key in df_collective_reindexed.index:
                collective_activity = df_collective_reindexed.loc[periode_key, jour] if jour in df_collective_reindexed.columns else ""
                
                if isinstance(collective_activity, str) and collective_activity.strip() != "":
                    collective_lines = collective_activity.strip().split('\n')
                    current_collective_block = []
                    current_block_time = heure_start
                    match_regex = r"(\d{1,2}h\d{0,2}) ?- ?(.*)" 
                    
                    for line in collective_lines:
                        line = line.strip()
                        if not line: continue
                        
                        match = re.match(match_regex, line)
                        
                        if match:
                            matched_time_str = match.group(1)
                            
                            if current_collective_block:
                                df_fused.loc[periode_key, jour].append({
                                    'time': current_block_time, 
                                    'content': "\n".join(current_collective_block),
                                    'is_collective': True,
                                    'type': None
                                })
                                
                            current_block_time = matched_time_str
                            content = match.group(2).strip()
                            current_collective_block = [f"{current_block_time} - {content}"]

                        else:
                            current_collective_block.append(line)
                            
                    if current_collective_block:
                          df_fused.loc[periode_key, jour].append({
                              'time': current_block_time, 
                              'content': "\n".join(current_collective_block),
                              'is_collective': True,
                              'type': None
                            })

        # B. Ajouter les RDV individuels
        if not df_appointments.empty:
            rdv_du_jour = df_appointments[df_appointments['Jour'] == jour]
            
            for _, rdv in rdv_du_jour.iterrows():
                heure_indiv = rdv['Heure'] 
                type_rdv = rdv['Type']
                
                heure_collective_start = map_time_to_collective_start(heure_indiv)
                period_key_for_rdv = next((k for k, v in COLLECTIVE_TIMES_MAP.items() if v == heure_collective_start), None)
                
                if period_key_for_rdv and period_key_for_rdv in df_fused.index:
                    # Ici on utilise la colonne 'Joueuse' qui contient le format Prénom NOM.UPPER
                    joueuse_full_name = rdv['Joueuse'] 
                    joueuse_parts = joueuse_full_name.split()
                    prenom = joueuse_parts[0] if joueuse_parts else 'Joueuse'
                    # Prendre la première lettre du NOM, qui est la dernière partie de 'Joueuse' (ex: MEREAU -> M.)
                    nom_initiale = (joueuse_parts[-1][0] + '.') if len(joueuse_parts) > 1 else ''
                    joueuse_display = f"{prenom} {nom_initiale}"

                    # Contenu du bloc individuel
                    rdv_text = f"Indiv {heure_indiv}: {joueuse_display} ({type_rdv})"
                    
                    df_fused.loc[period_key_for_rdv, jour].append({
                        'time': heure_indiv, 
                        'content': rdv_text,
                        'is_collective': False,
                        'type': type_rdv
                    })


    # 3. MISE EN FORME ET AFFICHAGE
    
    # Appliquer le formatage HTML
    df_display = df_fused.apply(lambda col: col.map(apply_styles_and_sort))

    # Transformer le DataFrame en HTML
    html_table = df_display.to_html(escape=False, header=True, index=True)

    # Styles spécifiques à la table fusionnée
    display_html_table_styles()
    
    st.markdown(html_table, unsafe_allow_html=True)


# --- TAB 3: VUE INDIVIDUELLE FILTRÉE (Mis à jour pour utiliser la présentation de la Tab 2) ---
def display_individual_planning_view(df_planning: pd.DataFrame, players_data: pd.DataFrame, friendly_name: str, date_key_for_load: str):
    """ Affiche le planning individuel (lecture seule) consolidé pour la joueuse sélectionnée. """
    st.header("Vue Individuelle Filtrée")
    
    # 1. Préparation de la liste des joueuses pour le SelectBox (CRÉATION de la colonne 'Joueuse' formatée)
    players_data['Joueuse'] = players_data.apply(
        lambda row: f"{row['Prénom']} {row['NOM'].upper()}", axis=1
    )
    player_names = sorted(players_data['Joueuse'].tolist()) 
    
    if not player_names:
        st.warning("Aucune joueuse chargée.")
        return
        
    # 2. Sélection de la joueuse à visualiser
    selected_player_name = st.selectbox(
        "Sélectionnez la joueuse pour visualiser son planning :",
        player_names,
        key="individual_selected_player_view"
    )
    
    st.subheader(f"Planning Consolidé de {selected_player_name} ({friendly_name})")
    
    if df_planning.empty:
        st.warning("Planning collectif non chargé pour cette semaine.")
        return
        
    # 3. CHARGEMENT ET FILTRAGE DES RDV INDIVIDUELS
    # df_appointments contient la liste de TOUS les RDV de la semaine
    df_appointments = get_appointments_df(load_individual_json_appointments(date_key_for_load))
    
    # *** POINT CLÉ : FILTRAGE ***
    df_appointments_filtered = pd.DataFrame()
    if not df_appointments.empty:
         # Le filtre fonctionne sur la colonne 'Joueuse' qui contient le format 'Prénom NOM.UPPER'
         # La clé 'Joueuse' dans le DataFrame (issue du JSON) est "Alizée MEREAU"
         # Le selected_player_name (issu du selectbox) est "Alizée MEREAU"
         df_appointments_filtered = df_appointments[
            df_appointments['Joueuse'] == selected_player_name
        ].copy()
        
    # 4. INITIALISATION DU DATAFRAME FUSIONNÉ
    df_fused = pd.DataFrame(index=PERIODES_KEYS, columns=JOURS_SEMAINE)
    df_fused = df_fused.apply(lambda col: [[] for _ in range(len(col))], axis=0) 

    df_collective_reindexed = df_planning.reindex(PERIODES_KEYS, fill_value="")

    # 5. POPULER LE DATAFRAME FUSIONNÉ
    for jour in JOURS_SEMAINE:
        
        # A. Ajouter les activités collectives (identique à la Tab 2)
        for periode_key in PERIODES_KEYS:
            heure_start = COLLECTIVE_TIMES_MAP.get(periode_key, "00:00")
            
            if periode_key in df_fused.index and periode_key in df_collective_reindexed.index:
                collective_activity = df_collective_reindexed.loc[periode_key, jour] if jour in df_collective_reindexed.columns else ""
                
                if isinstance(collective_activity, str) and collective_activity.strip() != "":
                    collective_lines = collective_activity.strip().split('\n')
                    current_collective_block = []
                    current_block_time = heure_start
                    match_regex = r"(\d{1,2}h\d{0,2}) ?- ?(.*)" 
                    
                    for line in collective_lines:
                        line = line.strip()
                        if not line: continue
                        
                        match = re.match(match_regex, line)
                        
                        if match:
                            matched_time_str = match.group(1)
                            
                            if current_collective_block:
                                df_fused.loc[periode_key, jour].append({
                                    'time': current_block_time, 
                                    'content': "\n".join(current_collective_block),
                                    'is_collective': True,
                                    'type': None
                                })
                                
                            current_block_time = matched_time_str
                            content = match.group(2).strip()
                            current_collective_block = [f"{current_block_time} - {content}"]

                        else:
                            current_collective_block.append(line)
                            
                    if current_collective_block:
                          df_fused.loc[periode_key, jour].append({
                              'time': current_block_time, 
                              'content': "\n".join(current_collective_block),
                              'is_collective': True,
                              'type': None
                            })


        # B. Ajouter les RDV individuels (UNIQUEMENT ceux filtrés)
        if not df_appointments_filtered.empty:
            
            rdv_du_jour = df_appointments_filtered[df_appointments_filtered['Jour'] == jour]
            
            for _, rdv in rdv_du_jour.iterrows():
                heure_indiv = rdv['Heure'] 
                type_rdv = rdv['Type']
                rdv_details = rdv.get('Détails', '')
                
                heure_collective_start = map_time_to_collective_start(heure_indiv)
                period_key_for_rdv = next((k for k, v in COLLECTIVE_TIMES_MAP.items() if v == heure_collective_start), None)
                
                if period_key_for_rdv and period_key_for_rdv in df_fused.index:
                    
                    # Contenu du bloc individuel pour l'affichage (plus détaillé ici)
                    rdv_text = f"Mon RDV {heure_indiv}: {type_rdv}"
                    if rdv_details:
                        rdv_text += f"\n- Détails: {rdv_details}"
                    
                    df_fused.loc[period_key_for_rdv, jour].append({
                        'time': heure_indiv, 
                        'content': rdv_text,
                        'is_collective': False,
                        'type': type_rdv
                    })

    # 6. MISE EN FORME ET AFFICHAGE (Identique à la Tab 2)
    
    # Appliquer le formatage HTML
    df_display = df_fused.apply(lambda col: col.map(apply_styles_and_sort))

    # Transformer le DataFrame en HTML
    html_table = df_display.to_html(escape=False, header=True, index=True)

    # Styles spécifiques à la table fusionnée
    display_html_table_styles()
    
    st.markdown(html_table, unsafe_allow_html=True)


# --- FONCTION PRINCIPALE ---
def main():
    """Fonction principale de l'application Streamlit."""
    
    # Assurez-vous que les répertoires existent (pour les Mocks)
    os.makedirs(FILE_PATH_PLANNING_INDIV, exist_ok=True)
    os.makedirs(FILE_PATH_PLANNING, exist_ok=True)
    
    st.set_page_config(layout="wide", page_title="Gestion du Planning Sportif")
    initialize_session_state()

    st.title("⚽ Gestion du Planning d'Équipe")

    # Chargement des données
    weekly_plannings = load_weekly_plannings() 
    week_names = get_week_names() 
    
    # CHARGEMENT DU DATAFRAME ORIGINAL DES JOUEUSES
    players_data_original = load_player_data() 
    
    if players_data_original.empty or not weekly_plannings:
        st.error("Veuillez charger les données de base (joueuses et planning collectif).")
        return

    # --- SÉLECTION DE LA SEMAINE ---
    available_dates = sorted(list(weekly_plannings.keys()))
    display_options = {
        date_key: week_names.get(date_key, f"Semaine du {date_key}") 
        for date_key in available_dates
    }
    
    current_monday = get_current_week_monday_str() 
    default_index_semaine = 0
    if available_dates:
        try:
            default_index_semaine = available_dates.index(current_monday)
        except ValueError:
            default_index_semaine = len(available_dates) - 1
            if default_index_semaine < 0: default_index_semaine = 0
            
        selected_date_key = st.selectbox(
            "Sélectionnez la semaine :",
            options=available_dates,
            format_func=lambda x: display_options[x],
            index=default_index_semaine 
        )
    else:
        st.error("Aucune semaine disponible.")
        return 

    df_planning = weekly_plannings.get(selected_date_key, pd.DataFrame())
    friendly_name = display_options.get(selected_date_key, "Semaine inconnue")
    
    
    # --- PRÉPARATION DU DATAFRAME POUR L'UI (Gestion de la joueuse sélectionnée) ---
    
    # 1. CRÉER UNE COPIE POUR L'UI (PROTECTION DU CACHE)
    players_df_for_ui = players_data_original.copy()
    
    # 2. Créer la colonne 'Joueuse' sur la copie (Prénom NOM.UPPER)
    players_df_for_ui['Joueuse'] = players_df_for_ui['Prénom'].astype(str) + ' ' + players_df_for_ui['NOM'].astype(str).str.upper()
    joueuses_list = players_df_for_ui['Joueuse'].dropna().unique().tolist()
    
    if not joueuses_list:
        st.warning("La liste des joueuses est vide après chargement.")
        return

    # 3. Mettre à jour l'état de session si nécessaire (None ou joueuse invalide)
    if st.session_state.selected_joueuse is None or st.session_state.selected_joueuse not in joueuses_list:
        # Tente de sélectionner "Alizée MEREAU" par défaut pour le test
        default_test_player = "Alizée MEREAU"
        if default_test_player in joueuses_list:
            st.session_state.selected_joueuse = joueuses_list[0]
        else:
            st.session_state.selected_joueuse = joueuses_list[0]
        
    # 4. Déterminer l'index du SelectBox
    try:
        default_index_joueuse = joueuses_list.index(st.session_state.selected_joueuse)
    except ValueError:
        default_index_joueuse = 0
        st.session_state.selected_joueuse = joueuses_list[0] # Fallback
    
    # ====================================================================
    # --- SYSTÈME D'ONGLETS ---
    # ====================================================================
    tab1_gestion, tab2_globale, tab3_individuelle = st.tabs([
        "1. Gestion (Collectif & Indiv)", 
        "2. Vue Globale Fusionnée",
        "3. Vue Individuelle Filtrée"
    ]) 

    # --- TAB 1: GESTION INTÉGRÉE ---
    with tab1_gestion:
        display_integrated_planning_management(
            df_planning, 
            players_df_for_ui, 
            selected_date_key, 
            save_appointment_persistent,
            delete_appointment_persistent,
            update_appointment_persistent 
        )
        
    # --- TAB 2: VUE GLOBALE FUSIONNÉE ---
    with tab2_globale:
        display_tab_globale(df_planning, selected_date_key)
        
    # --- TAB 3: VUE INDIVIDUELLE FILTRÉE ---
    with tab3_individuelle:
        display_individual_planning_view(
            df_planning, 
            players_data_original, 
            friendly_name, 
            selected_date_key
        )


if __name__ == "__main__":
    main()