import streamlit as st
import pandas as pd
import os
import json
import plotly.express as px
from datetime import date, timedelta
from typing import Union
from sections.constantes import DICT_REPLACE_COLS, LIST_ORGANISATION_COLS_MATCH, cols_ref_match, ONGLET_GPS_SAISON, ONGLET_GPS_TYPE_MATCH, cols_joueuses, agg_dict_indiv, agg_dict_semaine

##Données séance###
#Remplacement start et end time en minute pour avoir le temps total
def convert_to_minutes(time_str: str) -> int:
    h, m, s = map(int, time_str.split(':'))
    return round(h *60 + m + s/60)

#Fonctions créations de colonnes
def creer_colonne(df: pd.DataFrame) -> pd.DataFrame:
    df['VHSR effort'] = df['VHSR + SPR effort'] - df['Sprint effort'] ##nbr VHSR
    return df

#Fonction noms colonnes
def renommer_colonne(df: pd.DataFrame, col_rename: dict) -> pd.DataFrame:
    return df.rename(columns=col_rename) 


def convert_total_duration_minutes(df: pd.DataFrame) -> pd.DataFrame:
    df['Total Time'] = df['End Time'].apply(convert_to_minutes) - df['Start Time'].apply(convert_to_minutes)
    return df

def add_columns_info_path(df: pd.DataFrame, jour_semaine: str, saison_seance: str) -> pd.DataFrame:
    df['Jour semaine'] = jour_semaine
    df['Saison'] = saison_seance
    return df

def parse_date(df: pd.DataFrame) -> pd.DataFrame:
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    df['Date'] = df['Date'].dt.strftime('%d-%m-%Y')
    return df

#Fonctions de nettoyage du CSV
def pipeline_nettoyage(df: pd.DataFrame, j_semaine: str = None, s_seance: str = None) -> pd.DataFrame:
    df = convert_total_duration_minutes(df=df)
    df = creer_colonne(df=df)
    df = renommer_colonne(df=df, col_rename=DICT_REPLACE_COLS)
    df = add_columns_info_path(df=df, jour_semaine=j_semaine, saison_seance=s_seance)
    df = parse_date(df=df)
    return df

#Importer les chemins de tous les fichiers
def get_all_chemin(chemin: Union[str, None]=None) -> list:
    if chemin is None:
        liste_chemin = os.getcwd().split('\\')
        chemin = "\\".join(liste_chemin) + "\\data\\séance\\"
    fichiers = os.listdir(chemin)
    total = []
    for i in fichiers:
        final_chemin = chemin + i
        total.append(final_chemin)
    return total

def recuperer_jour_fichier(path_fichier: str) -> str:
    split = path_fichier.rsplit('.csv')
    split_2 = split[0].rsplit('_', maxsplit=2)
    info_type_seance = split_2[-2]
    info_saison_seance = split_2[-1]
    return info_type_seance, info_saison_seance

def recuperer_all_files_gps()-> pd.DataFrame:
    chemins = get_all_chemin()
    df_collectif = []
    for file in chemins:
        jour_semaine, saison_seance = recuperer_jour_fichier(file)
        data = pd.read_csv(file, parse_dates=['Date'], dayfirst=True)
        data = pipeline_nettoyage(df=data, j_semaine=jour_semaine, s_seance=saison_seance)
        df_collectif.append(data)
    df_collectif = pd.concat(df_collectif)
    df_collectif['Name'] = pd.Categorical(df_collectif['Name'], categories=cols_joueuses, ordered=True)
    df_collectif['Date'] = pd.to_datetime(df_collectif['Date'], dayfirst=True)
    df_collectif = df_collectif.sort_values(by=['Date', 'Name'], ascending=[False, True])
    cols = df_collectif.columns.tolist() # Réorganiser l'ordre des colonnes
    cols.insert(0, cols.pop(cols.index('Name')))
    df_collectif = df_collectif[cols]
    df_collectif['Date'] = df_collectif['Date'].dt.strftime('%d-%m-%Y')
    return df_collectif

#Filtre année, 
def filtre_match(df: pd.DataFrame) -> pd.DataFrame:
    selection_annee = st.segmented_control("Année", ONGLET_GPS_SAISON, selection_mode="single", default=ONGLET_GPS_SAISON[0])
    selection_type = st.segmented_control("Type match", ONGLET_GPS_TYPE_MATCH, selection_mode="single", default=ONGLET_GPS_TYPE_MATCH[0])
    df_filtrer = df[(df['Saison'] == selection_annee) & (df['Type match'] == selection_type)]
    if df_filtrer.empty:
        st.warning("Aucun match ne correspond à la sélection")
        return pd.DataFrame()
    match_possible = sorted(df_filtrer['Activity Name'].unique())
    selection_match = st.segmented_control("Match", options=match_possible, default=match_possible[0])
    df_final = df_filtrer[df_filtrer['Activity Name'] == selection_match]
    return df_final

def regrouper_par_semaine_groupe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    df['Date'] = pd.to_datetime(df['Date'])
    
    jour_de_la_semaine = df['Date'].dt.weekday
    df['date_debut_semaine'] = df['Date'] - pd.to_timedelta(jour_de_la_semaine, unit='D')
    
    df_par_semaine = df.groupby(['date_debut_semaine'], as_index=False).agg(agg_dict_semaine)
    df_par_semaine['Semaine'] = df_par_semaine['date_debut_semaine'].dt.strftime('%d-%m-%Y') + ' au ' + \
                                (df_par_semaine['date_debut_semaine'] + pd.Timedelta(days=6)).dt.strftime('%d-%m-%Y')
    
    cols_order = ['Semaine'] + [col for col in df_par_semaine.columns if col not in ['Semaine', 'date_debut_semaine']]
    df_par_semaine = df_par_semaine[cols_order]
    
    return df_par_semaine

def onglet_semaine(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        st.info("Aucune donnée disponible pour le filtre.")
        return pd.DataFrame()
    # 1. Convertir la colonne 'Semaine' en datetime pour s'assurer d'un tri correct.
    #    On utilise errors='coerce' pour mettre à NaT si le format n'est pas reconnu.
    #    Tu devras peut-être ajuster le format si la colonne n'est pas AAAA-MM-JJ.
    try:
        df['Semaine_dt'] = pd.to_datetime(df['Semaine'], format='%d/%m/%Y', errors='coerce')
        # Supposons le format est AAAA-MM-JJ. Si ton format est DD-MM-AAAA, ajoute dayfirst=True
        
        # 2. Trier le DataFrame sur cette nouvelle colonne de date
        df_sorted = df.sort_values(by='Semaine_dt', ascending=True)
        
        # 3. Extraire les semaines uniques et triées (en format string pour le multiselect)
        all_semaines_sorted = df_sorted['Semaine'].unique().tolist()
    except KeyError:
        # Si la colonne 'Semaine' n'existe pas ou si la conversion échoue de manière inattendue
        st.error("Erreur de colonne : la colonne 'Semaine' est manquante ou mal formatée pour le tri.")
        all_semaines_sorted = df['Semaine'].unique().tolist()
    # Déterminer la valeur par défaut : les 4 dernières semaines
    if len(all_semaines_sorted) > 6:
        default_selection = all_semaines_sorted[-6:]
    else:
        default_selection = all_semaines_sorted
    # Création du multiselect
    selected_semaines = st.multiselect(
        "Sélectionner les semaines",
        options=all_semaines_sorted, # La liste est maintenant triée chronologiquement
        default=default_selection
    )
    # Gestion du cas où AUCUNE SEMAINE n'est sélectionnée (Retourne le DataFrame complet)
    if not selected_semaines:
        st.warning("Aucune semaine n'a été sélectionnée. Affichage de toutes les données.")
        return df_sorted.drop(columns=['Semaine_dt']).copy() # On retourne le DF original sans la colonne temp
    # Filtrage (Si des semaines sont sélectionnées)
    filtered_df = df_sorted[df_sorted['Semaine'].isin(selected_semaines)].copy()
    # Nettoyage : On supprime la colonne temporaire de la sortie
    return filtered_df.drop(columns=['Semaine_dt'])

###Données GPS indiv###
def add_columns_session_rpe(df : pd.DataFrame) -> pd.DataFrame:
    # Remplacer les valeurs non numériques (comme 'None' ou celles créées par pd.merge) par 0
    df['Cardio'] = pd.to_numeric(df['Cardio'], errors='coerce').fillna(0)
    df['Muscu'] = pd.to_numeric(df['Muscu'], errors='coerce').fillna(0)
    df["s.Cardio"] = df['Field Time'] * df['Cardio']
    df["s.Muscu"] = df['Field Time'] * df['Muscu']
    df['Total Time'] = df['End Time'].apply(convert_to_minutes) - df['Start Time'].apply(convert_to_minutes)
    return df

def filtrer_dataframe_joueuse_indiv(df_gps, selection_annee, selection_type):
    if not selection_type: # Si rien n'est sélectionné, définir filtered_df
        filtered_df = df_gps.copy()
    else:
        masks = []
        if 'S' in selection_type:
            masks.append(df_gps['Activity Name'].str.startswith('S'))
        if 'C' in selection_type:
            masks.append(df_gps['Activity Name'].str.startswith('C'))
        if 'Réa' in selection_type:
            masks.append(df_gps['Activity Name'].str.startswith('Réa'))
        if 'M' in selection_type:
            masks.append(df_gps['Activity Name'].str.contains('-'))
        if masks : 
            combined_mask = pd.concat(masks, axis=1).any(axis=1) # Combinez les masques en une seule ligne avec un "OR" logique
            filtered_df = df_gps[combined_mask] # Appliquez le filtre
        else : 
            filtered_df = pd.DataFrame()
    if selection_annee and not filtered_df.empty:
        filtered_df = filtered_df[filtered_df['Saison'] == selection_annee]
    if not filtered_df.empty:
        filtered_df['Date'] = pd.to_datetime(filtered_df['Date'],format='%Y-%m-%d')
        filtered_df['Date'] = filtered_df['Date'].dt.strftime('%d-%m-%Y')
    return filtered_df

def choisir_une_joueuse_une_date(df: pd.DataFrame, key_prefix: str, default_player: str=None) -> pd.DataFrame:
    # 1. LOGIQUE DE SÉLECTION DE LA JOUEUSE
    df['Name'] = df['Name'].astype(str)
    list_joueuses = sorted(df['Name'].unique().tolist())
    if not list_joueuses:
        return pd.DataFrame()
    if default_player and default_player in list_joueuses:
        default_index = list_joueuses.index(default_player)
    else:
        default_index = 0
        
    def update_player_state():
        st.session_state['selected_player'] = st.session_state[f'{key_prefix}_joueuse_unique']
    selection_joueuse = st.selectbox(
        "Sélectionner une joueuse",
        options=list_joueuses,
        index=default_index,
        key=f'{key_prefix}_joueuse_unique',
        on_change=update_player_state
    )

    # 2. PRÉPARATION DU DATAFRAME FILTRÉ
    df_joueuse_indiv = df[df['Name'] == selection_joueuse].copy()
    df_joueuse_indiv['Date'] = pd.to_datetime(df_joueuse_indiv['Date'], dayfirst=True, errors='coerce')
    df_joueuse_indiv.dropna(subset=['Date'], inplace=True)
    
    # 3. DÉTERMINATION DES DATES MIN/MAX DISPONIBLES
    if not df_joueuse_indiv.empty:
        # Dates réelles du dataset filtré
        min_available_date = df_joueuse_indiv['Date'].min().date()
        max_available_date = df_joueuse_indiv['Date'].max().date()
    else:
        min_available_date = date.today()
        max_available_date = date.today()
        
    # 4. LOGIQUE DE PERSISTANCE DES DATES SÉLECTIONNÉES
    # 🔑 FIX : Définir 'today' ICI pour qu'elle soit accessible partout
    today = date.today()
    # --- Calcul pour la vue par défaut (6 DERNIÈRES SEMAINES) ---
    # Calcul de la fin de la semaine en cours (Dimanche)
    # Si Lundi = 0, Dimanche = 6.
    current_weekday = today.weekday()
    # Le nombre de jours jusqu'à dimanche (ou aujourd'hui si on est dimanche)
    days_until_sunday = (6 - current_weekday) 
    last_day_of_current_week = today + timedelta(days=days_until_sunday)
    # Calcul du début de la fenêtre de 6 semaines (Lundi)
    # On recule de 6 semaines complètes (6 * 7 jours) + le nombre de jours jusqu'au lundi précédent
    start_date_6_weeks_ago = today - timedelta(days=current_weekday + 6 * 7)
    
    # 🔑 Récupération/Initialisation de la date de DÉBUT mémorisée
    if 'selected_start_date' not in st.session_state:
        # 1. INITIALISATION PAR DÉFAUT (6 SEMAINES)
        # On borne la date de début à la date min disponible
        default_start = max(start_date_6_weeks_ago, min_available_date) 
    elif st.session_state['selected_start_date'] < min_available_date:
        # 2. OU si la date mémorisée est hors plage (trop ancienne)
        default_start = min_available_date
    else:
        # 3. OU utiliser la date mémorisée
        default_start = st.session_state['selected_start_date']
        
    # 🔑 Récupération/Initialisation de la date de FIN mémorisée
    if 'selected_end_date' not in st.session_state:
        # 1. INITIALISATION PAR DÉFAUT (SEMAINE EN COURS)
        # On borne la date de fin à la date max disponible
        default_end = min(last_day_of_current_week, max_available_date) 
    elif st.session_state['selected_end_date'] > max_available_date:
        # 2. OU si la date mémorisée est hors plage (trop récente)
        default_end = max_available_date
    else:
        # 3. OU utiliser la date mémorisée
        default_end = st.session_state['selected_end_date']

    # 5. SÉLECTEURS DE DATE
    # Fonction de rappel pour mémoriser les dates
    def update_date_state():
        st.session_state['selected_start_date'] = st.session_state[f'{key_prefix}_start_date_widget']
        st.session_state['selected_end_date'] = st.session_state[f'{key_prefix}_end_date_widget']

    col1, col2 = st.columns(2)
    with col1:
        date_debut = st.date_input(
            "Date de début", 
            value=default_start, # Utilise la date persistante ou la date min
            min_value=min_available_date,
            max_value=max_available_date,
            key=f'{key_prefix}_start_date_widget',
            on_change=update_date_state
        )
    with col2:
        date_fin = st.date_input(
            "Date de fin", 
            value=default_end, # Utilise la date persistante ou la date max
            min_value=min_available_date,
            max_value=max_available_date,
            key=f'{key_prefix}_end_date_widget',
            on_change=update_date_state
        )
    # 6. FILTRAGE FINAL
    # Conversion des dates pour le filtrage
    date_debut = pd.to_datetime(date_debut)
    date_fin = pd.to_datetime(date_fin)
    df_filtered_by_date = df_joueuse_indiv[
        (df_joueuse_indiv['Date'] >= date_debut) & (df_joueuse_indiv['Date'] <= date_fin)
    ].copy()
    df_filtered_by_date['Date'] = df_filtered_by_date['Date'].dt.date
    st.subheader(f"Données pour {selection_joueuse}")
    return df_filtered_by_date

def regrouper_par_semaine_civile(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    # On s'assure que la colonne 'Date' est au bon format
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    # On calcule explicitement la date du lundi pour chaque ligne de données
    jour_de_la_semaine = df['Date'].dt.weekday
    df['date_debut_semaine'] = df['Date'] - pd.to_timedelta(jour_de_la_semaine, unit='D')
    # On regroupe les données par joueuse et par le LUNDI de la semaine
    df_par_semaine = df.groupby(['Name', 'date_debut_semaine'], as_index=False).agg(agg_dict_indiv)
    # On formate la colonne 'Semaine' pour l'affichage
    df_par_semaine['Semaine'] = df_par_semaine['date_debut_semaine'].dt.strftime('%Y-%m-%d') + ' au ' + \
                               (df_par_semaine['date_debut_semaine'] + pd.Timedelta(days=6)).dt.strftime('%d-%m-%Y')
    # On réorganise les colonnes pour que 'Semaine' apparaisse après 'Name'
    cols_order = ['Name', 'Semaine'] + [col for col in df_par_semaine.columns if col not in ['Name', 'Semaine', 'date_debut_semaine']]
    df_par_semaine = df_par_semaine[cols_order]
    df_par_semaine['Semaine'] = df_par_semaine['Semaine'].str.split(' ').str[0]
    df_par_semaine = df_par_semaine.sort_values(by='Semaine', ascending=True)
    return df_par_semaine

@st.cache_data
def load_calendar_json(file_path):
    """Charge un fichier JSON de type {date_debut: nom_semaine} et le convertit en DataFrame."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Crée un DataFrame à partir du dictionnaire
        df = pd.DataFrame(list(data.items()), columns=['date_debut_str', 'nom'])
        # Convertit la colonne de date en type datetime
        df['date_debut'] = pd.to_datetime(df['date_debut_str'])
        return df[['date_debut', 'nom']]
    except FileNotFoundError:
        st.error(f"Erreur : le fichier {file_path} est introuvable.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Erreur lors du traitement du fichier JSON : {e}")
        return pd.DataFrame()

@st.cache_data
def load_and_merge_rpe(df_base):
    """
    Charge les données RPE depuis un fichier CSV et les fusionne
    avec un DataFrame de base.
    """
    try:
        df_rpe = pd.read_csv("data/rpe.csv")
    except (FileNotFoundError, pd.errors.EmptyDataError):
        df_rpe = pd.DataFrame(columns=['Name', 'Date', 'Jour semaine', 'Cardio', 'Muscu', 'Presence'])
    
    # Conversion de la colonne 'Date' dans les deux DataFrames pour la fusion
    df_base['Date'] = pd.to_datetime(df_base['Date'], dayfirst=True)
    df_rpe['Date'] = pd.to_datetime(df_rpe['Date'], dayfirst=True, errors='coerce')
    
    df_merged = pd.merge(df_base, df_rpe, on=['Name', 'Date', 'Jour semaine'], how='left')
    return df_merged

def generer_alertes_charge(df_charge_actuelle, df_charge_reference, cols_a_comparer, seuil_pourcentage=0.10):
    """
    Compare une charge observée (séance actuelle) à la charge moyenne de référence.
    """
    
    # 1. Jointure des deux DataFrames sur le nom (Name)
    df_compare = pd.merge(
        df_charge_actuelle, 
        df_charge_reference, 
        on='Name', 
        how='left', 
        suffixes=('_actuel', '_moyenne')
    )
    
    df_alertes = df_compare[['Name']].copy()
    
    # 2. Boucle de comparaison pour chaque métrique
    for col in cols_a_comparer:
        col_actuel = f'{col}_actuel'
        col_moyenne = f'{col}_moyenne'
        
        if col_actuel in df_compare.columns and col_moyenne in df_compare.columns:
            
            # Calcul des bornes de l'intervalle optimal
            df_compare['borne_sup'] = df_compare[col_moyenne] * (1 + seuil_pourcentage)
            df_compare['borne_inf'] = df_compare[col_moyenne] * (1 - seuil_pourcentage)
            
            # Création de la colonne de statut
            statut_alerte = []
            for index, row in df_compare.iterrows():
                val_act = row[col_actuel]
                val_moy = row[col_moyenne]
                borne_sup = row['borne_sup']
                borne_inf = row['borne_inf']
                
                if pd.isna(val_moy) or val_moy == 0:
                    statut = 'Nouvelle ou Moy. Inconnue'
                elif val_act > borne_sup:
                    statut = 'Surcharge'
                elif val_act >= borne_inf and val_act <= borne_sup:
                    statut = 'Optimal'
                else:
                    statut = 'Sous-charge'
                
                statut_alerte.append(statut)
            
            df_alertes[f'{col}'] = statut_alerte 
            
    return df_alertes


##Rapport de match###

#Séparer nom et mi-temps dans Name
def separer_nom_mi_temps(df: pd.DataFrame) -> pd.DataFrame:
    df[['Player Name','Mi-temps']] = df['Name'].str.split(' - ', expand=True)
    df.drop(columns='Name', inplace=True)
    return df

#Création de la colonne nbr effort VHSR
def creer_colonne_sprint(df: pd.DataFrame) -> pd.DataFrame:
    df['VHSR effort'] = df['VHSR + SPR effort'] - df['Sprint effort']
    df['Sprint (m)'] = df['SPR Total Distance (m)'] + df['SPR + Total Distance (m)']
    return df

#Supprimer les colonnes LSR, HSR
def supprimer_colonne(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=['LSR Total distance (m)', 'HSR Total distance (m)'], errors='ignore')
    return df

#Réorganiser les colonnes
def ordre_colonne(df: pd.DataFrame) -> pd.DataFrame:
    df = df[LIST_ORGANISATION_COLS_MATCH]
    return df

#Info type et saison de match
def add_columns_info_type_match_path(df: pd.DataFrame, type_match: str, saison_match: str) -> pd.DataFrame:
    df['Type match'] = type_match
    df['Saison'] = saison_match
    return df

#Fonctions de nettoyage du CSV match
def pipeline_nettoyage_match(df: pd.DataFrame, t_match: str = None, s_match: str = None) -> pd.DataFrame:
    df = renommer_colonne(df=df, col_rename=DICT_REPLACE_COLS)
    df = creer_colonne_sprint(df=df)
    df = separer_nom_mi_temps(df=df)
    df = supprimer_colonne(df=df)
    df = ordre_colonne(df=df)
    df = add_columns_info_type_match_path(df=df, type_match=t_match, saison_match=s_match)
    return df


##Importer les chemins de tous les fichiers matchs
def get_all_chemin_match(chemin: Union[str, None]=None) -> list:
    if chemin is None:
        liste_chemin = os.getcwd().split('\\')
        chemin = "\\".join(liste_chemin) + "\\data\\match\\"
    fichiers = os.listdir(chemin)
    total = []
    for i in fichiers:
        final_chemin = chemin + i
        total.append(final_chemin)
    return total

def recuperer_info_match_fichier(path_fichier: str) -> str:
    split = path_fichier.rsplit('.csv')
    split_2 = split[0].rsplit('_', maxsplit=2)
    info_type_match = split_2[-2]
    info_saison_match = split_2[-1]
    return info_type_match, info_saison_match

def recuperer_all_files_gps_match()-> pd.DataFrame:
    chemins = get_all_chemin_match()
    df_collectif = []
    for file in chemins:
        type_match, saison_match = recuperer_info_match_fichier(file) 
        data = pd.read_csv(file)
        data = pipeline_nettoyage_match(df=data, t_match=type_match, s_match=saison_match)
        df_collectif.append(data)
    df_collectif = pd.concat(df_collectif)
    df_collectif = df_collectif[cols_ref_match]
    return df_collectif

##Filtrer pour le rapport de match##
#Filtre année, type et match
def filtre_match(df: pd.DataFrame) -> pd.DataFrame:
    selection_annee = st.segmented_control("Année", ONGLET_GPS_SAISON, selection_mode="single", default=ONGLET_GPS_SAISON[0])
    selection_type = st.segmented_control("Type match", ONGLET_GPS_TYPE_MATCH, selection_mode="single", default=ONGLET_GPS_TYPE_MATCH[0])
    df_filtrer = df[(df['Saison'] == selection_annee) & (df['Type match'] == selection_type)]
    if df_filtrer.empty:
        st.warning("Aucun match ne correspond à la sélection")
        return pd.DataFrame()
    match_possible = sorted(df_filtrer['Activity Name'].unique())
    selection_match = st.segmented_control("Match", options=match_possible, default=match_possible[0])
    df_final = df_filtrer[df_filtrer['Activity Name'] == selection_match]
    return df_final

#Moyenne 
def dictionnaire_ref(list_indic: list=cols_ref_match[-12], params_agg: str='mean') -> dict:
    mean_ref = {}
    for col in list_indic:
        mean_ref[col] = params_agg
    return mean_ref

#Moyenne par joueuse
def filtrer_joueuse_match(df: pd.DataFrame, type_match: str='prepa'):
    df_ref_joueuse_match = df[df["Type match"] == type_match]
    return df_ref_joueuse_match
