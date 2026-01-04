import streamlit as st
import pandas as pd
from sections.menu.menu import custom_sidebar_menu
from sections.gps.section import onglet_seance, gps_files_uploader
from sections.visualisation.viz import creer_graph_dt, creer_graph_vhsr, creer_graph_spr, creer_graph_accel_charge
from sections.gps.pipeline import onglet_semaine, add_columns_session_rpe, load_and_merge_rpe
from sections.constantes import ONGLET_GPS_SAISON, ONGLET_GPS_TYPE, LIST_ORGANISATION_COLS_SEANCE, cols_ref_seance, cols_de_regroupement_seance, agg_dict_seance, agg_dict_semaine

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

# initialisation 
if 'all_gps_session' in st.session_state:
    df_gps = st.session_state['all_gps_session']

# Sidebar
custom_sidebar_menu()
uploaded_files_session = gps_files_uploader('séance')

# Main
st.title("Données GPS - Groupe")

### Fusionner les données GPS avec le RPE ###
FILE_PATH = "data/rpe.csv"
# Utiliser la fonction pour charger et fusionner les données
df_gps_avec_rpe = load_and_merge_rpe(df_gps.copy())
df_gps_avec_rpe['Cardio'] = pd.to_numeric(df_gps_avec_rpe['Cardio'], errors='coerce')
df_gps_avec_rpe['Muscu'] = pd.to_numeric(df_gps_avec_rpe['Muscu'], errors='coerce')
# Ajouter les colonnes s.RPE
df_gps_avec_rpe = add_columns_session_rpe(df=df_gps_avec_rpe)

### Filtres de séance ###
st.write("Données de séance")

# 1. On filtre les données pour les séances ("S") et on exclut les gardiennes
# Utilisez df_gps_avec_rpe qui contient déjà les données RPE et s.RPE
analyse_seance = df_gps_avec_rpe.loc[
    (df_gps_avec_rpe['Activity Name'].str.startswith('S')) & 
    (df_gps_avec_rpe['Position Name'] != 'Gardienne') & 
    (df_gps_avec_rpe['Presence'] != 'P') &
    (df_gps_avec_rpe['Presence'] != 'R')
].copy()

# 2. On définit le dictionnaire d'agrégation
agg_dict_seance = {}
for col in cols_ref_seance:
    if col in cols_de_regroupement_seance:
        continue
    # On s'assure que les colonnes sont numériques pour les calculs de moyenne
    analyse_seance[col] = pd.to_numeric(analyse_seance[col], errors='coerce')
    agg_dict_seance[col] = 'mean'

# 3. On ajoute les colonnes RPE et s.RPE au dictionnaire d'agrégation
agg_dict_seance['Cardio'] = 'mean'
agg_dict_seance['Muscu'] = 'mean'
agg_dict_seance['s.Cardio'] = 'mean'
agg_dict_seance['s.Muscu'] = 'mean'

# 4. On s'assure que la colonne de regroupement n'est pas dans le dictionnaire
for col in cols_de_regroupement_seance:
    agg_dict_seance[col] = 'first'
if 'Activity Name' in agg_dict_seance:
    del agg_dict_seance['Activity Name']

# 5. On regroupe les données par nom de séance
analyse_seance = analyse_seance.groupby(['Activity Name']).agg(agg_dict_seance).reset_index()

# 6. On formate et on affiche le DataFrame
analyse_seance = analyse_seance[LIST_ORGANISATION_COLS_SEANCE]
analyse_seance['Date'] = pd.to_datetime(analyse_seance['Date'], format='%d-%m-%Y')
analyse_seance['Date'] = analyse_seance['Date'].dt.strftime('%Y-%m-%d')
onglet_seance(df=analyse_seance)

st.write("Suivi de données")
### Filtre données totales par jour ###

# Définir le dictionnaire d'agrégation pour le groupe
agg_dict_groupe = {}
for col in cols_ref_seance:
    if col in cols_de_regroupement_seance:
        continue
    agg_dict_groupe[col] = 'mean'

# Ajouter les colonnes RPE et s.RPE
agg_dict_groupe = agg_dict_seance.copy()
agg_dict_groupe['Cardio'] = 'mean'
agg_dict_groupe['Muscu'] = 'mean'
agg_dict_groupe['s.Cardio'] = 'mean'
agg_dict_groupe['s.Muscu'] = 'mean'
if 'Activity Name' in agg_dict_groupe:
    del agg_dict_groupe['Activity Name']

# Filtrer les gardiennes
df_gps_groupe = df_gps_avec_rpe.loc[df_gps_avec_rpe['Position Name'] != 'Gardienne']

# Appliquer les filtres "Saison" et "Type séance" sur le DataFrame complet
selection_annee = st.segmented_control("Saison", ONGLET_GPS_SAISON, selection_mode="single", default=ONGLET_GPS_SAISON[0])
selection_type = st.segmented_control("Type séance", ONGLET_GPS_TYPE, selection_mode="multi", default=[ONGLET_GPS_TYPE[0],ONGLET_GPS_TYPE[2]])

if not selection_type:
    filtered_df = df_gps_groupe.copy()
else:
    masks = []
    if 'S' in selection_type:
        masks.append(df_gps_groupe['Activity Name'].str.startswith('S'))
    if 'C' in selection_type:
        masks.append(df_gps_groupe['Activity Name'].str.startswith('C'))
    if 'Réa' in selection_type:
        masks.append(df_gps_groupe['Activity Name'].str.startswith('Réa'))
    if 'M' in selection_type:
        masks.append(df_gps_groupe['Activity Name'].str.contains('-'))
    if masks:
        combined_mask = pd.concat(masks, axis=1).any(axis=1)
        filtered_df = df_gps_groupe[combined_mask]
    if selection_annee:
        filtered_df = filtered_df[filtered_df['Saison'] == selection_annee]

# Regrouper les données filtrées par 'Activity Name'
if not filtered_df.empty:
    # On convertit toutes les colonnes qui seront agrégées en numérique,
    # y compris les colonnes GPS
    for col in agg_dict_groupe:
        if col not in ['Date', 'Jour semaine', 'Activity Name']:
            filtered_df[col] = pd.to_numeric(filtered_df[col], errors='coerce')
    
    df_groupe_final = filtered_df.groupby(['Activity Name']).agg(agg_dict_groupe).reset_index()
    df_groupe_final['Date'] = pd.to_datetime(df_groupe_final['Date'], format='%d-%m-%Y')
    df_groupe_final['Date'] = df_groupe_final['Date'].dt.strftime('%Y-%m-%d')
    df_groupe_final = df_groupe_final.sort_values(by='Date', ascending=False)
    df_groupe_final = df_groupe_final[LIST_ORGANISATION_COLS_SEANCE]

    st.write("Données filtrées :")
    st.dataframe(df_groupe_final, hide_index=True)

    if 'df_groupe_final' not in st.session_state:
        st.session_state['df_groupe_final'] = df_groupe_final
    else: 
        df_groupe_final = st.session_state['df_groupe_final']

else:
    st.write('Aucune séance ne correspond à la sélection')


###Regrouper par semaine d'entrainement
st.markdown("---")
st.subheader("Données GPS par semaine")

# 1. On crée un DataFrame qui ne contient que les séances et les matchs
mask_seance_match = (filtered_df['Activity Name'].str.startswith('S')) | (filtered_df['Activity Name'].str.contains('-'))
df_seance_match_uniquement = filtered_df[mask_seance_match].copy()

# 2. On vérifie que le DataFrame n'est pas vide avant de le traiter
if not df_seance_match_uniquement.empty:
    # --- AGRÉGATION INTERMÉDIAIRE PAR SÉANCE ---
    agg_dict_seance['Date'] = 'first'
    
    # On agrège les données individuelles par séance
    df_agreg_par_seance = df_seance_match_uniquement.groupby(['Activity Name'], as_index=False).agg(agg_dict_seance)

    # --- AGRÉGATION FINALE PAR SEMAINE ---
    # On prépare le DataFrame pour le regroupement par semaine
    df_agreg_par_seance['Date'] = pd.to_datetime(df_agreg_par_seance['Date'])
    jour_de_la_semaine = df_agreg_par_seance['Date'].dt.weekday
    df_agreg_par_seance['date_debut_semaine'] = df_agreg_par_seance['Date'] - pd.to_timedelta(jour_de_la_semaine, unit='D')
    
    # On effectue le regroupement par semaine et la somme
    df_par_semaine_groupe = df_agreg_par_seance.groupby(['date_debut_semaine'], as_index=False).agg(agg_dict_semaine)
    
    # On formate les colonnes pour l'affichage
    df_par_semaine_groupe['Semaine'] = df_par_semaine_groupe['date_debut_semaine'].dt.strftime('%d-%m-%Y') + ' au ' + \
                                (df_par_semaine_groupe['date_debut_semaine'] + pd.Timedelta(days=6)).dt.strftime('%d-%m-%Y')
    
    cols_order = ['Semaine'] + [col for col in df_par_semaine_groupe.columns if col not in ['Semaine', 'date_debut_semaine']]
    df_par_semaine_groupe = df_par_semaine_groupe[cols_order]

    if not df_par_semaine_groupe.empty:
        df_par_semaine_groupe = df_par_semaine_groupe.drop(['Cardio', 'Muscu'], axis=1)
        df_par_semaine_groupe = onglet_semaine(df=df_par_semaine_groupe)

        col1, col2, col3 = st.columns(3)

        with col1:
            creer_graph_dt(
                df=df_par_semaine_groupe,
                x_col='Semaine',
                y_col='Total Distance (m)',
                titre='Distance totale par semaine',
                couleur_barre='gold',
                x_label='Semaine',
                y_label='Total Distance (m)'
            )
        
        couleurs_spr = {
            'SPR Total Distance (m)': 'firebrick', # Un rouge clair ou marron
            'SPR + Total Distance (m)': 'red' # Un rouge plus pétant
        }

        with col3:
            creer_graph_spr(
            df= df_par_semaine_groupe, 
            x_col= 'Semaine', 
            y_cols= ['SPR Total Distance (m)', 'SPR + Total Distance (m)'],
            line_col='Sprint effort',
            titre= 'Distance en SPR par semaine', 
            couleur_barre=couleurs_spr,
            x_label='Semaine',
            y1_label='Distance (m)',
            y2_label='Nombre SPR'
            )

        with col2:
            creer_graph_vhsr(
            df=df_par_semaine_groupe,
            x_col='Semaine',
            y_col='VHSR Total Distance (m)',
            line_col='VHSR effort',
            titre='Distance VHSR par semaine',
            couleur_barre='blue',
            x_label='Semaine',
            y1_label='VHSR Total Distance (m)',
            y2_label='Nombre VHSR' 
            )

        col1, col2, col3 = st.columns(3)
        
        couleurs_accel = {
            'Accel >2m.s²': 'orange', 
            'Decel >2m.s²': 'green' 
        }

        with col1:
            creer_graph_accel_charge(
            df=df_par_semaine_groupe,
            x_col='Semaine',
            y_cols=['Accel >2m.s²', 'Decel >2m.s²'],
            titre='Nombre accel et decel par semaine',
            couleur_barre=couleurs_accel,
            x_label='Semaine',
            y_label='Unité arbitraire'
            )
        
        couleurs_accel = {
            's.Cardio': 'grey', 
            's.Muscu': '#FF1493' 
        }

        with col2:
            creer_graph_accel_charge(
            df=df_par_semaine_groupe,
            x_col='Semaine',
            y_cols=['s.Cardio', 's.Muscu'],
            titre='Charge int par semaine',
            couleur_barre=couleurs_accel,
            x_label='Semaine',
            y_label='Unité arbitraire'
            )

        with col3:
            df_par_semaine_groupe = df_par_semaine_groupe[['Semaine', 'Total Time', 'Field Time', 'V max']]
            st.dataframe(
                df_par_semaine_groupe.style
                    .format(precision=1)
                    .highlight_max(axis=0, props='font-weight: bold; background-color: #e0f7fa;'),
                use_container_width=True, # Important pour remplir la colonne
                hide_index=True 
            )

    else:
        st.write("Pas de données de séances ou de matchs pour cette période.")
else:
    st.write("Pas de données de séances ou de matchs pour cette période.")


