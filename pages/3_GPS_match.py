import streamlit as st
import pandas as pd
from sections.menu.menu import custom_sidebar_menu
from sections.visualisation.viz import jauge_distance, jauge_intensite, jauge_nbr, jauge_barre
from sections.gps.section import gps_files_uploader
from sections.gps.pipeline import filtre_match
from sections.constantes import cols_ref_match, cols_de_regroupement, agg_dict, cols_de_regroupement_match

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

# Initialisation
if 'all_gps_match' in st.session_state:
    df_gps_match = st.session_state['all_gps_match']

# Sidebar
uploaded_files_session = gps_files_uploader('match')

# Main
st.title("Rapport de match")

# Filtre par match 
df_gps_filtre_match = filtre_match(df=df_gps_match)

#Filtre match
st.header("Données de match - Somme")
agg_dict_somme = {}
for col in cols_ref_match:
    if col in cols_de_regroupement_match:
        continue
    if col == 'V max':
        agg_dict_somme[col] = 'max'
    elif col == 'Meterage Per Minute':
        agg_dict_somme[col] = 'mean'
    else:
        agg_dict_somme[col] = 'sum'
df_ref_somme = df_gps_filtre_match.groupby(cols_de_regroupement_match).agg(agg_dict_somme).reset_index()
df_ref_somme['VHSR effort'] = df_ref_somme['VHSR + SPR effort'] - df_ref_somme['Sprint effort']
df_ref_somme = df_ref_somme.drop(['Player Name', 'Position Name', 'Mi-temps', 'Type match', 'Saison'], axis=1)

agg_dict_mean_max = {}
for col in df_gps_filtre_match.columns:
    if col in cols_de_regroupement_match:
        continue
    if col == 'V max':
        agg_dict_mean_max[col] = 'max'
    elif col == 'Meterage Per Minute':
        agg_dict_mean_max[col] = 'mean'
    else:
        agg_dict_mean_max[col] = 'sum'
df_ref_mean_max = df_gps_match.groupby(cols_de_regroupement_match).agg(agg_dict_mean_max).reset_index()
agg_avg_dict = {col: 'mean' for col in cols_ref_match[-11:] if col in df_ref_mean_max.columns}
agg_max_dict = {col: 'max' for col in cols_ref_match[-11:] if col in df_ref_mean_max.columns}
df_moyenne_totale = df_ref_mean_max.agg(agg_avg_dict)
df_max_total = df_ref_mean_max.agg(agg_max_dict)
df_resume = pd.DataFrame({'Moyenne': df_moyenne_totale, 'Max': df_max_total}).transpose()
df_resume['VHSR effort'] = df_resume['VHSR + SPR effort'] - df_resume['Sprint effort']

col1, col2, col3 = st.columns(3)

valeur_dt = df_ref_somme.loc[0, 'Total Distance (m)']
valeur_mean_dt = df_resume.loc['Moyenne', 'Total Distance (m)']
valeur_max_dt = df_resume.loc['Max', 'Total Distance (m)']

with col1:
    jauge_distance(
        valeur_match= valeur_dt,
        valeur_moyenne= valeur_mean_dt,
        valeur_max= valeur_max_dt,
        couleur_barre= 'gold',
        titre= 'Distance totale (m)'
    )

valeur_vhsr = df_ref_somme.loc[0, 'VHSR Total Distance (m)']
valeur_mean_vhsr = df_resume.loc['Moyenne', 'VHSR Total Distance (m)']
valeur_max_vhsr = df_resume.loc['Max', 'VHSR Total Distance (m)']

with col2:
    jauge_intensite(
            valeur_match= valeur_vhsr,
            valeur_moyenne= valeur_mean_vhsr,
            valeur_max= valeur_max_vhsr,
            couleur_barre= 'darkblue',
            titre= 'VHSR Distance (m)'
        )

valeur_sprint = df_ref_somme.loc[0, 'Sprint (m)']
valeur_mean_sprint = df_resume.loc['Moyenne', 'Sprint (m)']
valeur_max_sprint = df_resume.loc['Max', 'Sprint (m)']

with col3:
    jauge_intensite(
            valeur_match= valeur_sprint,
            valeur_moyenne= valeur_mean_sprint,
            valeur_max= valeur_max_sprint,
            couleur_barre= 'firebrick',
            titre= 'SPR Distance (m)'
        )
    
col1, col2, col3, col4, col5, col6 = st.columns(6)

nbr_vhsr = df_ref_somme.loc[0, 'VHSR effort']
nbr_mean_vhsr = df_resume.loc['Moyenne', 'VHSR effort']
nbr_max_vhsr = df_resume.loc['Max', 'VHSR effort']

with col3:
    jauge_nbr(
        valeur_match= nbr_vhsr,
        valeur_moyenne= nbr_mean_vhsr,
        valeur_max= nbr_max_vhsr,
        couleur_barre= 'darkblue',
        titre= 'Nombre VHSR'
    )

nbr_spr = df_ref_somme.loc[0, 'Sprint effort']
nbr_mean_spr = df_resume.loc['Moyenne', 'Sprint effort']
nbr_max_spr = df_resume.loc['Max', 'Sprint effort']

with col5:
    jauge_nbr(
        valeur_match= nbr_spr,
        valeur_moyenne= nbr_mean_spr,
        valeur_max= nbr_max_spr,
        couleur_barre= 'firebrick',
        titre= 'Nombre sprint'
    )

vmax = df_ref_somme.loc[0, 'V max']
mean_vmax = df_resume.loc['Moyenne', 'V max']
max_vmax = df_resume.loc['Max', 'V max']

with col6:
    jauge_barre(
        valeur_match= vmax,
        valeur_moyenne= mean_vmax,
        valeur_max= max_vmax,
        couleur_barre= 'firebrick',
        titre= 'V max'
    )


###Données de match sur 90'
st.markdown("---")
st.header('Données de match - 90mins')

groupe_90 = df_ref_somme 
col_a_diviser = [
    "Field Time",
    "Total Distance (m)",
    "VHSR Total Distance (m)",
    "SPR Total Distance (m)",
    "SPR + Total Distance (m)",
    "Sprint (m)",
    "VHSR + SPR effort",
    "VHSR effort",
    "Sprint effort",
    "Accel >2m.s²",
    "Decel >2m.s²"
]
for col in col_a_diviser:
    # On vérifie que la colonne existe avant de diviser pour éviter une erreur.
    if col in groupe_90.columns:
        groupe_90[col] = groupe_90[col] / 11

stat_90 = df_resume
col_a_diviser_2 = [
    "Total Distance (m)",
    "VHSR Total Distance (m)",
    "SPR Total Distance (m)",
    "SPR + Total Distance (m)",
    "Sprint (m)",
    "VHSR + SPR effort",
    "VHSR effort",
    "Sprint effort",
    "Accel >2m.s²",
    "Decel >2m.s²"
]
for col in col_a_diviser_2:
    # On vérifie que la colonne existe avant de diviser pour éviter une erreur.
    if col in groupe_90.columns:
        stat_90[col] = stat_90[col] / 11

col1, col2, col3 = st.columns(3)

valeur_dt = groupe_90.loc[0, 'Total Distance (m)']
valeur_mean_dt = stat_90.loc['Moyenne', 'Total Distance (m)']
valeur_max_dt = stat_90.loc['Max', 'Total Distance (m)']

with col1:
    jauge_distance(
        valeur_match= valeur_dt,
        valeur_moyenne= valeur_mean_dt,
        valeur_max= valeur_max_dt,
        couleur_barre= 'gold',
        titre= 'Distance totale (m)'
    )

valeur_vhsr = groupe_90.loc[0, 'VHSR Total Distance (m)']
valeur_mean_vhsr = stat_90.loc['Moyenne', 'VHSR Total Distance (m)']
valeur_max_vhsr = stat_90.loc['Max', 'VHSR Total Distance (m)']

with col2:
    jauge_intensite(
            valeur_match= valeur_vhsr,
            valeur_moyenne= valeur_mean_vhsr,
            valeur_max= valeur_max_vhsr,
            couleur_barre= 'darkblue',
            titre= 'VHSR Distance (m)'
        )

valeur_sprint = groupe_90.loc[0, 'Sprint (m)']
valeur_mean_sprint = stat_90.loc['Moyenne', 'Sprint (m)']
valeur_max_sprint = stat_90.loc['Max', 'Sprint (m)']

with col3:
    jauge_intensite(
            valeur_match= valeur_sprint,
            valeur_moyenne= valeur_mean_sprint,
            valeur_max= valeur_max_sprint,
            couleur_barre= 'firebrick',
            titre= 'SPR Distance (m)'
        )
    
col1, col2, col3, col4, col5, col6 = st.columns(6)

nbr_vhsr = groupe_90.loc[0, 'VHSR effort']
nbr_mean_vhsr = stat_90.loc['Moyenne', 'VHSR effort']
nbr_max_vhsr = stat_90.loc['Max', 'VHSR effort']

with col1:
    field_time = groupe_90.loc[0, 'Field Time']

    # Utilisation d'un conteneur pour encadrer le contenu
    with st.container(border=True): # border=True ajoute une belle bordure autour
        # Utiliser st.markdown pour un titre stylisé
        st.markdown("**⏱️ Temps effectif**", unsafe_allow_html=True)
        
        # Afficher la valeur principale en utilisant un titre de taille plus grande
        st.markdown(
            f"<h2 style='text-align: center; color: #1f77b4;'>{field_time:.1f} <small>min</small></h2>", 
            unsafe_allow_html=True
        )

with col3:
    jauge_nbr(
        valeur_match= nbr_vhsr,
        valeur_moyenne= nbr_mean_vhsr,
        valeur_max= nbr_max_vhsr,
        couleur_barre= 'darkblue',
        titre= 'Nombre VHSR'
    )

nbr_spr = groupe_90.loc[0, 'Sprint effort']
nbr_mean_spr = stat_90.loc['Moyenne', 'Sprint effort']
nbr_max_spr = stat_90.loc['Max', 'Sprint effort']

with col5:
    jauge_nbr(
        valeur_match= nbr_spr,
        valeur_moyenne= nbr_mean_spr,
        valeur_max= nbr_max_spr,
        couleur_barre= 'firebrick',
        titre= 'Nombre sprint'
    )

###
st.markdown("---")
st.write("Comparatif mi-temps")
#Par mi-temps
agg_dict_mi_temps = {}
for col in cols_ref_match:
    if col in cols_de_regroupement:
        continue
    if col == 'V max':
        agg_dict_mi_temps[col] = 'max'
    elif col == 'Meterage Per Minute':
        agg_dict_mi_temps[col] = 'mean'
    else:
        agg_dict_mi_temps[col] = 'sum'
df_ref_mi_temps = df_gps_filtre_match.groupby('Mi-temps').agg(agg_dict).reset_index()
st.dataframe(df_ref_mi_temps, hide_index=True)

####
st.markdown("---")
st.header("Données au poste")
#Par poste du match
agg_dict_poste = {}
for col in cols_ref_match:
    if col in cols_de_regroupement:
        continue
    if col == 'V max':
        agg_dict_poste[col] = 'max'
    elif col == 'Meterage Per Minute':
        agg_dict_poste[col] = 'mean'
    else:
        agg_dict_poste[col] = 'sum'
df_ref_poste = df_gps_filtre_match.groupby(cols_de_regroupement).agg(agg_dict_poste).reset_index()
df_ref_poste = df_ref_poste.groupby('Position Name').agg(agg_dict).reset_index()
df_ref_poste['VHSR effort'] = df_ref_poste['VHSR + SPR effort'] - df_ref_poste['Sprint effort']
df_ref_poste = df_ref_poste.drop(['SPR Total Distance (m)', 'SPR + Total Distance (m)', 'VHSR + SPR effort'], axis=1)
df_ref_poste = df_ref_poste[['Position Name', 'Total Distance (m)', 'VHSR Total Distance (m)', 'VHSR effort', 'Sprint (m)', 'Sprint effort', 'Accel >2m.s²', 'Decel >2m.s²', 'V max', 'Meterage Per Minute']]
st.dataframe(
    df_ref_poste.style
            .format(precision=1)
            .highlight_max(axis=0, props='font-weight: bold; background-color: #e0f7fa;'),
        use_container_width=True, # Important pour remplir la colonne
        hide_index=True
)

####
st.markdown("---")
st.header("Données joueuses")
#Le match actuel
df_ref = df_gps_filtre_match[cols_ref_match]

agg_dict_match = {}
for col in cols_ref_match:
    if col in cols_de_regroupement:
        continue
    if col == 'V max':
        agg_dict_match[col] = 'max'
    elif col == 'Meterage Per Minute':
        agg_dict_match[col] = 'mean'
    else:
        agg_dict_match[col] = 'sum'
df_ref_joueuses = df_gps_filtre_match.groupby(cols_de_regroupement).agg(agg_dict_match).reset_index()
df_ref_joueuses['VHSR effort'] = df_ref_joueuses['VHSR + SPR effort'] - df_ref_joueuses['Sprint effort']
df_ref_joueuses = df_ref_joueuses.drop(['Mi-temps', 'Activity Name', 'Position Name', 'Type match', 'Saison', 'SPR Total Distance (m)', 'SPR + Total Distance (m)', 'VHSR + SPR effort'], axis=1)
df_ref_joueuses = df_ref_joueuses[["Player Name", "Field Time", "Total Distance (m)", "VHSR Total Distance (m)", "Sprint (m)", 'VHSR effort', 'Sprint effort', "Accel >2m.s²", "Decel >2m.s²", "V max", "Meterage Per Minute"]]

#Meilleur match
agg_dict_best = {}
for col in cols_ref_match:
    if col in cols_de_regroupement:
        continue
    if col == 'V max':
        agg_dict_best[col] = 'max'
    elif col == 'Meterage Per Minute':
        agg_dict_best[col] = 'mean'
    else:
        agg_dict_best[col] = 'sum'
df_ref_max = df_gps_match.groupby(cols_de_regroupement).agg(agg_dict_best).reset_index()
df_meilleures_performances = df_ref_max.groupby('Player Name').agg(agg_dict).reset_index()
# **CLÉ CRITIQUE : STOCKAGE DANS SESSION_STATE**
st.session_state['df_best_match_all_players'] = df_meilleures_performances

joueuses_match = df_ref_joueuses['Player Name'].unique()
df_meilleures_performances_inmatch = df_meilleures_performances.copy()
df_meilleures_performances_inmatch = df_meilleures_performances_inmatch[df_meilleures_performances_inmatch['Player Name'].isin(joueuses_match)]
df_meilleures_performances_inmatch.loc[:, 'VHSR effort'] = df_meilleures_performances_inmatch['VHSR + SPR effort'] - df_meilleures_performances_inmatch['Sprint effort']
df_meilleures_performances_inmatch = df_meilleures_performances_inmatch.drop(['SPR Total Distance (m)', 'SPR + Total Distance (m)', 'VHSR + SPR effort'], axis=1)
df_meilleures_performances_inmatch = df_meilleures_performances_inmatch[["Player Name", "Total Distance (m)", "VHSR Total Distance (m)", "Sprint (m)", 'VHSR effort', 'Sprint effort', "Accel >2m.s²", "Decel >2m.s²", "V max", "Meterage Per Minute"]]

nbr_ligne = len(df_ref_joueuses)
hauteur_ligne = 36
hauteur_entete = 35
hauteur_dyna = (hauteur_ligne * nbr_ligne) + hauteur_entete


col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📊 Statistiques du match")
    st.dataframe(
        df_ref_joueuses.style
            .format(precision=1)
            .highlight_max(axis=0, props='font-weight: bold; background-color: #e0f7fa;'),
        height=hauteur_dyna,
        use_container_width=True, # Important pour remplir la colonne
        hide_index=True 
    )

with col2:
    st.markdown("### 📊 Meilleur match")
    st.dataframe(
        df_meilleures_performances_inmatch.style
            .format(precision=1)
            .highlight_max(axis=0, props='font-weight: bold; background-color: #e0f7fa;'),
        height=hauteur_dyna,
        use_container_width=True, # Important pour remplir la colonne
        hide_index=True 
    )