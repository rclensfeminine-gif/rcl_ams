import streamlit as st
import pandas as pd
import os
import numpy as np
import altair as alt
from sections.menu.menu import custom_sidebar_menu
from sections.gps.pipeline import filtre_match
from sections.constantes import cols_ref_match_indiv, cols_de_regroupement_indiv
from sections.visualisation.viz import creer_graph_poids, creer_graph_poids_blessure, creer_percent_MG, creer_MM, create_sauts_performance, create_CMJ_unilat_blessure, create_LHT_chart, create_combined_hop_chart, create_CMJ_unilat_perf, create_combined_hop_performance
from sections.joueuses.pipeline import creer_graph_comparatif_et_somme_plis, colorer_seuil_difference, colorer_seuil_ratio, colorer_seuil_ratio_fib_abd, style_performance_improvements, display_stats_joueuse_cards, get_player_benchmarks, calculate_position_benchmarks, get_poste_all_match, afficher_comparaison_match, afficher_radar_performance, get_player_all_match, get_player_summary_stats

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

# Définition du chemin du fichier d'identité
FICHIER_ID_JOUEUSES = 'data/identite.csv' 
DOSSIER_DATA = 'data/suivi'
FICHIER_ANTHROPO_FIXE = 'data/suivi/anthropo_fixes.csv' 
FICHIER_ANTHROPO_SUIVI = 'data/suivi/anthropo_suivi.csv' 
FICHIER_SUIVI_BLESSURES = 'data/suivi/blessures_suivi.csv'
FICHIER_SUIVI_TESTING = 'data/suivi/testing_suivi.csv'
FICHIER_ANTECEDENTS = 'data/suivi/antecedents.csv'
FICHIER_ISOCINETISME = 'data/suivi/isocinetisme.csv'
FICHIER_HOP_TEST = 'data/suivi/hop_test.csv'
FICHIER_SAUTS = 'data/suivi/sauts.csv'
FICHIER_DYNAMO = 'data/suivi/dynamo.csv'

# Pour éliminer les warnings
pd.set_option('future.no_silent_downcasting', True)

################################################
# 1. CHARGEMENT DES FICHIERS ET AUTRES FONCTIONS

@st.cache_data
def charger_liste_joueuses():
    """Charge le DataFrame d'identité des joueuses."""
    if not os.path.exists(FICHIER_ID_JOUEUSES):
        st.error("Le fichier d'identité des joueuses (data/identite.csv) est introuvable.")
        return pd.DataFrame()
        
    try:
        df = pd.read_csv(FICHIER_ID_JOUEUSES, encoding='utf-8')
        
        # S'assurer que les colonnes clés existent pour le sélecteur
        if 'Prénom' not in df.columns or 'NOM' not in df.columns:
            st.error("Les colonnes 'Prénom' ou 'NOM' sont manquantes dans le fichier d'identité.")
            return pd.DataFrame()
            
        # Créer la colonne de nom complet pour l'affichage dans le selectbox
        df['Nom Complet'] = df['Prénom'] + ' ' + df['NOM']
        return df
        
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier d'identité : {e}")
        return pd.DataFrame()

@st.cache_data
def charger_fichier_suivi(chemin_fichier):
    """
    Charge un fichier CSV de suivi, gère les erreurs, assure le format DataFrame, 
    et crée la colonne 'Nom Complet'.
    """
    if not os.path.exists(chemin_fichier):
        st.warning(f"Le fichier de suivi est introuvable : {chemin_fichier}")
        return pd.DataFrame()
        
    try:
        df = pd.read_csv(chemin_fichier, encoding='utf-8')
        
        # S'assurer que les colonnes clés existent pour le filtrage
        if 'Prénom' not in df.columns or 'NOM' not in df.columns:
            st.warning(f"Les colonnes 'Prénom' ou 'NOM' sont manquantes dans {chemin_fichier}.")
            return pd.DataFrame()
            
        # 🚨 AJOUT : Créer la colonne de nom complet pour uniformité
        df['Nom Complet'] = df['Prénom'] + ' ' + df['NOM']

        # Convertir la colonne de date si elle existe (essentiel pour les graphiques)
        if 'Date Test' in df.columns:
            df['Date Test'] = pd.to_datetime(df['Date Test'], errors='coerce')
        
        return df
            
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier {chemin_fichier}: {e}")
        return pd.DataFrame()
    
def filtrer_et_trier(df, nom_complet): # <-- La fonction prend maintenant 'nom_complet'
    """Filtre un DataFrame de suivi par joueuse et trie par date."""
    if df.empty:
        return pd.DataFrame()
    
    # 🚨 FILTRAGE SIMPLIFIÉ
    df_filtre = df[
        (df['Nom Complet'] == nom_complet) # <-- Utilisation du Nom Complet
    ].copy()
    
    if 'Date Test' in df_filtre.columns:
        df_filtre = df_filtre.sort_values(by='Date Test', ascending=False)

    df_filtre = df_filtre.drop(['Prénom', 'NOM', 'Nom Complet'], axis=1)
        
    return df_filtre

def filtrer_et_trier_match(df, nom_complet): # <-- La fonction prend maintenant 'nom_complet'
    """Filtre un DataFrame de suivi par joueuse et trie par date."""
    if df.empty:
        return pd.DataFrame()
    
    # 🚨 FILTRAGE SIMPLIFIÉ
    df_filtre = df[
        (df['Player Name'] == nom_complet) # <-- Utilisation du Nom Complet
    ].copy()

    df_filtre = df_filtre.drop(['Player Name'], axis=1)
        
    return df_filtre

# ----------------------------------------------------------------------
# 2. CHARGEMENT DE TOUS LES DATASETS
# ----------------------------------------------------------------------

df_identite = charger_liste_joueuses()
df_anthropo_fixe = charger_fichier_suivi(FICHIER_ANTHROPO_FIXE)
df_anthropo_suivi = charger_fichier_suivi(FICHIER_ANTHROPO_SUIVI) 
df_blessures = charger_fichier_suivi(FICHIER_SUIVI_BLESSURES) 
df_antecedents = charger_fichier_suivi(FICHIER_ANTECEDENTS) 
df_isocinetisme = charger_fichier_suivi(FICHIER_ISOCINETISME) 
df_hop_test = charger_fichier_suivi(FICHIER_HOP_TEST) 
df_sauts = charger_fichier_suivi(FICHIER_SAUTS) 
df_dynamo = charger_fichier_suivi(FICHIER_DYNAMO)

# Initialisation
if 'all_gps_match' in st.session_state:
    df_gps_match = st.session_state['all_gps_match']

df_gps_match.copy()
# ----------------------------------------------------------------------
# 3. SÉLECTION DE LA JOUEUSE
# ----------------------------------------------------------------------

if df_identite.empty:
    st.stop()

liste_noms = df_identite['Nom Complet'].sort_values().unique().tolist()
liste_noms.insert(0, "Sélectionner une joueuse...")

joueuse_selectionnee = st.sidebar.selectbox(
    "Choisissez la joueuse à analyser :",
    options=liste_noms,
    index=0 
)

# ----------------------------------------------------------------------
# 4. FILTRER PAR LA JOUEUSE SELECTIONNEE
# ----------------------------------------------------------------------

if joueuse_selectionnee == "Sélectionner une joueuse...":
    st.title("Suivi Individuel des Joueuses 📈")
    st.info("Veuillez sélectionner une joueuse dans le menu de gauche pour afficher son profil détaillé.")
    st.stop()


# Filtrer les données pour la joueuse sélectionnée
profil_joueuse = df_identite[df_identite['Nom Complet'] == joueuse_selectionnee].iloc[0]

df_anthropo_fixe_j = filtrer_et_trier(df_anthropo_fixe, joueuse_selectionnee)
df_anthropo_suivi_j = filtrer_et_trier(df_anthropo_suivi, joueuse_selectionnee)
df_blessures_j = filtrer_et_trier(df_blessures, joueuse_selectionnee)
df_antecedents_j = filtrer_et_trier(df_antecedents, joueuse_selectionnee)
df_isocinetisme_j = filtrer_et_trier(df_isocinetisme, joueuse_selectionnee)
df_hop_test_j = filtrer_et_trier(df_hop_test, joueuse_selectionnee)
df_sauts_j = filtrer_et_trier(df_sauts, joueuse_selectionnee)
df_dynamo_j = filtrer_et_trier(df_dynamo, joueuse_selectionnee)
df_gps_match_j = filtrer_et_trier_match(df_gps_match, joueuse_selectionnee)


df_anthropo_suivi_j = df_anthropo_suivi_j.sort_values(by='Date', ascending=True)
df_blessures_j = df_blessures_j.sort_values(by='Date blessure', ascending=True)
df_antecedents_j = df_antecedents_j.sort_values(by='Date blessure', ascending=True)
df_isocinetisme_j = df_isocinetisme_j.sort_values(by='Date Test', ascending=True)
df_hop_test_j = df_hop_test_j.sort_values(by='Date Test', ascending=True)
df_sauts_j = df_sauts_j.sort_values(by='Date Test', ascending=True)
df_dynamo_j = df_dynamo_j.sort_values(by='Date Test', ascending=True)

df_anthropo_suivi_j['∑ Plis'] = df_anthropo_suivi_j['Biceps'] + df_anthropo_suivi_j['Triceps'] + df_anthropo_suivi_j['Sous-Scap'] + df_anthropo_suivi_j['Sup-Illiaque'] + df_anthropo_suivi_j['Sub-Spinal'] + df_anthropo_suivi_j['Abdo'] + df_anthropo_suivi_j['Mollet']
df_anthropo_plis_j = df_anthropo_suivi_j.drop('Poids (kg)', axis=1)
df_anthropo_plis_j = df_anthropo_plis_j.dropna(subset=['Biceps'])
df_anthropo_plis_j["% MG"] = (df_anthropo_plis_j['Triceps'] + df_anthropo_plis_j['Sous-Scap'] + df_anthropo_plis_j['Sub-Spinal'] + df_anthropo_plis_j['Abdo'])*0.213+7.9
df_anthropo_plis_j["% MG"] = pd.to_numeric(df_anthropo_plis_j["% MG"], errors='coerce')
taille = df_anthropo_fixe_j["Taille (cm)"].iloc[0]
tour_poignet = df_anthropo_fixe_j["Tour Poignet (cm)"].iloc[0]
df_anthropo_plis_j["% MM"]  = 24.051-(0.095*df_anthropo_plis_j["Triceps"])-(0.138*profil_joueuse["Age"])+(5.86*0)-(0.145*df_anthropo_plis_j["Sous-Scap"])-(0.06*df_anthropo_plis_j["Sub-Spinal"])-(0.132*df_anthropo_plis_j["Biceps"])-(0.05*df_anthropo_suivi_j["Poids (kg)"])+(0.07*taille)+(1.893*tour_poignet)
df_anthropo_plis_j["MM (kg)"] = df_anthropo_suivi_j["Poids (kg)"] * df_anthropo_plis_j["% MM"] /100

colonnes_edition_dynamo = [
        'Date Test', 'Soléaire D', 'Soléaire G', 'Soléaire H barre', 'Sym soléaire',
        'Gastro D', 'Gastro G', 'Sym gastro',
        'Tibial post D', 'Tibial post G', 'Sym tibial post',
        'Fibulaire D', 'Fibulaire G', 'Sym fibulaire',
        'Ratio fibulaire / tibial post D', 'Ratio fibulaire / tibial post G',
        'Abducteur D', 'Abducteur G', 'Sym abducteur',
        'Adducteur D', 'Adducteur G', 'Sym adducteur',
        'Ratio ADD / ABD D', 'Ratio ADD / ABD G'
]

df_dynamo_j = df_dynamo_j[colonnes_edition_dynamo]
df_dynamo_j = df_dynamo_j.drop('Soléaire H barre', axis=1)

# ----------------------------------------------------------------------
# 5. PRESENTATION DE LA JOUEUSE
# ----------------------------------------------------------------------

st.title(f"Profil de {joueuse_selectionnee} 📈")
st.markdown("---")

# Affichage des informations de base
col1, col2, col3, col4 = st.columns(4)
col1.metric("Poste Principal", profil_joueuse['1er Poste'])
col2.metric("N° Maillot", int(profil_joueuse['N°']) if pd.notna(profil_joueuse['N°']) else "N/A")
col3.metric("Âge", f"{profil_joueuse['Age']} ans" if pd.notna(profil_joueuse['Age']) else "N/A")
col4.metric("Latéralité", profil_joueuse['Latéralité'] if pd.notna(profil_joueuse['Latéralité']) else "N/A")


# ----------------------------------------------------------------------
# 6. STRUCTURE DES ONGLET DETAILEES
# ----------------------------------------------------------------------

st.markdown("---")
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Performances", "Anthropométrie", "Prévention blessures", "Données de match", "Données de séance"]
)

###########################################################
# Performance 
###########################################################
with tab1:
    ########## Tests isocinétique
    st.subheader("Tests isocinétiques")
    df_isocinetisme_perf = df_isocinetisme_j.drop(['Dif Q60°', 'Dif IJ60°', 'Dif Q240°', 'Dif IJ240°', 'Dif IJExc', 'Ratio IJ/Q60° D', 'Ratio IJ/Q60° G', 'Ratio IJ/Q240° D', 'Ratio IJ/Q240° G', 'Ratio Mixte D', 'Ratio Mixte G'], axis=1)
    df_isocinetisme_perf['Date Test'] = pd.to_datetime(df_isocinetisme_perf['Date Test'], errors='coerce')
    df_isocinetisme_perf['Date Test'] = df_isocinetisme_perf['Date Test'].dt.date

    df_isocinetisme_perf_styled = style_performance_improvements(df_isocinetisme_perf)
    st.dataframe(df_isocinetisme_perf_styled, use_container_width=True, hide_index=True)

    ########## Tests dynamo
    st.markdown("---")
    st.subheader("Tests dynamo")
    df_dynamo_perf = df_dynamo_j.drop(['Sym soléaire', 'Sym gastro', 'Sym tibial post', 'Sym fibulaire', 'Sym abducteur', 'Sym adducteur', 'Ratio fibulaire / tibial post G', 'Ratio fibulaire / tibial post D', 'Ratio ADD / ABD D', 'Ratio ADD / ABD G'], axis=1)
    df_dynamo_perf['Date Test'] = pd.to_datetime(df_dynamo_perf['Date Test'], errors='coerce')
    df_dynamo_perf['Date Test'] = df_dynamo_perf['Date Test'].dt.date    
    
    df_dynamo_perf_styled = style_performance_improvements(df_dynamo_perf)
    st.dataframe(df_dynamo_perf_styled, use_container_width=True, hide_index=True)

    ########## Tests de sauts et hop
    # 1. Préparation sauts et hop test
    df_hop_test_perf = df_hop_test_j.drop(['SHT D1', 'SHT D2', 'SHT D3', 'Nbr SHT D', 'SHT G1', 'SHT G2', 'SHT G3', 'Nbr SHT G', 'THT D1', 'THT D2', 'THT D3', 'Nbr THT D', 'THT G1', 'THT G2', 'THT G3', 'Nbr THT G', 'CHT D1', 'CHT D2', 'CHT D3', 'Nbr CHT D', 'CHT G1', 'CHT G2', 'CHT G3', 'Nbr CHT G', 'Sym SHT', 'Sym THT', 'Sym CHT'], axis=1)
    df_hop_test_perf['Date Test'] = pd.to_datetime(df_hop_test_perf['Date Test'], errors='coerce')
    df_hop_test_perf['Date Test'] = df_hop_test_perf['Date Test'].dt.date

    df_sauts_perf = df_sauts_j.drop(['CMJ 1', 'CMJ 2', 'CMJ 3', 'CMJ Bras 1', 'CMJ Bras 2', 'CMJ Bras 3', 'CMJ 1J D1', 'CMJ 1J D2', 'CMJ 1J D3', 'CMJ 1J G1', 'CMJ 1J G2', 'CMJ 1J G3', 'SRJT 5 Mean 1', 'SRJT 5 Mean 2', 'SRJT 5 Mean 3', 'SRJT 5 RSI 1', 'SRJT 5 RSI 2', 'SRJT 5 RSI 3'], axis=1)
    # Assurer le chargement et le nettoyage des colonnes essentielles
    if 'Date Test' in df_sauts_perf.columns:
        df_sauts_perf['Date Test'] = pd.to_datetime(
            df_sauts_perf['Date Test'], 
            errors='coerce'
        )
        df_sauts_perf.dropna(subset=['Date Test'], inplace=True)
        
        performance_cols = [
            'Max CMJ', 'Max CMJ Bras', 'Max CMJ 1J D', 'Max CMJ 1J G', 
            'Max SRJT 5 Mean', 'Max SRJT 5 RSI'
        ]
        for col in performance_cols:
            if col in df_sauts_perf.columns:
                df_sauts_perf[col] = pd.to_numeric(df_sauts_perf[col], errors='coerce')

    # ----------------------------------------------------------------------
    # 2. CRÉATION ET AFFICHAGE DES GRAPHIQUES (DISPOSITION VERTICALE MAINTENUE)
    # ----------------------------------------------------------------------
    st.markdown("---")
    st.subheader("Tests de sauts")

    # 2 colonnes pour les 2 premiers graphiques
    col1, col2 = st.columns(2) 

    # --- Graphique 1: Max CMJ vs Max CMJ Bras (1 axe Y) ---
    y_cols_primary_1 = ['Max CMJ', 'Max CMJ Bras']
    fig1 = create_sauts_performance(
        df_sauts_perf, 
        y_cols_primary_1, 
        "CMJ performance",
        yaxis_primary_label="Hauteur (cm)"
    )
    with col1:
        if fig1:
            st.plotly_chart(fig1, use_container_width=True)

    # --- Graphique 3: Max SRJT 5 Mean (Y1) et Max SRJT 5 RSI (Y2) (Pleine largeur) ---
    y_cols_primary_3 = ['Max SRJT 5 Mean'] 
    y_cols_secondary_3 = ['Max SRJT 5 RSI']  

    fig3 = create_sauts_performance(
        df_sauts_perf, 
        y_cols_primary_3, 
        "Moyenne hauteur et RSI au SRJT 5",
        y_cols_secondary=y_cols_secondary_3,
        yaxis_primary_label="Hauteur (cm)", 
        yaxis_secondary_label="RSI"
    )
    with col2:
        if fig3:
            st.plotly_chart(fig3, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    # CMJ unilatéral (CMJ 1J)
    bar_cols_perf_target = ['Max CMJ 1J D', 'Max CMJ 1J G']

    fig_CMJ_perf = create_CMJ_unilat_perf(
        df_sauts_perf, 
        bar_cols_perf_target,
        "CMJ unilatérale (CMJ 1J)",
        y1_label="Hauteur (cm)"
    )

    with col1:    
        if fig_CMJ_perf:
            st.plotly_chart(fig_CMJ_perf, use_container_width=True)


    # Single Hop Test (SHT)
    # Les noms de colonnes doivent être ceux exacts de votre DataFrame
    SHT_cols = ['Max SHT D', 'Max SHT G']

    fig_sht = create_combined_hop_performance(
        df=df_hop_test_perf, 
        bar_cols=SHT_cols, 
        test_name='Max SHT', 
        test_name_legende='SHT', 
        y1_unit="cm"
    )

    # Triple Hop Test (THT)
    THT_cols = ['Max THT D', 'Max THT G']

    fig_tht = create_combined_hop_performance(
        df=df_hop_test_perf, 
        bar_cols=THT_cols, 
        test_name='Max THT', 
        test_name_legende='THT', 
        y1_unit="cm"
    )

    with col2:
        if fig_sht:
            st.plotly_chart(fig_sht, use_container_width=True)
        
    with col3:
        if fig_tht:
            st.plotly_chart(fig_tht, use_container_width=True)


###########################################################
# Anthropo 
###########################################################
with tab2:

    if not df_anthropo_suivi_j.empty:
        creer_graph_poids(
            df=df_anthropo_suivi_j,
            x_col='Date',
            y_col='Poids (kg)',
            titre=f"Évolution du poids"
        )

        creer_graph_comparatif_et_somme_plis(
            df_anthropo_plis_j, 
            f"Évolution des plis cutanés"
        )

    
        col1, col2 = st.columns(2)
        with col1:
            # Assurez-vous que la colonne est bien numérique pour calculer le min
            df_anthropo_plis_j['% MG'] = pd.to_numeric(df_anthropo_plis_j['% MG'], errors='coerce')

            # Calculer le minimum de la masse grasse et enlever une marge de 3
            min_mg = df_anthropo_plis_j['% MG'].min()
            y_min_mg_ajuste = min_mg - 2

            if y_min_mg_ajuste < 0:
                y_min_mg_ajuste = 0.0 # Un pourcentage ne peut pas être négatif

            creer_percent_MG(
            df=df_anthropo_plis_j,
            x_col='Date',
            y_col='% MG',
            titre=f"Évolution % de masse grasse",
            y_range_min= y_min_mg_ajuste
        )  
            
        with col2:
            # Assurez-vous que la colonne est bien numérique pour calculer le min
            df_anthropo_plis_j['MM (kg)'] = pd.to_numeric(df_anthropo_plis_j['MM (kg)'], errors='coerce')

            # Calculer le minimum de la masse grasse et enlever une marge de 3
            min_mm = df_anthropo_plis_j['MM (kg)'].min()
            y_min_mm_ajuste = min_mm - 1

            if y_min_mm_ajuste < 0:
                y_min_mm_ajuste = 0.0 # Un pourcentage ne peut pas être négatif

            creer_MM(
            df=df_anthropo_plis_j,
            x_col='Date',
            y_col='MM (kg)',
            titre=f"Évolution de la masse musculaire",
            y_range_min= y_min_mm_ajuste
        ) 

    else:
        st.info("Pas assez de données de poids enregistrées pour cette joueuse pour afficher le graphique.")
    


###########################################################
# Préventions de blessures
###########################################################

with tab3:
    ###### Tests ISOCINETIQUES
    if 'Date Test' in df_isocinetisme_j.columns:
        try:
            # Convertir la colonne en datetime
            df_isocinetisme_j['Date Test'] = pd.to_datetime(df_isocinetisme_j['Date Test'], errors='coerce')
            # Conserver uniquement la partie date (format YYYY-MM-DD)
            df_isocinetisme_j['Date Test'] = df_isocinetisme_j['Date Test'].dt.date
        except Exception as e:
            st.error(f"Erreur lors du formatage de la colonne 'Date Test': {e}")

    # Assurez-vous que les colonnes numériques sont bien de type numérique
    for col in df_isocinetisme_j.columns:
        if 'Dif' in col or 'Ratio' in col:
            df_isocinetisme_j[col] = pd.to_numeric(df_isocinetisme_j[col], errors='coerce')

    # ----------------------------------------------------------------------
    # 1. DÉFINITION DE LA LOGIQUE DE STYLE
    # ----------------------------------------------------------------------

    # Liste des colonnes de différence à colorer, basée sur le nommage exact de votre CSV
    # Identifier les colonnes à arrondir
    COLONNES_PUISSANCE = [col for col in df_isocinetisme_j.columns if ('Q' in col or 'IJ' in col) and not 'Ratio' in col]
    COLONNES_DIFFERENCE_RATIO = [col for col in df_isocinetisme_j.columns if 'Ratio' in col]

    COLONNES_DIFFERENCE = [
        'Dif Q60°', 
        'Dif IJ60°', 
        'Dif Q240°', 
        'Dif IJ240°', 
        'Dif IJExc'
    ] 

    COLONNES_RATIO = [
        'Ratio IJ/Q60° D', 
        'Ratio IJ/Q60° G', 
        'Ratio IJ/Q240° D', 
        'Ratio IJ/Q240° G', 
        'Ratio Mixte D', 
        'Ratio Mixte G'
    ]

    # ----------------------------------------------------------------------
    # 2. APPLICATION DU STYLE ET AFFICHAGE
    # ----------------------------------------------------------------------

    # 1. Créer une liste unique de toutes les colonnes à styliser
    colonnes_a_styliser = COLONNES_DIFFERENCE + COLONNES_RATIO
    colonnes_existantes = [col for col in colonnes_a_styliser if col in df_isocinetisme_j.columns]

    # Initialiser le DataFrame stylisé avec le DataFrame original
    # 💡 IMPORTANT : Ceci doit se faire APRÈS l'arrondi des données.
    df_isocinetisme_style = df_isocinetisme_j.copy().style 

    if colonnes_existantes:
        # arrondi
        format_dict = {col: "{:.0f}" for col in COLONNES_PUISSANCE if col in df_isocinetisme_j.columns}
        format_dict.update({col: "{:.2f}" for col in COLONNES_DIFFERENCE_RATIO if col in df_isocinetisme_j.columns})

        # Appliquer le format d'affichage
        df_isocinetisme_style = df_isocinetisme_style.format(format_dict)

        # Appliquer le style de DIFFERENCE
        dif_colonnes_existantes = [col for col in COLONNES_DIFFERENCE if col in df_isocinetisme_j.columns]
        if dif_colonnes_existantes:
            df_isocinetisme_style = df_isocinetisme_style.apply(
                lambda x: x.apply(colorer_seuil_difference), 
                subset=dif_colonnes_existantes
            )

        # Appliquer le style de RATIO
        ratio_colonnes_existantes = [col for col in COLONNES_RATIO if col in df_isocinetisme_j.columns]
        if ratio_colonnes_existantes:
            # On appelle la fonction directement et on force l'application ligne par ligne (axis=1)
            df_isocinetisme_style = df_isocinetisme_style.apply(
                colorer_seuil_ratio, 
                subset=ratio_colonnes_existantes,
                axis=1 # Ceci est CRUCIAL pour que la fonction reçoive une ligne (Series) et non une cellule unique
            )

        # 3. Afficher le DataFrame stylisé dans Streamlit
        st.subheader("Tests isocinétiques")
        st.dataframe(df_isocinetisme_style, hide_index=True)

    else:
        st.warning("Aucune des colonnes de différence ou de ratio isocinétique n'a été trouvée pour l'application du style.")
        st.dataframe(df_isocinetisme_j)

    
    ###### Tests DYNAMO
    if 'Date Test' in df_dynamo_j.columns:
        try:
            # Convertir la colonne en datetime
            df_dynamo_j['Date Test'] = pd.to_datetime(df_dynamo_j['Date Test'], errors='coerce')
            # Conserver uniquement la partie date (format YYYY-MM-DD)
            # Note: ceci renvoie un objet python 'date' qui s'affiche bien
            df_dynamo_j['Date Test'] = df_dynamo_j['Date Test'].dt.date
        except Exception as e:
            st.error(f"Erreur lors du formatage de la colonne 'Date Test': {e}")


    # ----------------------------------------------------------------------
    # 1. DÉFINITION DE LA LOGIQUE DE STYLE
    # ----------------------------------------------------------------------

    # Liste des colonnes de force brutes (D ou G)
    COLONNES_FORCE_BRUTE = [
        col for col in df_dynamo_j.columns 
        if ('D' in col or 'G' in col) and not ('Ratio' in col or 'Sym' in col)
    ]

    COLONNES_SYMETRIE = [
        'Sym soléaire',
        'Sym gastro', 
        'Sym tibial post', 
        'Sym fibulaire', 
        'Sym abducteur', 
        'Sym adducteur'
    ] 

    COLONNES_RATIO_FIB_TIB = [
        'Ratio fibulaire / tibial post D', 
        'Ratio fibulaire / tibial post G' 
    ]

    COLONNES_RATIO_ADD_ABD = [
        'Ratio ADD / ABD D', 
        'Ratio ADD / ABD G'
    ]

    # ----------------------------------------------------------------------
    # 2. APPLICATION DU STYLE ET AFFICHAGE
    # ----------------------------------------------------------------------

    # 1. Créer une liste unique de toutes les colonnes à styliser
    colonnes_a_styliser = COLONNES_SYMETRIE + COLONNES_RATIO_FIB_TIB + COLONNES_RATIO_ADD_ABD
    colonnes_existantes = [col for col in colonnes_a_styliser if col in df_dynamo_j.columns]

    # Initialiser le DataFrame stylisé avec le DataFrame original
    df_dynamo_style = df_dynamo_j.copy().style 

    if colonnes_existantes:
        format_dict = {}
        
        # ------------------------------------------------------------------
        # Construction du format_dict - Assurer que 'Date Test' est ignorée
        # ------------------------------------------------------------------
        
        # 1. Forces brutes (D/G) - Assumons 1 décimale (comme dans le st.column_config)
        force_brute_cols = [col for col in COLONNES_FORCE_BRUTE if col in df_dynamo_j.columns]
        format_dict.update({col: "{:.1f}" for col in force_brute_cols})
        
        # 2. Symétrie (0 décimale pour les pourcentages de symétrie)
        symetrie_cols_existantes = [col for col in COLONNES_SYMETRIE if col in df_dynamo_j.columns]
        format_dict.update({col: "{:.0f}" for col in symetrie_cols_existantes})
        
        # 3. Ratios (2 décimales)
        ratio_cols_existantes = [
            col for col in COLONNES_RATIO_FIB_TIB + COLONNES_RATIO_ADD_ABD 
            if col in df_dynamo_j.columns
        ]
        format_dict.update({col: "{:.2f}" for col in ratio_cols_existantes})

        # 4. EXCLUSION EXPLICITE DE 'Date Test' si elle est dans le dictionnaire par erreur
        if 'Date Test' in format_dict:
            del format_dict['Date Test']

        # Appliquer le format d'affichage
        df_dynamo_style = df_dynamo_style.format(format_dict)

        # ----------------------------------------------------
        # Application des styles conditionnels
        # ----------------------------------------------------
        
        # Appliquer le style de DIFFERENCE (Symétrie)
        symetrie_cols_existantes = [col for col in COLONNES_SYMETRIE if col in df_dynamo_j.columns]
        if symetrie_cols_existantes:
            df_dynamo_style = df_dynamo_style.apply(
                lambda x: x.map(colorer_seuil_difference), 
                subset=symetrie_cols_existantes,
                axis=0
            )

        # Appliquer le style de RATIO (Fib/Tib)
        ratio_fib_colonnes_existantes = [col for col in COLONNES_RATIO_FIB_TIB if col in df_dynamo_j.columns]
        if ratio_fib_colonnes_existantes:
            df_dynamo_style = df_dynamo_style.apply(
                colorer_seuil_ratio_fib_abd, 
                subset=ratio_fib_colonnes_existantes,
                axis=1 
            )

        # Appliquer le style de RATIO (ADD/ABD)
        ratio_abd_colonnes_existantes = [col for col in COLONNES_RATIO_ADD_ABD if col in df_dynamo_j.columns]
        if ratio_abd_colonnes_existantes:
            df_dynamo_style = df_dynamo_style.apply(
                colorer_seuil_ratio_fib_abd, 
                subset=ratio_abd_colonnes_existantes,
                axis=1 
            )
        # 3. Afficher le DataFrame stylisé dans Streamlit
        st.markdown("---")
        st.subheader("Tests dynamo")
        st.dataframe(df_dynamo_style, hide_index=True)

    ###### Tests SAUTS et HOP 
    # Sauts
    df_sauts_blessures = df_sauts_perf.drop(['Max CMJ', 'Max CMJ Bras', 'Max SRJT 5 Mean', 'Max SRJT 5 RSI', 'Date Label'], axis=1)
    df_sauts_blessures['Sym D/G'] = abs((df_sauts_blessures['Max CMJ 1J D']-df_sauts_blessures['Max CMJ 1J G'])/df_sauts_blessures['Max CMJ 1J D']) *100
    if 'Date Test' in df_sauts_blessures.columns:
        df_sauts_blessures['Date Test'] = pd.to_datetime(
            df_sauts_blessures['Date Test'], 
            errors='coerce'
        )
        df_sauts_blessures.dropna(subset=['Date Test'], inplace=True)
            
        performance_cols = [
            'Max CMJ 1J D', 'Max CMJ 1J G', 'Sym D/G'
        ]
        for col in performance_cols:
            if col in df_sauts_blessures.columns:
                df_sauts_blessures[col] = pd.to_numeric(df_sauts_blessures[col], errors='coerce')

    # Hop tests
    df_hop_test_blessure = df_hop_test_j.drop(['SHT D1', 'SHT D2', 'SHT D3', 'Nbr SHT D', 'SHT G1', 'SHT G2', 'SHT G3', 'Nbr SHT G', 'THT D1', 'THT D2', 'THT D3', 'Nbr THT D', 'THT G1', 'THT G2', 'THT G3', 'Nbr THT G', 'CHT D1', 'CHT D2', 'CHT D3', 'Nbr CHT D', 'CHT G1', 'CHT G2', 'CHT G3', 'Nbr CHT G'], axis=1)
    df_hop_test_blessure['Date Test'] = pd.to_datetime(df_hop_test_blessure['Date Test'], errors='coerce')
    df_hop_test_blessure['Date Test'] = df_hop_test_blessure['Date Test'].dt.date
    df_hop_test_blessure['Sym LHT'] = abs((df_hop_test_blessure['LHT D']-df_hop_test_blessure['LHT G'])/df_hop_test_blessure['LHT G'])*100

    if 'Date Test' in df_hop_test_blessure.columns:
        df_hop_test_blessure['Date Test'] = pd.to_datetime(
            df_hop_test_blessure['Date Test'], 
            errors='coerce'
        )
        df_hop_test_blessure.dropna(subset=['Date Test'], inplace=True)
        df_hop_test_blessure.sort_values(by='Date Test', inplace=True)

    
    # ----------------------------------------------------------------------
    # AFFICHAGE DE TOUS LES GRAPHIQUES DE SAUTS
    # ----------------------------------------------------------------------
    st.markdown("---")
    st.subheader("Tests de sauts")

    charts_config = [
        {
            "test_name": "Single Hop Test (SHT)",
            "test_name_legende": "SHT",
            "bar_cols": ['Max SHT D', 'Max SHT G'],
            "point_col": 'Sym SHT',
            "y1_unit": "cm"
        },
        {
            "test_name": "Triple Hop Test (THT)",
            "test_name_legende": "THT",
            "bar_cols": ['Max THT D', 'Max THT G'],
            "point_col": 'Sym THT',
            "y1_unit": "cm"
        },
        {
            "test_name": "Crossover Hop Test (CHT)",
            "test_name_legende": "CHT",
            "bar_cols": ['Max CHT D', 'Max CHT G'],
            "point_col": 'Sym CHT',
            "y1_unit": "cm"
        },
    ]

    cols = st.columns(3)
        
    for i, config in enumerate(charts_config):
        with cols[i % 3]:
            fig_hop = create_combined_hop_chart(
                df_hop_test_blessure,
                config["bar_cols"],
                config["point_col"],
                config["test_name"],
                config["test_name_legende"],
                config["y1_unit"]
            )
            if fig_hop:
                st.plotly_chart(fig_hop, use_container_width=True)

    
    
    col1, col2 = st.columns(2)
    with col2:
        bar_cols_target = ['Max CMJ 1J D', 'Max CMJ 1J G']
        point_col_target = 'Sym D/G'

        fig_final = create_CMJ_unilat_blessure(
            df_sauts_blessures, 
            bar_cols_target, 
            point_col_target,
            "CMJ unilatérale (CMJ 1J)",
            y1_label="Hauteur (cm)",
            y2_label="Asymétrie D/G (%)"
        )
        
        if fig_final:
            st.plotly_chart(fig_final, use_container_width=True)

    with col1:
        fig_LHT = create_LHT_chart(df_hop_test_blessure)
            
        if fig_LHT:
            st.plotly_chart(fig_LHT, use_container_width=True)

    
    ###### BLESSURES ET ANTECEDENTS    
    # Si df_blessures_j et df_antecedents_j sont vides, on crée un DF vide pour éviter l'erreur.
    if df_blessures_j.empty and df_antecedents_j.empty:
        st.info(f"Aucune donnée de blessure ou d'antécédent trouvée pour {joueuse_selectionnee}.")
        
    # ----------------------------------------------------------------------
    # Fonction Utile : Générer des Ticks Entiers pour l'axe X
    # ----------------------------------------------------------------------
    def generate_integer_ticks(df, column_name):
        """
        Génère une liste de ticks entiers lisibles (0, 1, 2, 3, ...) 
        pour un axe de graphique basé sur la valeur maximale de la colonne.
        """
        if df.empty:
            return [0]
        
        max_val = df[column_name].max()
        
        # S'assurer que le max est au moins 1 pour avoir une échelle minimale
        max_val = max(1, max_val)
        
        # Créer une liste de [0, 1, ..., max_val]
        # np.arange(start, stop + 1, step)
        ticks = np.arange(0, max_val + 1, 1).tolist()
        return ticks


    # ----------------------------------------------------------------------
    # 1. Unification et Préparation Finale
    # ----------------------------------------------------------------------

    # Concaténation des deux DataFrames
    df_profil = pd.concat([df_blessures_j, df_antecedents_j], ignore_index=True)

    # Convertir la date (pour tri chronologique si besoin)
    df_profil['Date blessure'] = pd.to_datetime(
        df_profil['Date blessure'], errors='coerce'
    )
    
    # ----------------------------------------------------------------------
    # CORRECTION : Nettoyage sélectif des valeurs manquantes (fillna)
    # ----------------------------------------------------------------------

    # Identifier les colonnes de type 'object' (chaînes de caractères)
    string_cols = df_profil.select_dtypes(include=['object']).columns

    # Remplacer les valeurs manquantes (NaN) par 'Non spécifié' SEULEMENT dans les colonnes de texte
    for col in string_cols:
        df_profil[col] = df_profil[col].fillna('Non spécifié')

    # S'assurer que les chaînes vides ou 'None' sont également traitées dans les colonnes de texte
    df_profil.replace({'None': 'Non spécifié', '': 'Non spécifié'}, inplace=True)


    # ----------------------------------------------------------------------
    # 2. Visualisations pour le Profil Individuel
    # ----------------------------------------------------------------------
    st.markdown("---")

    
    # --- A. Répartition par Localisation ---

    st.subheader("Localisation des blessures")

    df_loc_count = df_profil.groupby(
        ['Localisation']
    ).size().reset_index(name='Nombre de Cas')

    # Générer les Ticks pour le Tableau 1
    loc_ticks = generate_integer_ticks(df_loc_count, 'Nombre de Cas')


    # Utilisation de la Localisation pour la couleur, car nous n'avons plus de Statut
    chart_loc = alt.Chart(df_loc_count).mark_bar().encode(
        # Nombre de cas sur X - FORMATAGE ET TICKS AJOUTÉS
        x=alt.X('Nombre de Cas', 
                title="Nombre de Cas", 
                axis=alt.Axis(format='d', values=loc_ticks)), # <-- AJOUT DE 'values=loc_ticks'
        # Localisation sur Y, triée par fréquence descendante
        y=alt.Y('Localisation', sort='-x', title="Localisation"),
        # Couleur basée sur la localisation pour la distinction visuelle
        color=alt.Color('Localisation', title="Localisation"),
        # Affichage des valeurs dans les barres
        text=alt.Text('Nombre de Cas', format='.0f'),
        tooltip=['Localisation', 'Nombre de Cas']
    ).properties(
        title=f'Localisations des cas pour {joueuse_selectionnee}' # Utilisation de joueuse_selectionnee
    ).interactive()

    st.altair_chart(chart_loc, use_container_width=True)
    

    # --- B. Répartition par Type de Blessure ---

    st.subheader("Type de blessure")

    df_type_count = df_profil.groupby(
        ['Type Blessure']
    ).size().reset_index(name='Nombre de Cas')

    # Générer les Ticks pour le Tableau 2
    type_ticks = generate_integer_ticks(df_type_count, 'Nombre de Cas')

    chart_type = alt.Chart(df_type_count).mark_bar().encode(
        # Nombre de cas sur X - FORMATAGE ET TICKS AJOUTÉS
        x=alt.X('Nombre de Cas', 
                title="Nombre de Cas", 
                axis=alt.Axis(format='d', values=type_ticks)), # <-- AJOUT DE 'values=type_ticks'
        y=alt.Y('Type Blessure', sort='-x', title="Type de Blessure"),
        # Couleur basée sur le Type de Blessure pour la distinction visuelle
        color=alt.Color('Type Blessure', title="Type de Blessure"),
        text=alt.Text('Nombre de Cas', format='.0f'),
        tooltip=['Type Blessure', 'Nombre de Cas']
    ).properties(
        title='Types de Blessure (Antécédents et Actuelles)'
    ).interactive()

    st.altair_chart(chart_type, use_container_width=True)

    
    # --- C. Historique Chronologique (Frise avec Lignes et Couleur par Type) ---

    st.subheader("Frise chronologique des blessures")

    # 1. COUCHE DE RÈGLES / GRILLE : Utilisant mark_rule pour les séparations horizontales
    # Cela crée les lignes de grille claires pour chaque Localisation
    line_layer = alt.Chart(df_profil).mark_rule(
        color='lightgray', 
        strokeWidth=1, 
        opacity=1 
    ).encode(
        # L'encodage 'y' définit où la ligne est tracée. On utilise Localisation.
        y=alt.Y('Localisation', title="Localisation")
    )

    # 2. Couche des Points 
    point_layer = alt.Chart(df_profil).mark_circle(
        size=150 # Taille fixe pour la clarté
    ).encode(
        x=alt.X('Date blessure', title="Date de l'événement"),
        y=alt.Y('Localisation'), # Réutiliser la même colonne pour aligner les points sur la ligne
        
        # Couleur basée uniquement sur le Type de Blessure (simple, sans conditionnel)
        color=alt.Color('Type Blessure', title="Type Blessure"),
        
        tooltip=['Date blessure', 'Localisation', 'Gravité', 'Remarque', 'Jours Absents']
    )
    
    # 3. Combinaison des deux couches
    # On utilise la combinaison (+) pour superposer les points sur les lignes
    chart_chrono = (line_layer + point_layer).properties(
        title="Évolution chronologique des Antécédents et Blessures"
    ).interactive()

    st.altair_chart(chart_chrono, use_container_width=True)

    if not df_anthropo_suivi_j.empty:
        creer_graph_poids_blessure(
        df_poids=df_anthropo_suivi_j,
        df_blessure=df_blessures_j, # Utilisé ici comme argument
        x_col='Date',
        y_col='Poids (kg)',
        titre='Évolution du Poids et Événements de Blessure (Joueuse X)'
    )

    st.markdown("---")
    
    # Affichage du DataFrame combiné pour vérification
    with st.expander("Afficher les DataFrames de comptage pour vérification des doublons"):
        st.markdown("##### 1. Comptage par Localisation (`df_loc_count`)")
        st.dataframe(df_loc_count)
        st.markdown(f"**Vérification des fréquences :** {list(df_loc_count['Nombre de Cas'].values)}")
        st.markdown(f"**Graduations utilisées (Ticks) :** {loc_ticks}")
        
        st.markdown("##### 2. Comptage par Type de Blessure (`df_type_count`)")
        st.dataframe(df_type_count)
        st.markdown(f"**Vérification des fréquences :** {list(df_type_count['Nombre de Cas'].values)}")
        st.markdown(f"**Graduations utilisées (Ticks) :** {type_ticks}")
        
        st.markdown("---")
        st.markdown("##### DataFrame Combiné (`df_profil`)")
        st.dataframe(df_profil)


with tab4:
    df_gps_match_j_filtre = filtre_match(df=df_gps_match_j)
    df_gps_match_j_filtre['VHSR effort'] = df_gps_match_j_filtre['VHSR + SPR effort'] - df_gps_match_j_filtre['Sprint effort']
    df_gps_match_j_filtre = df_gps_match_j_filtre.drop(['SPR Total Distance (m)', 'SPR + Total Distance (m)', 'VHSR + SPR effort'], axis=1)

    ##### Match complet actuel ######
    agg_dict_match = {}
    for col in cols_ref_match_indiv:
        if col in cols_de_regroupement_indiv:
            continue
        if col == 'V max':
            agg_dict_match[col] = 'max'
        elif col == 'Meterage Per Minute':
            agg_dict_match[col] = 'mean'
        else:
            agg_dict_match[col] = 'sum'
    df_ref_joueuse = df_gps_match_j_filtre.groupby(cols_de_regroupement_indiv).agg(agg_dict_match).reset_index()
    df_ref_joueuse = df_ref_joueuse.drop(['Activity Name', 'Position Name', 'Type match', 'Saison', 'Mi-temps'], axis=1)

    if isinstance(df_ref_joueuse, pd.DataFrame):
        if not df_ref_joueuse.empty:
            # On prend la première (et seule) ligne
            joueuse_stats = df_ref_joueuse.iloc[0]
        else:
            st.error("Le DataFrame de référence est vide.")
            st.stop()
    else:
        joueuse_stats = df_ref_joueuse
    

    # --- Données individuels vs mes références
    st.subheader("📊 Performance du match vs Mes références")
    st.markdown("Les références de match sont calculées sur tous les match de plus de 60 mins")
    stats_df, nb_matchs = get_player_summary_stats(df_gps_match_j, min_time=60)
    afficher_comparaison_match(joueuse_stats, stats_df)
    st.divider()

    # --- Résumé + comparaison données du poste
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🏆 Résumé de vos données")
        st.markdown('')
        st.markdown('')
        st.markdown('')
        afficher_radar_performance(joueuse_stats, stats_df)

    with col2:
        st.markdown(f"### 🏆 Référentiel de Performance au Poste : {profil_joueuse['1er Poste']}")
        st.write("En BLEU la moyenne et en ROUGE le max de votre POSTE sur les match de plus de 60 mins")

        df_gps_poste = get_poste_all_match(df_gps_match)
        benchmarks_stats, _ = calculate_position_benchmarks(df_gps_poste)
        stats_joueuse = get_player_benchmarks(benchmarks_stats, profil_joueuse)
        display_stats_joueuse_cards(stats_joueuse)


    ####### Tous mes matchs ########
    st.divider()
    st.subheader("Tous mes matchs au club")
    tous_les_match = get_player_all_match(df_gps_match_j)    
    st.dataframe(tous_les_match, hide_index=True)


with tab5:
    st.write("Synthèse des charges, RPE et Wellness sur les 7 derniers jours.")
