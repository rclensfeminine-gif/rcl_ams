import streamlit as st
import pandas as pd
from sections.menu.menu import custom_sidebar_menu
from sections.constantes import cols_num, cols_stat

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

# --- Vérification de l'initialisation ---
# On vérifie si la fonction calculer_toutes_references a été exécutée.
if 'df_statistiques_base' not in st.session_state or 'ref_j1_cumule' not in st.session_state:
    st.error("Veuillez d'abord charger et initialiser les données depuis la page d'Accueil.")
    st.stop()

if 'df_groupe_final' in st.session_state:
    df_groupe_final = st.session_state['df_groupe_final']
# --- Récupération des DataFrames pré-calculés ---

# DataFrame de base filtré et préparé (pour les statistiques inter-semaine)
df_statistiques = st.session_state['df_statistiques_base']

# DataFrames de moyennes individuelles par jour (pour le cumul)
df_j1_mean_par_joueuse = st.session_state['ref_j1_brute'] 
df_j2_mean_par_joueuse = st.session_state['ref_j2_brute'] 
df_j3_mean_par_joueuse = st.session_state['ref_j3_brute'] 
df_j4_mean_par_joueuse = st.session_state['ref_j4_brute'] 

# On récupère également les DataFrames cumulés pour l'affichage final
df_charge_j1_j2_j3_j4_cumulee = st.session_state['ref_j1_cumule'] 
df_charge_j2_j3_j4_cumulee = st.session_state['ref_j2_cumule'] 
df_charge_j3_j4_cumulee = st.session_state['ref_j3_cumule'] 


# --- Fonctions utilitaires d'affichage (pour les statistiques inter-semaine) ---

# Le calcul des moyennes de groupe (Moyenne, Max, Écart type) 
# doit être fait ici car il n'est utilisé que pour cet affichage.
def calculer_stats_groupe_par_jour(df_statistiques, jour_label, cols_num):
    """Calcule et affiche les stats de groupe (Mean, Max, Std) pour un jour donné."""
    df_jour = df_statistiques[df_statistiques['Jour semaine'] == jour_label]
    
    # 1. Déterminer les métriques valides (cols_num doit être importé ou défini)
    # Assurez-vous d'importer 'cols_num' depuis 'sections.constantes' si ce n'est pas déjà fait.
    try:
        from sections.constantes import cols_num
    except ImportError:
        st.warning("Assurez-vous que 'cols_num' est accessible ou défini.")
        return pd.DataFrame() # Retourne un DF vide si erreur
        
    metrics_to_mean = [col for col in cols_num if col in df_jour.columns]
    
    if df_jour.empty or not metrics_to_mean:
        return pd.DataFrame()

    # 2. Calculer les statistiques
    df_temp = df_jour[metrics_to_mean].agg(['mean', 'max', 'std'])
    df_stats = df_temp.round(2)
    
    # 3. Renommage
    df_stats = df_stats.rename(index={
        'mean': f'Moyenne {jour_label}',
        'max': f'Max {jour_label}',
        'std': f'Ecart type {jour_label}'
    })
    return df_stats


# --- main de la page ---
st.title("Statistiques - Groupe")

# --- statistiques inter-semaine ---
st.header("Statistiques : inter-semaine")
st.dataframe(df_groupe_final)

def assigner_microcycle(df):
    """
    Identifie et assigne un identifiant de microcycle personnalisé '4J-X'.
    (Le corps de la fonction avec toutes les logiques d'exclusion reste le même.)
    """
    df = df.copy()
    
    # 1. Préparation et tri (Identique)
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'])
        
    df = df.sort_values(by=['Date']).reset_index(drop=True)
    df['Jour semaine'] = df['Jour semaine'].str.strip().str.upper()
    
    jours_a_cumuler_strictement = ['J-4', 'J-3', 'J-2', 'J-1']
    est_jour_standard_cumul = df['Jour semaine'].isin(jours_a_cumuler_strictement)
    
    # 2. Création de l'ID de Microcycle (PASS 1: Attribution large) (Identique)
    est_j4_strict = (df['Jour semaine'] == 'J-4')
    df.loc[est_j4_strict, 'Microcycle ID'] = est_j4_strict.cumsum()
    df['Microcycle ID'] = df['Microcycle ID'].ffill()
    df.loc[~df['Jour semaine'].str.startswith('J-'), 'Microcycle ID'] = pd.NA
    
    # 3. Identification des Cycles à Exclure (PASS 2: Contamination et J-5) (Identique)
    cycles_a_exclure = set()

    # --- EXCLUSION 1 : Contamination interne ---
    cycles_contamines = df.loc[
        (df['Microcycle ID'].notna()) & (~est_jour_standard_cumul), 
        'Microcycle ID'
    ].unique()
    cycles_a_exclure.update(cycles_contamines)

    # --- EXCLUSION 2 : Présence d'un J-5 avant le J-4 ---
    indices_j4 = df.loc[df['Jour semaine'] == 'J-4'].index
    for index in indices_j4:
        if index > 0 and df.loc[index - 1, 'Jour semaine'] == 'J-5':
            id_a_exclure = df.loc[index, 'Microcycle ID']
            if pd.notna(id_a_exclure):
                cycles_a_exclure.add(id_a_exclure)

    # 4. Application de l'Exclusion Totale (Identique)
    if len(cycles_a_exclure) > 0:
        df.loc[df['Microcycle ID'].isin(cycles_a_exclure), 'Microcycle ID'] = pd.NA

    # 5. Filtrage final (Identique)
    df.loc[~est_jour_standard_cumul, 'Microcycle ID'] = pd.NA 

    # ----------------------------------------------------------------------
    # 🚨 MODIFICATION : CRÉATION DE L'ID PERSONNALISÉ '4J-X' 🚨
    # ----------------------------------------------------------------------
    
    # Convertir l'ID numérique flottant en chaîne de caractères, puis ajouter le préfixe
    df['Microcycle ID'] = df['Microcycle ID'].astype('Int64').astype(str)
    
    # Remplacer les identifiants numériques par '4J-X' si l'ID n'est pas 'nan'
    df['Microcycle ID'] = df['Microcycle ID'].apply(
        lambda x: f"4J-{x}" if x not in ('<NA>', 'nan') else pd.NA
    )
    
    return df

# df_synthese est le DataFrame que vous avez calculé "autre part"
if 'Jour semaine' in df_groupe_final.columns and 'Date' in df_groupe_final.columns:
    df_inter_semaine = assigner_microcycle(df_groupe_final)
    st.dataframe(df_inter_semaine)
else:
    print("Erreur : Colonnes 'Date' ou 'Jour semaine' manquantes pour identifier le microcycle.")

df_analyse_4J = df_inter_semaine.groupby(
    ['Microcycle ID']
)[cols_stat].sum().reset_index()

st.dataframe(df_analyse_4J)

st.markdown("---")
st.header("Statistiques : intra-semaine")
# Affichage des statistiques de groupe pour chaque jour (calculées à la volée)
try:
    from sections.constantes import cols_num
except ImportError:
    st.error("Impossible d'importer 'cols_num'. Veuillez vérifier votre fichier constantes.")
    st.stop()
    

jours = ['J-1', 'J-2', 'J-3', 'J-4']
for jour in jours:
    df_stats_jour = calculer_stats_groupe_par_jour(df_statistiques, jour, cols_num)
    if not df_stats_jour.empty:
        st.subheader(f"Statistiques {jour}")
        st.dataframe(df_stats_jour)


# -------------------------------------------------------------------------------------------------
# --- Statistiques Individuelles ---
# -------------------------------------------------------------------------------------------------

st.markdown("---")
st.title("Statistiques - Individuelles")
st.header("Charge moyenne par jour")

# Affichage des DataFrames de moyennes individuelles par jour (pré-calculés)
st.subheader("Charge Moyenne J-1 → Joueuse")
st.dataframe(df_j1_mean_par_joueuse.round(2), hide_index=True)

st.subheader("Charge Moyenne J-2 → Joueuse")
st.dataframe(df_j2_mean_par_joueuse.round(2), hide_index=True)

st.subheader("Charge Moyenne J-3 → Joueuse")
st.dataframe(df_j3_mean_par_joueuse.round(2), hide_index=True)

st.subheader("Charge Moyenne J-4 → Joueuse")
st.dataframe(df_j4_mean_par_joueuse.round(2), hide_index=True)


# --- Charge cumulée sur la semaine ---
st.markdown("---")
st.header("Charge cumulée sur la semaine")

# Affichage des DataFrames cumulés (pré-calculés)

# J-4 (Affichage de la référence unique)
st.subheader("Charge Moyenne Individuelle : J-4")
st.dataframe(df_j4_mean_par_joueuse.round(2), hide_index=True)

# J-3 + J-4
if not df_charge_j3_j4_cumulee.empty:
    st.subheader("Charge Moyenne Cumulée : J-3")
    st.dataframe(df_charge_j3_j4_cumulee, hide_index=True)

# J-2 + J-3 + J-4
if not df_charge_j2_j3_j4_cumulee.empty:
    st.subheader("Charge Moyenne Cumulée : J-2")
    st.dataframe(df_charge_j2_j3_j4_cumulee, hide_index=True)
    
# J-1 + J-2 + J-3 + J-4
if not df_charge_j1_j2_j3_j4_cumulee.empty:
    st.subheader("Charge Moyenne Cumulée : J-1")
    st.dataframe(df_charge_j1_j2_j3_j4_cumulee, hide_index=True)