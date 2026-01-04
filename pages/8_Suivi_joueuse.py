import streamlit as st
import pandas as pd
import os
from sections.menu.menu import custom_sidebar_menu
from datetime import date
from sections.joueuses.pipeline import calculer_ratios_isocinetisme, calculer_metriques_hop_test, calculer_metriques_sauts, calculer_metriques_dynamo, sauvegarder_suivi_global, sauvegarder_fixes, sauvegarder_df_global, init_session_state_poids_plis, generate_pli_inputs_optimized
from sections.joueuses.pipeline import PLIS_NOMS, PLIS_PRISES_COLS

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

# ----------------------------------------------------------------------
# 1. FONCTIONS DE CHARGEMENT
# ----------------------------------------------------------------------
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


@st.cache_data(show_spinner="Chargement des données de profil...")
def charger_toutes_data():
    """Charge l'ensemble des DataFrames de suivi."""
    
    # 1. Liste des Joueuses (Identité)
    if os.path.exists(FICHIER_ID_JOUEUSES):
        df_id = pd.read_csv(FICHIER_ID_JOUEUSES, encoding='utf-8')
        df_id['Nom Complet'] = df_id['Prénom'].astype(str) + ' ' + df_id['NOM'].astype(str)
        df_id = df_id.dropna(subset=['Prénom', 'NOM'], how='all')
    else:
        st.error("Fichier d'identité introuvable. Assurez-vous que 'data/identite.csv' existe.")
        # Retourne 9 DataFrames vides
        return (pd.DataFrame(),) * 9

    # 2. Données Anthropométriques Fixes
    colonnes_fixes = ['Prénom', 'NOM', 'Taille (cm)', 'EIAS - Maléole D', 'EIAS - Maléole G', 'Tour Poignet (cm)']
    if os.path.exists(FICHIER_ANTHROPO_FIXE):
        df_fixes = pd.read_csv(FICHIER_ANTHROPO_FIXE, encoding='utf-8')
    else:
        df_fixes = pd.DataFrame(columns=colonnes_fixes)
    
    # 3. Données Anthropométriques Suivies (Longitudinales)
    # AJOUT de la colonne 'Remarque' + des colonnes pour les prises individuelles des plis
    colonnes_suivi = ['Date', 'Prénom', 'NOM', 'Poids (kg)', 'Remarque'] + PLIS_NOMS + PLIS_PRISES_COLS # <- AJOUT DE 'Remarque'
    
    if os.path.exists(FICHIER_ANTHROPO_SUIVI):
        df_suivi = pd.read_csv(FICHIER_ANTHROPO_SUIVI, encoding='utf-8')
        if 'Date' in df_suivi.columns:
            # Conversion essentielle de la date
            df_suivi['Date'] = pd.to_datetime(df_suivi['Date'], format='mixed', errors='coerce').dt.date
    else:
        df_suivi = pd.DataFrame(columns=colonnes_suivi)

    # --- Les autres fichiers (non modifiés) ---
    def charger_df(filepath, date_col, columns):
        if os.path.exists(filepath):
            df = pd.read_csv(filepath, encoding='utf-8')
            if date_col in df.columns:
                 df[date_col] = pd.to_datetime(df[date_col], format='mixed', errors='coerce').dt.date
            return df
        else:
            return pd.DataFrame(columns=columns)

    df_blessures = charger_df(FICHIER_SUIVI_BLESSURES, 'Date Début', ['Date Début', 'Prénom', 'NOM', 'Localisation', 'Gravité', 'SC/AC', 'Cycle Menstruel', 'Remarque', 'Type Blessure', 'Type Séance', 'Type Surface', 'Date Réa', 'Date RTP', 'Date RTC'])
    df_antecedent = charger_df(FICHIER_ANTECEDENTS, 'Date blessure', ['Date blessure', 'Prénom', 'NOM', 'Localisation', 'Gravité', 'SC/AC', 'Cycle Menstruel', 'Remarque', 'Type Blessure', 'Type Séance', 'Type Surface', 'Date RTR', 'Date Réa', 'Date RTP', 'Date RTC'])
    df_isocinetisme = charger_df(FICHIER_ISOCINETISME, 'Date Test', ['Date Test', 'Prénom', 'NOM', 'Remarque', 'Q60° D', 'Q60° G', 'Dif Q60°', 'IJ60° D', 'IJ60° G', 'Dif IJ60°', 'Q240° D', 'Q240° G', 'Dif Q240°', 'IJ240° D', 'IJ240° G', 'Dif IJ240°', 'IJExc D', 'IJExc G', 'Dif IJExc'])
    df_hop_test = charger_df(FICHIER_HOP_TEST, 'Date Test', ['Date Test', 'Prénom', 'NOM', 'SHT D1', 'SHT D2', 'SHT D3', 'Nbr SHT D', 'SHT G1', 'SHT G2', 'SHT G3', 'Nbr SHT G', 'THT D1', 'THT D2', 'THT D3', 'Nbr THT D', 'THT G1', 'THT G2', 'THT G3', 'Nbr THT G', 'CHT D1', 'CHT D2', 'CHT D3', 'Nbr CHT D', 'CHT G1', 'CHT G2', 'CHT G3', 'Nbr CHT G', 'LHT D', 'LHT G'])
    df_sauts = charger_df(FICHIER_SAUTS, 'Date Test', ['Date Test', 'Prénom', 'NOM', 'CMJ 1', 'CMJ 2', 'CMJ 3', 'CMJ Bras 1', 'CMJ Bras 2', 'CMJ Bras 3', 'CMJ 1J D1', 'CMJ 1J D2', 'CMJ 1J D3', 'CMJ 1J G1', 'CMJ 1J G2', 'CMJ 1J G3', 'SRJT 5 Mean 1', 'SRJT 5 RSI 1', 'SRJT 5 Mean 2', 'SRJT 5 RSI 2', 'SRJT 5 Mean 3', 'SRJT 5 RSI 3'])
    df_dynamo = charger_df(FICHIER_DYNAMO, 'Date Test', ['Date Test', 'Prénom', 'NOM', 'Soléaire D', 'Soléaire G', 'Soléaire H barre', 'Gastro D', 'Gastro G', 'Tibial post D', 'Tibial post G', 'Fibulaire D', 'Fibulaire G', 'Abducteur D', 'Abducteur G', 'Adducteur D', 'Adducteur G'])
        
    return df_id, df_fixes, df_suivi, df_blessures, df_antecedent, df_isocinetisme, df_hop_test, df_sauts, df_dynamo

# Charger toutes les données
df_identite, df_fixes_historique, df_suivi_historique, df_blessures_historique, df_antecedent_historique, df_isocinetisme_historique, df_hop_test_historique, df_sauts_historique, df_dynamo_historique = charger_toutes_data()

# ----------------------------------------------------------------------
# 4. SÉLECTION ET INITIALISATION DU PROFIL
# ----------------------------------------------------------------------

st.title("Suivi Anthropométrique")

if df_identite.empty:
    st.error("Le tableau d'identité est vide. Veuillez d'abord le remplir via la page Identité.")
    st.stop()

# Créer la liste des noms pour le sélecteur
liste_noms = df_identite['Nom Complet'].sort_values().unique().tolist()

joueuse_selectionnee = st.sidebar.selectbox(
    "Joueuse :",
    options=liste_noms
)

# Récupérer Prénom et NOM séparés
infos_joueuse_id = df_identite[df_identite['Nom Complet'] == joueuse_selectionnee].iloc[0]
prenom_j = infos_joueuse_id['Prénom']
nom_j = infos_joueuse_id['NOM']

# Filtrer les données fixes existantes pour cette joueuse
df_fixes_joueuse = df_fixes_historique[
    (df_fixes_historique['Prénom'] == prenom_j) & 
    (df_fixes_historique['NOM'] == nom_j)
]
profil_fixes_initial = df_fixes_joueuse.iloc[0].to_dict() if not df_fixes_joueuse.empty else {}

# ----------------------------------------------------------------------
# 5. SECTION DES MESURES FIXES (Formulaire 1)
# ----------------------------------------------------------------------

st.subheader(f"1. Mesures statiques : {joueuse_selectionnee}")
st.warning("⚠️ Ces valeurs écraseront les précédentes dans le fichier 'anthropo_fixes.csv'.")

with st.form(key='form_fixes'):
    
    col_f1, col_f2, col_f3 = st.columns(3)
    
    # Fonction utilitaire pour obtenir la valeur initiale
    def get_val(key, default=None):
        return profil_fixes_initial.get(key) if profil_fixes_initial.get(key) is not None else default

    with col_f1:
        taille = st.number_input("Taille (cm)", value=get_val('Taille (cm)'), min_value=0.0, format="%.1f")
        
    with col_f2:
        eias_d = st.number_input("EIAS - Maléole D (cm)", value=get_val('EIAS - Maléole D'), min_value=0.0, format="%.1f")
        eias_g = st.number_input("EIAS - Maléole G (cm)", value=get_val('EIAS - Maléole G'), min_value=0.0, format="%.1f")
        
    with col_f3:
        tour_poignet = st.number_input("Tour Poignet (cm)", value=get_val('Tour Poignet (cm)'), min_value=0.0, format="%.1f")
        
    submit_fixes = st.form_submit_button("💾 Enregistrer/Écraser les mesures fixes")

if submit_fixes:
    # 1. Créer une nouvelle ligne avec les données mises à jour
    nouvelle_ligne_fixes = {
        'Prénom': prenom_j,
        'NOM': nom_j,
        'Taille (cm)': taille,
        'EIAS - Maléole D': eias_d,
        'EIAS - Maléole G': eias_g,
        'Tour Poignet (cm)': tour_poignet
    }
    
    # 2. Mettre à jour le DataFrame fixe
    df_fixes_temp = df_fixes_historique.copy()
    
    # Trouver l'index existant ou créer une nouvelle ligne
    condition_fixes = (df_fixes_historique['Prénom'] == prenom_j) & (df_fixes_historique['NOM'] == nom_j)
    
    if condition_fixes.any():
        idx = df_fixes_joueuse.index[0]
        for key, value in nouvelle_ligne_fixes.items():
            df_fixes_temp.loc[idx, key] = value
    else:
        # Si la joueuse n'existe pas encore dans le fichier fixe, ajouter la ligne
        df_fixes_temp = pd.concat([df_fixes_temp, pd.DataFrame([nouvelle_ligne_fixes])], ignore_index=True)


    if sauvegarder_fixes(df_fixes_temp):
        st.success(f"✅ Mesures fixes de {joueuse_selectionnee} mises à jour dans 'anthropo_fixes.csv'.")

# ----------------------------------------------------------------------
# 7. SECTION DES MESURES LONGITUDINALES (Poids, Remarques et Plis)
# ----------------------------------------------------------------------

st.markdown("---")
st.subheader(f"2. Mesures suivies : {joueuse_selectionnee}")

# --- 1. SÉLECTION DE LA DATE ET POIDS ---
col_date_align, col_poids_align = st.columns([1, 2])
with col_date_align:
    # Utiliser st.session_state pour conserver la date si possible
    if 'current_date_input' not in st.session_state:
        st.session_state['current_date_input'] = date.today()
        
    date_saisie = st.date_input(
        "Date de la mesure", 
        value=st.session_state['current_date_input'],
        key='date_input_key'
    )
    st.session_state['current_date_input'] = date_saisie

date_a_comparer = date_saisie

# Détermination si une mesure existe déjà pour cette date
mesure_a_modifier = df_suivi_historique[
    (df_suivi_historique['Prénom'] == prenom_j) & 
    (df_suivi_historique['NOM'] == nom_j) &
    (df_suivi_historique['Date'] == date_a_comparer) 
]

suppression_possible = not mesure_a_modifier.empty

# --- 2. INITIALISATION GLOBALE DE LA SESSION_STATE ---
init_session_state_poids_plis(mesure_a_modifier, date_a_comparer)


# --- 3. SAISIE DU POIDS ---
POIDS_KEY = "poids_input"

with col_poids_align:
    # La valeur initiale est gérée par session_state (initialisée plus haut)
    poids = st.number_input(
        "Poids (kg)", 
        value=st.session_state.get(POIDS_KEY), 
        min_value=0.0,
        format="%.1f", 
        help="Saisir le poids mesuré. Obligatoire.",
        key=POIDS_KEY 
    )

if suppression_possible:
    st.warning(f"Mesure existante pour le **{date_saisie.strftime('%d/%m/%Y')}** : **MODIFICATION / SUPPRESSION**.")
else:
    st.info(f"Aucune mesure pour le **{date_saisie.strftime('%d/%m/%Y')}** : **AJOUT**.")


# --- 4. SAISIE DE LA REMARQUE (NOUVEAU) ---
st.write("---") 
st.write("#### Remarque")

REMARQUE_KEY = "remarque_input"
# La valeur initiale est gérée par session_state (initialisée plus haut)
remarque = st.text_area(
    "Remarque / Contexte de la mesure (Entraînement, cycle, fatigue...)",
    value=st.session_state.get(REMARQUE_KEY), # Utilise la valeur chargée/gardée en session state
    key=REMARQUE_KEY,
    height=70,
    help="Note contextuelle liée à la mesure de poids et plis de cette date.",
    label_visibility="collapsed"
)


# --- 5. SAISIE DES PLIS ---

st.write("---") 
st.write("#### Plis Cutanés (mm) - Saisir les prises 1 et 2")
st.caption("La moyenne est calculée automatiquement lors de l'enregistrement.")

# Regrouper les plis par colonnes pour un affichage plus compact (4 par 4)
plis_col_1 = PLIS_NOMS[:4]
plis_col_2 = PLIS_NOMS[4:]

col_gauche, col_droite = st.columns(2)

# Colonne de Gauche
with col_gauche:
    for pli in plis_col_1:
        generate_pli_inputs_optimized(pli)

# Colonne de Droite
with col_droite:
    for pli in plis_col_2:
        generate_pli_inputs_optimized(pli)


# --- 6. LE FORMULAIRE POUR LES BOUTONS ---
with st.form(key='form_suivi'):
    col_save, col_delete = st.columns(2)

    with col_save:
        submit_suivi = st.form_submit_button("💾 Enregistrer/Mettre à jour la Saisie")

    with col_delete:
        delete_suivi = st.form_submit_button(
            "🗑️ Supprimer la Mesure",
            disabled=not suppression_possible,
            help="Supprime la mesure de poids et plis pour la date sélectionnée."
        )

# --- 7. TRAITEMENT ET SAUVEGARDE ---

if submit_suivi or delete_suivi:
    
    # 1. Identifier la condition de la ligne 
    condition_ligne = (
        (df_suivi_historique['Prénom'] == prenom_j) & 
        (df_suivi_historique['NOM'] == nom_j) &
        (df_suivi_historique['Date'] == date_a_comparer)
    )

    # --- GESTION DE LA SUPPRESSION ---
    if delete_suivi:
        if condition_ligne.any():
            # Créer une copie du DataFrame sans la ligne à supprimer
            df_suivi_temp = df_suivi_historique[~condition_ligne].copy()
            
            if sauvegarder_suivi_global(df_suivi_temp): 
                st.success(f"🗑️ Mesure du **{date_saisie.strftime('%d/%m/%Y')}** supprimée pour {joueuse_selectionnee}.")
                st.cache_data.clear()
                st.rerun()
        else:
            st.error("❌ Aucune mesure trouvée à supprimer pour cette date.")

    # --- GESTION DE L'ENREGISTREMENT / MISE À JOUR (Upsert) ---
    elif submit_suivi: 
        poids_final = st.session_state.get("poids_input")
        remarque_finale = st.session_state.get("remarque_input", "").strip() # <- Récupération de la remarque
        
        # Validation obligatoire : le poids doit être saisi
        if poids_final is None or pd.isna(poids_final) or poids_final <= 0:
            st.error("❌ Le Poids (kg) est obligatoire (et doit être supérieur à zéro) pour l'enregistrement de suivi.")
            
        else:
            # 2. Calcul des plis 
            plis_finaux = {}
            plis_prises_individuelles = {} # Stocke les prises 1 et 2
            
            for pli_name in PLIS_NOMS:
                val_1 = st.session_state.get(f"pli_{pli_name}_1")
                val_2 = st.session_state.get(f"pli_{pli_name}_2")
                
                # Filtrer les valeurs non nulles (None, 0.0)
                valeurs_saisies = [v for v in [val_1, val_2] if v is not None and v > 0]
                
                # Stockage des prises individuelles (même si seulement 1 ou 2)
                plis_prises_individuelles[f"{pli_name}_1"] = val_1 if val_1 is not None and val_1 > 0 else None
                plis_prises_individuelles[f"{pli_name}_2"] = val_2 if val_2 is not None and val_2 > 0 else None
                
                if len(valeurs_saisies) >= 1:
                    # Calculer la moyenne ou prendre la valeur unique
                    plis_finaux[pli_name] = round(sum(valeurs_saisies) / len(valeurs_saisies), 1)
                else:
                    plis_finaux[pli_name] = None 
                        
            # 3. Création de la nouvelle entrée
            nouvelle_entree = {
                'Date': date_saisie,
                'Prénom': prenom_j,
                'NOM': nom_j,
                'Poids (kg)': poids_final,
                'Remarque': remarque_finale if remarque_finale else None, # <- Ajout de la Remarque (None si vide)
                **plis_finaux,
                **plis_prises_individuelles # Ajout des prises individuelles
            }
            
            df_suivi_temp = df_suivi_historique.copy()
            
            # 4. Logique d'Upsert (Mise à jour ou Ajout)
            if condition_ligne.any():
                # Mise à jour (Update)
                idx = df_suivi_temp[condition_ligne].index[0]
                for key, value in nouvelle_entree.items():
                    # Utiliser loc pour une affectation explicite
                    df_suivi_temp.loc[idx, key] = value
                message_succes = f"✅ Mesure du **{date_saisie.strftime('%d/%m/%Y')}** mise à jour pour {joueuse_selectionnee}."
            else:
                # Ajout (Insert)
                # Filtrer les valeurs None pour éviter de créer des colonnes NaN inutiles lors du concat
                nouvelle_entree_filtree = {k: v for k, v in nouvelle_entree.items() if v is not None}
                # Pour s'assurer que toutes les colonnes existent, on utilise pd.concat
                df_suivi_temp = pd.concat([df_suivi_temp, pd.DataFrame([nouvelle_entree_filtree])], ignore_index=True)
                message_succes = f"✅ Nouvelle mesure enregistrée pour {joueuse_selectionnee} à la date du **{date_saisie.strftime('%d/%m/%Y')}**."
                
            # 5. Sauvegarde finale
            if sauvegarder_suivi_global(df_suivi_temp):
                st.success(message_succes)
                st.cache_data.clear() # Vider le cache pour forcer le rechargement
                st.rerun()




##################################
st.markdown("---")
st.title("Suivi blessure")
st.subheader(f"Ajouter une blessure pour {joueuse_selectionnee}")
st.info("ℹ️ Une blessure est identifiée par la **Joueuse** et la **Date de Début**. Remplissez le formulaire ci-dessous pour ajouter, modifier ou supprimer une blessure.")

# ----------------------------------------------------------------------
# 9. SECTION SUIVI BLESSURES (Affichage et Édition Directe)
# ----------------------------------------------------------------------

# Colonnes de date à convertir dans le DF historique global, si elles existent
colonnes_date_global = ['Date blessure', 'Date RTR', 'Date RTC', 'Date Réa', 'Date RTP']

for col in colonnes_date_global:
    if col in df_blessures_historique.columns:
        # Convertit les valeurs en dates. Les valeurs invalides (NaN/FLOAT) deviennent NaT.
        df_blessures_historique[col] = pd.to_datetime(
            df_blessures_historique[col], 
            errors='coerce'
        )

# 1. Filtrage initial et copie
df_blessures_joueuse = df_blessures_historique[
    (df_blessures_historique['Prénom'] == prenom_j) & 
    (df_blessures_historique['NOM'] == nom_j)
].sort_values(by='Date blessure', ascending=False).copy()

# 2. Préparation des colonnes pour le st.data_editor (y compris le calcul)

if not df_blessures_joueuse.empty:
    
    # 2a. Conversion des colonnes Texte/Selectbox (pour éviter l'erreur FLOAT)
    # On force la conversion des colonnes de texte/select box à string et on remplace les NaN par ''
    cols_a_convertir_str = ['Remarque', 'Localisation', 'Type Blessure', 'Gravité', 'Type Séance', 'Type Surface', 'Cycle Menstruel']
    for col in cols_a_convertir_str:
        if col in df_blessures_joueuse.columns:
            # S'assurer que les valeurs non renseignées sont des chaînes vides pour st.data_editor
            df_blessures_joueuse[col] = df_blessures_joueuse[col].fillna('').astype(str)

    # 2b. Recalcul des Jours Absents (utilise Date RTP comme date de fin)
    # Assurez-vous que Date RTP est un objet datetime pour le calcul
    df_blessures_joueuse['Date RTP'] = pd.to_datetime(df_blessures_joueuse['Date RTP']) 
    date_fin_calc = df_blessures_joueuse['Date RTP'].fillna(pd.to_datetime(date.today()))
    df_blessures_joueuse['Jours Absents'] = (date_fin_calc - df_blessures_joueuse['Date blessure']).dt.days
    
    # 2c. Nettoyage des dates après calcul (pas de conversion supplémentaire nécessaire)
    
else:
    # 🚨 Si le DF est vide, assurez-vous qu'il contient toutes les colonnes requises.
    colonnes_base = list(df_blessures_historique.columns) 
    if 'Jours Absents' not in colonnes_base:
        colonnes_base.append('Jours Absents')
        
    df_blessures_joueuse = pd.DataFrame(columns=colonnes_base)

    
# --- Définition des Options et Colonnes (Identique) ---
OPTIONS_LOCALISATION = ["COM", "MALADE","HDC","DOS","ABDO","PSOAS D","PSOAS G","HANCHE D","HANCHE G","ADD D","ADD G","ISCHIO","ISCHIO D","ISCHIO G",
                         "SEMI M D","SEMI M G","SEMI T D","SEMI T G","B FEM D","B FEM G","QUADRI D","QUADRI G","QUADRI","SARTORIUS D","SARTORIUS G",
                         "V MED D","V MED G","V LAT D","V LAT G","D ANT", "D ANT D","D ANT G","GENOUX","GENOU D","GENOU G","PATELLA D","PATELLA G",
                         "LIG PAT D","LIG PAT G","MENISQUE D","MENISQUE G","LCA D","LCA G","LLI D","LLI G","LLE D","LLE G","POPLITE D","POPLITE G","TFL D",
                         "TFL G","MOLLETS","MOLLET D","MOLLET G","T ACHILLE D","T ACHILLE G","SOLEAIRE D","SOLEAIRE G","SEVER D","SEVER G"
                         "TIB ANT D","TIB ANT G","LONG FIB D","LONG FIB G","CHEVILLE D","CHEVILLE G","LTFP D","LTFP G","LTFA D","LTFA G","LCF D","LCF G","PIED D","PIED G"]
OPTIONS_GRAVITE = ["SYNOV","Epimysium","DOMS","DOULEURS","CON","INF","OSGOOD","SEVER","F","FF","LUX","LOMBALGIE","LUMBAGO","OEDEME","HERNIE","OP",
                   "ENTORSE","CERVICALES","G1","G2","G3","G4","LCA"]
OPTIONS_TYPE_BLESSURE = ["ART/LIG","MUSC","COM","OS"]
OPTIONS_SEANCE = ['Entrainement', 'Match']
OPTIONS_SURFACE = ['Herbe', 'Synthétique', "Salle"]
OPTIONS_CYCLE = ["Lutéale", "Folliculaire", "Late Folliculaire", "Règle"]
OPTIONS_CONTACT = ['SC', 'AC']


config_colonnes_blessure = {
    'Date blessure': st.column_config.DateColumn("Date blessure", format="YYYY/MM/DD", required=True),
    'Localisation': st.column_config.SelectboxColumn("Localisation", options=OPTIONS_LOCALISATION, required=True),
    'Type Blessure': st.column_config.SelectboxColumn("Type Blessure", options=OPTIONS_TYPE_BLESSURE, required=True),
    'Gravité': st.column_config.SelectboxColumn("Gravité", options=OPTIONS_GRAVITE),
    'SC/AC': st.column_config.SelectboxColumn("SC/AC", options=OPTIONS_CONTACT),
    'Type Séance': st.column_config.SelectboxColumn("Type Séance", options=OPTIONS_SEANCE),
    'Type Surface': st.column_config.SelectboxColumn("Type Surface", options=OPTIONS_SURFACE),
    'Cycle Menstruel': st.column_config.SelectboxColumn("Cycle Menstruel", options=OPTIONS_CYCLE),
    'Date RTR': st.column_config.DateColumn("Date RTR", format="YYYY/MM/DD"),
    'Date Réa': st.column_config.DateColumn("Date Réa", format="YYYY/MM/DD"),
    'Date RTP': st.column_config.DateColumn("Date RTP", format="YYYY/MM/DD"),
    'Date RTC': st.column_config.DateColumn("Date RTC", format="YYYY/MM/DD"),
    'Remarque': st.column_config.TextColumn("Remarque", width="large"),
    
    # Colonnes d'information (non éditables)
    'Prénom': st.column_config.TextColumn("Prénom", disabled=True),
    'NOM': st.column_config.TextColumn("NOM", disabled=True),
    'Jours Absents': st.column_config.NumberColumn("Jours Absents", format="%d j", disabled=True)
}

# --- Colonnes à afficher et éditer ---
colonnes_edition = list(config_colonnes_blessure.keys())

# --- Le st.data_editor remplace le tableau et le formulaire ---
df_blessures_modifiees = st.data_editor(
    df_blessures_joueuse[colonnes_edition],
    column_config=config_colonnes_blessure,
    hide_index=True,
    num_rows="dynamic", # <-- Permet l'ajout et la suppression de lignes
    key="editor_antecdent"
)

# ----------------------------------------------------------------------
# SAUVEGARDE ET LOGIQUE DES CHANGEMENTS
# ----------------------------------------------------------------------

# 1. Nettoyage : On ne garde que les lignes avec une date de début
df_blessures_modifiees_clean = df_blessures_modifiees.dropna(subset=['Date blessure'])

# 2. Ajout des clés (Prénom/NOM) pour les nouvelles lignes
if not df_blessures_modifiees_clean.empty:
    # Utilisation de .loc pour éviter le SettingWithCopyWarning
    df_blessures_modifiees_clean.loc[:, 'Prénom'] = prenom_j
    df_blessures_modifiees_clean.loc[:, 'NOM'] = nom_j

# 3. Comparaison avec l'historique global pour détecter les changements
# On prépare le DF historique SANS la joueuse actuelle pour la reconstruction
df_historique_sans_joueuse = df_blessures_historique[
    (df_blessures_historique['Prénom'] != prenom_j) | 
    (df_blessures_historique['NOM'] != nom_j)
].copy()

# Si le nombre de lignes (ajouts/suppressions) ou les valeurs changent
if df_blessures_modifiees_clean.shape[0] != df_blessures_joueuse.shape[0] or not df_blessures_modifiees_clean.equals(df_blessures_joueuse[colonnes_edition].dropna(subset=['Date blessure'])):
    
    st.warning(f"⚠️ {df_blessures_modifiees_clean.shape[0]} lignes de blessure à sauvegarder/mettre à jour.")
    
    # Le bouton de sauvegarde doit être placé EN DEHORS du data_editor
    if st.button("💾 SAUVEGARDER L'HISTORIQUE DES BLESSURES", key="save_blessures"):
        
        # Reconstruire le DataFrame global
        df_final_blessures = pd.concat([df_historique_sans_joueuse, df_blessures_modifiees_clean], ignore_index=True)
        
        # Sauvegarde
        if sauvegarder_df_global(df_final_blessures, FICHIER_SUIVI_BLESSURES):
            st.success("✅ Historique des blessures mis à jour.")
            st.cache_data.clear()
            st.rerun()
else:
    st.info("Aucune modification en attente (Isocinétique).")



##################################
st.markdown("---")
st.title("Antécédents de blessure")

st.subheader(f"Ajouter un antécédent pour {joueuse_selectionnee}")
st.info("ℹ️ Un antécédent est identifié par la **Joueuse** et la **Date de Début**. Remplissez le formulaire ci-dessous pour ajouter, modifier ou supprimer un antécédent.")

# ----------------------------------------------------------------------
# 10. SECTION SUIVI ANTECEDENTS (Affichage et Édition Directe)
# ----------------------------------------------------------------------

# Colonnes de date à convertir dans le DF historique global, si elles existent
colonnes_date_global = ['Date blessure', 'Date RTR', 'Date RTC', 'Date Réa', 'Date RTP']

for col in colonnes_date_global:
    if col in df_antecedent_historique.columns:
        # Convertit les valeurs en dates. Les valeurs invalides (NaN/FLOAT) deviennent NaT.
        df_antecedent_historique[col] = pd.to_datetime(
            df_antecedent_historique[col], 
            errors='coerce'
        )

# 1. Filtrage initial et copie
df_antecedent_joueuse = df_antecedent_historique[
    (df_antecedent_historique['Prénom'] == prenom_j) & 
    (df_antecedent_historique['NOM'] == nom_j)
].sort_values(by='Date blessure', ascending=False).copy()

# 2. Préparation des colonnes pour le st.data_editor (y compris le calcul)

if not df_antecedent_joueuse.empty:
    
    # 2a. Conversion des colonnes Texte/Selectbox (pour éviter l'erreur FLOAT)
    # On force la conversion des colonnes de texte/select box à string et on remplace les NaN par ''
    cols_a_convertir_str = ['Remarque', 'Localisation', 'Type Blessure', 'Gravité', 'Type Séance', 'Type Surface', 'Cycle Menstruel']
    for col in cols_a_convertir_str:
        if col in df_antecedent_joueuse.columns:
            # S'assurer que les valeurs non renseignées sont des chaînes vides pour st.data_editor
            df_antecedent_joueuse[col] = df_antecedent_joueuse[col].fillna('').astype(str)

    # 2b. Recalcul des Jours Absents (utilise Date RTP comme date de fin)
    # Assurez-vous que Date RTP est un objet datetime pour le calcul
    df_antecedent_joueuse['Date RTP'] = pd.to_datetime(df_antecedent_joueuse['Date RTP']) 
    date_fin_calc = df_antecedent_joueuse['Date RTP'].fillna(pd.to_datetime(date.today()))
    df_antecedent_joueuse['Jours Absents'] = (date_fin_calc - df_antecedent_joueuse['Date blessure']).dt.days
    
    # 2c. Nettoyage des dates après calcul (pas de conversion supplémentaire nécessaire)
    
else:
    # 🚨 Si le DF est vide, assurez-vous qu'il contient toutes les colonnes requises.
    colonnes_base = list(df_antecedent_historique.columns) 
    if 'Jours Absents' not in colonnes_base:
        colonnes_base.append('Jours Absents')
        
    df_antecedent_joueuse = pd.DataFrame(columns=colonnes_base)

    
# --- Définition des Options et Colonnes (Identique) ---
config_colonnes_antecedent = {
    'Date blessure': st.column_config.DateColumn("Date blessure", format="YYYY/MM/DD", required=True),
    'Localisation': st.column_config.SelectboxColumn("Localisation", options=OPTIONS_LOCALISATION, required=True),
    'Type Blessure': st.column_config.SelectboxColumn("Type Blessure", options=OPTIONS_TYPE_BLESSURE, required=True),
    'Gravité': st.column_config.SelectboxColumn("Gravité", options=OPTIONS_GRAVITE),
    'SC/AC': st.column_config.SelectboxColumn("SC/AC", options=OPTIONS_CONTACT),
    'Type Séance': st.column_config.SelectboxColumn("Type Séance", options=OPTIONS_SEANCE),
    'Type Surface': st.column_config.SelectboxColumn("Type Surface", options=OPTIONS_SURFACE),
    'Cycle Menstruel': st.column_config.SelectboxColumn("Cycle Menstruel", options=OPTIONS_CYCLE),
    'Date RTR': st.column_config.DateColumn("Date RTR", format="YYYY/MM/DD"),
    'Date Réa': st.column_config.DateColumn("Date Réa", format="YYYY/MM/DD"),
    'Date RTP': st.column_config.DateColumn("Date RTP", format="YYYY/MM/DD"),
    'Date RTC': st.column_config.DateColumn("Date RTC", format="YYYY/MM/DD"),
    'Remarque': st.column_config.TextColumn("Remarque", width="large"),
    
    # Colonnes d'information (non éditables)
    'Prénom': st.column_config.TextColumn("Prénom", disabled=True),
    'NOM': st.column_config.TextColumn("NOM", disabled=True),
    'Jours Absents': st.column_config.NumberColumn("Jours Absents", format="%d j", disabled=True)
}

# --- Colonnes à afficher et éditer ---
colonnes_edition = list(config_colonnes_antecedent.keys())

# --- Le st.data_editor remplace le tableau et le formulaire ---
df_antecedent_modifiees = st.data_editor(
    df_antecedent_joueuse[colonnes_edition],
    column_config=config_colonnes_antecedent,
    hide_index=True,
    num_rows="dynamic", # <-- Permet l'ajout et la suppression de lignes
    key="editor_antecedent"
)

# ----------------------------------------------------------------------
# SAUVEGARDE ET LOGIQUE DES CHANGEMENTS (Identique à votre code)
# ----------------------------------------------------------------------

# 1. Nettoyage : On ne garde que les lignes avec une date de début
df_antecedent_modifiees_clean = df_antecedent_modifiees.dropna(subset=['Date blessure'])

# 2. Ajout des clés (Prénom/NOM) pour les nouvelles lignes
if not df_antecedent_modifiees_clean.empty:
    # Utilisation de .loc pour éviter le SettingWithCopyWarning
    df_antecedent_modifiees_clean.loc[:, 'Prénom'] = prenom_j
    df_antecedent_modifiees_clean.loc[:, 'NOM'] = nom_j

# 3. Comparaison avec l'historique global pour détecter les changements
# On prépare le DF historique SANS la joueuse actuelle pour la reconstruction
df_historique_sans_joueuse = df_antecedent_historique[
    (df_antecedent_historique['Prénom'] != prenom_j) | 
    (df_antecedent_historique['NOM'] != nom_j)
].copy()

# Si le nombre de lignes (ajouts/suppressions) ou les valeurs changent
if df_antecedent_modifiees_clean.shape[0] != df_antecedent_joueuse.shape[0] or not df_antecedent_modifiees_clean.equals(df_antecedent_joueuse[colonnes_edition].dropna(subset=['Date blessure'])):
    
    st.warning(f"⚠️ {df_antecedent_modifiees_clean.shape[0]} lignes de blessure à sauvegarder/mettre à jour.")
    
    # Le bouton de sauvegarde doit être placé EN DEHORS du data_editor
    if st.button("💾 SAUVEGARDER L'HISTORIQUE DES BLESSURES", key="save_blessures"):
        
        # Reconstruire le DataFrame global
        df_final_antecedent = pd.concat([df_historique_sans_joueuse, df_antecedent_modifiees_clean], ignore_index=True)
        
        # Sauvegarde
        if sauvegarder_df_global(df_final_antecedent, FICHIER_ANTECEDENTS):
            st.success("✅ Historique des blessures mis à jour.")
            st.cache_data.clear()
            st.rerun()
else:
    st.info("Aucune modification en attente.")

##################################
st.markdown("---")
st.title("Suivi testing")

st.header("🔬 1. Suivi des Tests Isocinétiques")

colonne_date = ['Date Test']

for col in colonne_date:
    if col in df_isocinetisme_historique.columns:
        # Convertit les valeurs en dates. Les valeurs invalides (NaN/FLOAT) deviennent NaT.
        df_isocinetisme_historique[col] = pd.to_datetime(
            df_isocinetisme_historique[col], 
            errors='coerce'
        )

# 1. Filtrage initial et copie
df_isocinetisme_joueuse = df_isocinetisme_historique[
    (df_isocinetisme_historique['Prénom'] == prenom_j) & 
    (df_isocinetisme_historique['NOM'] == nom_j)
].sort_values(by='Date Test', ascending=False).copy()

# 2. Préparation des colonnes pour le st.data_editor (y compris le calcul)

if not df_isocinetisme_joueuse.empty:
    
    # 2a. Conversion des colonnes Texte/Selectbox (pour éviter l'erreur FLOAT) On force la conversion des colonnes de texte/select box à string et on remplace les NaN par ''
    col_a_convertir_str = ['Remarque']
    for col in col_a_convertir_str:
        if col in df_isocinetisme_joueuse.columns:
            # S'assurer que les valeurs non renseignées sont des chaînes vides pour st.data_editor
            df_isocinetisme_joueuse[col] = df_isocinetisme_joueuse[col].fillna('').astype(str)
    
    # 2b. 💡 NOUVEAU : Calculer les ratios pour l'affichage
    df_isocinetisme_joueuse = calculer_ratios_isocinetisme(df_isocinetisme_joueuse)

    # 2c. Assurez-vous d'avoir toutes les colonnes de ratio si le DF est vide
else:
        colonnes_base = calculer_ratios_isocinetisme(df_isocinetisme_historique.head(0)).columns
        df_isocinetisme_joueuse = pd.DataFrame(columns=colonnes_base)
        colonnes_base = list(df_isocinetisme_historique.columns) 
        df_isocinetisme_joueuse = pd.DataFrame(columns=colonnes_base)

# --- Définition des Options et Colonnes (Identique) ---
config_colonnes_isocinetisme = {
    'Date Test': st.column_config.DateColumn("Date Test", format="YYYY/MM/DD", required=True),
    'Remarque': st.column_config.TextColumn("Remarque", width="large"),
    'Q60° D': st.column_config.NumberColumn("Q60° D", format="%.1f", help="Couple de force maximale"),
    'Q60° G': st.column_config.NumberColumn("Q60° G", format="%.1f", help="Couple de force maximale"),
    'Dif Q60°': st.column_config.NumberColumn("Dif Q60°", format="%.1f", help="Différence de couple"),
    'IJ60° D': st.column_config.NumberColumn("IJ60° D", format="%.1f", help="Couple de force maximale"),
    'IJ60° G': st.column_config.NumberColumn("IJ60° G", format="%.1f", help="Couple de force maximale"),
    'Dif IJ60°': st.column_config.NumberColumn("Dif IJ60°", format="%.1f", help="Différence de couple"),
    'Q240° D': st.column_config.NumberColumn("Q240° D", format="%.1f", help="Couple de force maximale"),
    'Q240° G': st.column_config.NumberColumn("Q240° G", format="%.1f", help="Couple de force maximale"),
    'Dif Q240°': st.column_config.NumberColumn("Dif Q240°", format="%.1f", help="Différence de couple"),
    'IJ240° D': st.column_config.NumberColumn("IJ240° D", format="%.1f", help="Couple de force maximale"),
    'IJ240° G': st.column_config.NumberColumn("IJ240° G", format="%.1f", help="Couple de force maximale"),
    'Dif IJ240°': st.column_config.NumberColumn("Dif IJ240°", format="%.1f", help="Différence de couple"),
    'IJExc D': st.column_config.NumberColumn("IJExc D", format="%.1f", help="Couple de force maximale"),
    'IJExc G': st.column_config.NumberColumn("IJExc G", format="%.1f", help="Couple de force maximale"),
    'Dif IJExc': st.column_config.NumberColumn("Dif IJExc", format="%.1f", help="Différence de couple"),
    
    # 🚨 NOUVELLES COLONNES DE RATIO (Calculées et Désactivées)
    'Ratio IJ/Q60° D': st.column_config.NumberColumn("Ratio IJ/Q60° D", format="%.2f", disabled=True),
    'Ratio IJ/Q60° G': st.column_config.NumberColumn("Ratio IJ/Q60° G", format="%.2f", disabled=True),
    'Ratio IJ/Q240° D': st.column_config.NumberColumn("Ratio IJ/Q240° D", format="%.2f", disabled=True),
    'Ratio IJ/Q240° G': st.column_config.NumberColumn("Ratio IJ/Q240° G", format="%.2f", disabled=True),
    'Ratio Mixte D': st.column_config.NumberColumn("Ratio Mixte D", format="%.2f", disabled=True),
    'Ratio Mixte G': st.column_config.NumberColumn("Ratio Mixte G", format="%.2f", disabled=True),
    
    # Colonnes d'information (non éditables)
    'Prénom': st.column_config.TextColumn("Prénom", disabled=True),
    'NOM': st.column_config.TextColumn("NOM", disabled=True),
}

# --- Colonnes à afficher et éditer ---
colonnes_edition = list(config_colonnes_isocinetisme.keys())

# --- Le st.data_editor remplace le tableau et le formulaire ---
df_isocinetisme_modifiees = st.data_editor(
    df_isocinetisme_joueuse[colonnes_edition],
    column_config=config_colonnes_isocinetisme,
    hide_index=True,
    num_rows="dynamic", # <-- Permet l'ajout et la suppression de lignes
    key="editor_isocinetisme"
)

# ----------------------------------------------------------------------
# SAUVEGARDE ET LOGIQUE DES CHANGEMENTS
# ----------------------------------------------------------------------

# 1. Nettoyage : On ne garde que les lignes avec une date de début
df_isocinetisme_modifiees_clean = df_isocinetisme_modifiees.dropna(subset=['Date Test'])

# 2. Ajout des clés (Prénom/NOM) pour les nouvelles lignes
if not df_isocinetisme_modifiees_clean.empty:
    df_isocinetisme_modifiees_clean.loc[:, 'Prénom'] = prenom_j
    df_isocinetisme_modifiees_clean.loc[:, 'NOM'] = nom_j

# 3. Comparaison avec l'historique global pour détecter les changements
# On prépare le DF historique SANS la joueuse actuelle pour la reconstruction
df_historique_sans_joueuse = df_isocinetisme_historique[
    (df_isocinetisme_historique['Prénom'] != prenom_j) | 
    (df_isocinetisme_historique['NOM'] != nom_j)
].copy()

# Si le nombre de lignes (ajouts/suppressions) ou les valeurs changent
if df_isocinetisme_modifiees_clean.shape[0] != df_isocinetisme_joueuse.shape[0] or not df_isocinetisme_modifiees_clean.equals(df_isocinetisme_joueuse[colonnes_edition].dropna(subset=['Date Test'])):
    
    st.warning(f"⚠️ {df_isocinetisme_modifiees_clean.shape[0]} lignes de tests isociténique à sauvegarder/mettre à jour.")
    
    # Le bouton de sauvegarde doit être placé EN DEHORS du data_editor
    if st.button("💾 SAUVEGARDER L'HISTORIQUE DES TESTS ISOCINETIQUE", key="save_isocinetisme"):
        
        # Re-calculer les ratios sur le DataFrame modifié par l'utilisateur
        df_a_sauvegarder = calculer_ratios_isocinetisme(df_isocinetisme_modifiees_clean)

        # Reconstruire le DataFrame global
        df_final_isocinetisme = pd.concat([df_historique_sans_joueuse, df_a_sauvegarder], ignore_index=True)
        
        # Sauvegarde
        if sauvegarder_df_global(df_final_isocinetisme, FICHIER_ISOCINETISME):
            st.success("✅ Historique des tests isocinétique mis à jour.")
            st.cache_data.clear()
            st.rerun()
else:
    st.info("Aucune modification en attente.")

# ----------------------------------------------------------------------
# 2. SUIVI HOP TEST (Logique Corrigée pour la KeyError)
# ----------------------------------------------------------------------

st.markdown("---")
st.header("🏃 2. Suivi des Hop Tests")

# Définition complète des colonnes d'entrée du Hop Test pour l'initialisation
COLONNES_ENTREE_HOP_TEST = [
    'Date Test',
    'SHT D1', 'SHT D2', 'SHT D3', 'Nbr SHT D', 
    'SHT G1', 'SHT G2', 'SHT G3', 'Nbr SHT G', 
    'THT D1', 'THT D2', 'THT D3', 'Nbr THT D', 
    'THT G1', 'THT G2', 'THT G3', 'Nbr THT G', 
    'CHT D1', 'CHT D2', 'CHT D3', 'Nbr CHT D', 
    'CHT G1', 'CHT G2', 'CHT G3', 'Nbr CHT G', 
    'LHT D', 'LHT G', 
    'Prénom', 'NOM'
]

# Colonnes calculées pour les Hop Tests (récupérées de la fonction de calcul)
COLONNES_CALCULEES_HOP_TEST = [
    'Mean SHT D', 'Mean SHT G', 'Max SHT D', 'Max SHT G', 'Sym SHT', 
    'Mean THT D', 'Mean THT G', 'Max THT D', 'Max THT G', 'Sym THT', 
    'Mean CHT D', 'Mean CHT G', 'Max CHT D', 'Max CHT G', 'Sym CHT'
]

# Toutes les colonnes à afficher/éditer
COLONNES_A_AFFICHER_HOP = COLONNES_ENTREE_HOP_TEST + COLONNES_CALCULEES_HOP_TEST

# 1. Filtrage initial et copie
df_hop_test_joueuse = df_hop_test_historique[
    (df_hop_test_historique['Prénom'] == prenom_j) & 
    (df_hop_test_historique['NOM'] == nom_j)
].sort_values(by='Date Test', ascending=False).copy()


# 2. Préparation et Calcul des métriques
if not df_hop_test_joueuse.empty:
    df_hop_test_joueuse = calculer_metriques_hop_test(df_hop_test_joueuse)
else:
    all_cols_to_create = COLONNES_A_AFFICHER_HOP 
    df_hop_test_joueuse = pd.DataFrame(columns=all_cols_to_create)


# ----------------------------------------------------------------------
# DÉFINITION ET AFFICHAGE HOP TEST
# ----------------------------------------------------------------------

# Définition des configurations de colonnes pour Streamlit (Doit être complet)
config_colonnes_hop = {
    'Date Test': st.column_config.DateColumn("Date Test", format="YYYY/MM/DD", required=True),
    'SHT D1': st.column_config.NumberColumn("SHT D1 (cm)", format="%.1f"),
    'SHT D2': st.column_config.NumberColumn("SHT D2 (cm)", format="%.1f"),
    'SHT D3': st.column_config.NumberColumn("SHT D3 (cm)", format="%.1f"),
    'Nbr SHT D': st.column_config.NumberColumn("Nbr SHT D", format="%d", help="Nombre de sauts SHT D valides"),
    'SHT G1': st.column_config.NumberColumn("SHT G1 (cm)", format="%.1f"),
    'SHT G2': st.column_config.NumberColumn("SHT G2 (cm)", format="%.1f"),
    'SHT G3': st.column_config.NumberColumn("SHT G3 (cm)", format="%.1f"),
    'Nbr SHT G': st.column_config.NumberColumn("Nbr SHT G", format="%d", help="Nombre de sauts SHT G valides"),
    'THT D1': st.column_config.NumberColumn("THT D1 (cm)", format="%.1f"),
    'THT D2': st.column_config.NumberColumn("THT D2 (cm)", format="%.1f"),
    'THT D3': st.column_config.NumberColumn("THT D3 (cm)", format="%.1f"),
    'Nbr THT D': st.column_config.NumberColumn("Nbr THT D", format="%d"),
    'THT G1': st.column_config.NumberColumn("THT G1 (cm)", format="%.1f"),
    'THT G2': st.column_config.NumberColumn("THT G2 (cm)", format="%.1f"),
    'THT G3': st.column_config.NumberColumn("THT G3 (cm)", format="%.1f"),
    'Nbr THT G': st.column_config.NumberColumn("Nbr THT G", format="%d"),
    'CHT D1': st.column_config.NumberColumn("CHT D1 (cm)", format="%.1f"),
    'CHT D2': st.column_config.NumberColumn("CHT D2 (cm)", format="%.1f"),
    'CHT D3': st.column_config.NumberColumn("CHT D3 (cm)", format="%.1f"),
    'Nbr CHT D': st.column_config.NumberColumn("Nbr CHT D", format="%d"),
    'CHT G1': st.column_config.NumberColumn("CHT G1 (cm)", format="%.1f"),
    'CHT G2': st.column_config.NumberColumn("CHT G2 (cm)", format="%.1f"),
    'CHT G3': st.column_config.NumberColumn("CHT G3 (cm)", format="%.1f"),
    'Nbr CHT G': st.column_config.NumberColumn("Nbr CHT G", format="%d"),
    'LHT D': st.column_config.NumberColumn("LHT D", format="%.1f"),
    'LHT G': st.column_config.NumberColumn("LHT G", format="%.1f"),
    
    # Métriques calculées (désactivées)
    'Mean SHT D': st.column_config.NumberColumn("Mean SHT D", format="%.2f", disabled=True),
    'Mean SHT G': st.column_config.NumberColumn("Mean SHT G", format="%.2f", disabled=True),
    'Max SHT D': st.column_config.NumberColumn("Max SHT D", format="%.2f", disabled=True),
    'Max SHT G': st.column_config.NumberColumn("Max SHT G", format="%.2f", disabled=True),
    'Sym SHT': st.column_config.NumberColumn("Sym SHT (%)", format="%.2f", disabled=True),
    'Mean THT D': st.column_config.NumberColumn("Mean THT D", format="%.2f", disabled=True),
    'Mean THT G': st.column_config.NumberColumn("Mean THT G", format="%.2f", disabled=True),
    'Max THT D': st.column_config.NumberColumn("Max THT D", format="%.2f", disabled=True),
    'Max THT G': st.column_config.NumberColumn("Max THT G", format="%.2f", disabled=True),
    'Sym THT': st.column_config.NumberColumn("Sym THT (%)", format="%.2f", disabled=True),
    'Mean CHT D': st.column_config.NumberColumn("Mean CHT D", format="%.2f", disabled=True),
    'Mean CHT G': st.column_config.NumberColumn("Mean CHT G", format="%.2f", disabled=True),
    'Max CHT D': st.column_config.NumberColumn("Max CHT D", format="%.2f", disabled=True),
    'Max CHT G': st.column_config.NumberColumn("Max CHT G", format="%.2f", disabled=True),
    'Sym CHT': st.column_config.NumberColumn("Sym CHT (%)", format="%.2f", disabled=True),
    
    'Prénom': st.column_config.TextColumn("Prénom", disabled=True),
    'NOM': st.column_config.TextColumn("NOM", disabled=True),
}


# On utilise la liste explicite de colonnes pour garantir l'ordre et la présence
colonnes_a_afficher_hop_final = [col for col in COLONNES_A_AFFICHER_HOP if col in df_hop_test_joueuse.columns]
config_hop_filtree = {k: v for k, v in config_colonnes_hop.items() if k in colonnes_a_afficher_hop_final}


st.write(f"Historique pour **{joueuse_selectionnee}** ({df_hop_test_joueuse.shape[0]} tests)")

df_hop_test_modifiees = st.data_editor(
    # Utiliser la liste explicite de colonnes pour s'assurer qu'elles apparaissent
    df_hop_test_joueuse[colonnes_a_afficher_hop_final],
    column_config=config_hop_filtree,
    hide_index=True,
    num_rows="dynamic", 
    key="editor_hop_test"
)

# ----------------------------------------------------------------------
# SAUVEGARDE HOP TEST
# ----------------------------------------------------------------------

# 1. Nettoyage : On ne garde que les lignes avec une date de test
df_hop_test_modifiees_clean = df_hop_test_modifiees.dropna(subset=['Date Test'])

# 2. Ajout des clés (Prénom/NOM) pour les nouvelles lignes
if not df_hop_test_modifiees_clean.empty:
    df_hop_test_modifiees_clean.loc[:, 'Prénom'] = prenom_j
    df_hop_test_modifiees_clean.loc[:, 'NOM'] = nom_j

# 3. Comparaison avec l'historique global pour détecter les changements
df_historique_sans_joueuse_hop = df_hop_test_historique[
    (df_hop_test_historique['Prénom'] != prenom_j) | 
    (df_hop_test_historique['NOM'] != nom_j)
].copy()

# Filtrage du DF original avant comparaison (pour éviter les problèmes de colonnes calculées)
df_original_hop_compare = df_hop_test_joueuse[colonnes_a_afficher_hop_final].dropna(subset=['Date Test'])

# Si le nombre de lignes ou les valeurs changent
if df_hop_test_modifiees_clean.shape[0] != df_original_hop_compare.shape[0] or not df_hop_test_modifiees_clean.equals(df_original_hop_compare):
    
    st.warning(f"⚠️ {df_hop_test_modifiees_clean.shape[0]} lignes de Hop Test à sauvegarder/mettre à jour.")
    
    if st.button("💾 SAUVEGARDER L'HISTORIQUE DES HOP TESTS", key="save_hop_test"):
        
        # Re-calculer les métriques sur le DataFrame modifié par l'utilisateur
        df_a_sauvegarder = calculer_metriques_hop_test(df_hop_test_modifiees_clean)

        # 🚨 CORRECTION DU FUTUREWARNING 🚨
        # On utilise une liste et on s'assure que seuls les DataFrames non vides sont inclus
        dfs_to_concat = []
        
        # 1. Ajouter l'historique des autres joueuses si non vide
        if not df_historique_sans_joueuse_hop.empty:
            dfs_to_concat.append(df_historique_sans_joueuse_hop)
            
        # 2. Ajouter les données modifiées/ajoutées de la joueuse actuelle si non vide
        if not df_a_sauvegarder.empty:
            dfs_to_concat.append(df_a_sauvegarder)
            
        
        if dfs_to_concat:
             # Reconstruire le DataFrame global
            df_final_hop_test = pd.concat(dfs_to_concat, ignore_index=True)

            # Sauvegarde
            if sauvegarder_df_global(df_final_hop_test, FICHIER_HOP_TEST):
                st.success("✅ Historique des Hop Tests mis à jour.")
                st.cache_data.clear()
                st.rerun()
            else:
                st.error("❌ Échec de la sauvegarde.")
        else:
             # Cas où il n'y a absolument rien à sauvegarder (historique vide + nouveau DF vide)
             st.info("Aucune donnée à enregistrer : L'historique et les modifications sont vides.")
             
else:
    st.info("Aucune modification en attente (Hop Test).")



# ----------------------------------------------------------------------
# 3. SUIVI SAUTS (NOUVEAU)
# ----------------------------------------------------------------------
st.markdown("---")
st.header("⬆️ 3. Suivi des Tests de Sauts")

df_sauts_joueuse = df_sauts_historique[
    (df_sauts_historique['Prénom'] == prenom_j) & 
    (df_sauts_historique['NOM'] == nom_j)
].sort_values(by='Date Test', ascending=False).copy()

if not df_sauts_joueuse.empty:
    df_sauts_joueuse = calculer_metriques_sauts(df_sauts_joueuse)
else:
    colonnes_de_base_avec_metriques_sauts = calculer_metriques_sauts(df_sauts_historique.head(0).copy()).columns
    all_cols_sauts = list(set(list(df_sauts_historique.columns) + list(colonnes_de_base_avec_metriques_sauts)))
    df_sauts_joueuse = pd.DataFrame(columns=all_cols_sauts)

# Définition des colonnes attendues
colonnes_edition_sauts = [
    'Date Test', 'CMJ 1', 'CMJ 2', 'CMJ 3', 'Max CMJ',
    'CMJ Bras 1', 'CMJ Bras 2', 'CMJ Bras 3', 'Max CMJ Bras',
    'CMJ 1J D1', 'CMJ 1J D2', 'CMJ 1J D3', 'Max CMJ 1J D',
    'CMJ 1J G1', 'CMJ 1J G2', 'CMJ 1J G3', 'Max CMJ 1J G',
    'SRJT 5 Mean 1', 'SRJT 5 Mean 2', 'SRJT 5 Mean 3', 'Max SRJT 5 Mean',
    'SRJT 5 RSI 1', 'SRJT 5 RSI 2', 'SRJT 5 RSI 3', 'Max SRJT 5 RSI',
    'Prénom', 'NOM'
]

config_colonnes_sauts = {
    'Date Test': st.column_config.DateColumn("Date Test", format="YYYY/MM/DD", required=True),
    'CMJ 1': st.column_config.NumberColumn("CMJ 1 (cm)", format="%.1f"),
    'CMJ 2': st.column_config.NumberColumn("CMJ 2 (cm)", format="%.1f"),
    'CMJ 3': st.column_config.NumberColumn("CMJ 3 (cm)", format="%.1f"),
    'Max CMJ': st.column_config.NumberColumn("Max CMJ (cm)", format="%.2f", disabled=True, help="Maximum des 3 essais CMJ"),
    'CMJ Bras 1': st.column_config.NumberColumn("CMJ Bras 1 (cm)", format="%.1f"),
    'CMJ Bras 2': st.column_config.NumberColumn("CMJ Bras 2 (cm)", format="%.1f"),
    'CMJ Bras 3': st.column_config.NumberColumn("CMJ Bras 3 (cm)", format="%.1f"),
    'Max CMJ Bras': st.column_config.NumberColumn("Max CMJ Bras (cm)", format="%.2f", disabled=True),
    'CMJ 1J D1': st.column_config.NumberColumn("CMJ 1J D1 (cm)", format="%.1f"),
    'CMJ 1J D2': st.column_config.NumberColumn("CMJ 1J D2 (cm)", format="%.1f"),
    'CMJ 1J D3': st.column_config.NumberColumn("CMJ 1J D3 (cm)", format="%.1f"),
    'Max CMJ 1J D': st.column_config.NumberColumn("Max CMJ 1J D (cm)", format="%.2f", disabled=True),
    'CMJ 1J G1': st.column_config.NumberColumn("CMJ 1J G1 (cm)", format="%.1f"),
    'CMJ 1J G2': st.column_config.NumberColumn("CMJ 1J G2 (cm)", format="%.1f"),
    'CMJ 1J G3': st.column_config.NumberColumn("CMJ 1J G3 (cm)", format="%.1f"),
    'Max CMJ 1J G': st.column_config.NumberColumn("Max CMJ 1J G (cm)", format="%.2f", disabled=True),
    'SRJT 5 Mean 1': st.column_config.NumberColumn("SRJT 5 Mean 1 (cm)", format="%.1f"),
    'SRJT 5 Mean 2': st.column_config.NumberColumn("SRJT 5 Mean 2 (cm)", format="%.1f"),
    'SRJT 5 Mean 3': st.column_config.NumberColumn("SRJT 5 Mean 3 (cm)", format="%.1f"),
    'Max SRJT 5 Mean': st.column_config.NumberColumn("Max SRJT 5 Mean (cm)", format="%.2f", disabled=True),
    'SRJT 5 RSI 1': st.column_config.NumberColumn("SRJT 5 RSI 1", format="%.2f"),
    'SRJT 5 RSI 2': st.column_config.NumberColumn("SRJT 5 RSI 2", format="%.2f"),
    'SRJT 5 RSI 3': st.column_config.NumberColumn("SRJT 5 RSI 3", format="%.2f"),
    'Max SRJT 5 RSI': st.column_config.NumberColumn("Max SRJT 5 RSI", format="%.2f", disabled=True),
    'Prénom': st.column_config.TextColumn("Prénom", disabled=True),
    'NOM': st.column_config.TextColumn("NOM", disabled=True),
}

colonnes_a_afficher_sauts = [col for col in colonnes_edition_sauts if col in df_sauts_joueuse.columns]
config_sauts_filtree = {k: v for k, v in config_colonnes_sauts.items() if k in colonnes_a_afficher_sauts}

st.write(f"Historique pour **{joueuse_selectionnee}** ({df_sauts_joueuse.shape[0]} tests)")
df_sauts_modifiees = st.data_editor(
    df_sauts_joueuse[colonnes_a_afficher_sauts],
    column_config=config_sauts_filtree,
    hide_index=True,
    num_rows="dynamic",
    key="editor_sauts"
)

# --- SAUVEGARDE SAUTS ---
df_sauts_modifiees_clean = df_sauts_modifiees.dropna(subset=['Date Test'])
if not df_sauts_modifiees_clean.empty:
    df_sauts_modifiees_clean.loc[:, 'Prénom'] = prenom_j
    df_sauts_modifiees_clean.loc[:, 'NOM'] = nom_j

df_historique_sans_joueuse_sauts = df_sauts_historique[
    (df_sauts_historique['Prénom'] != prenom_j) | 
    (df_sauts_historique['NOM'] != nom_j)
].copy()

df_original_sauts_compare = df_sauts_joueuse[colonnes_a_afficher_sauts].dropna(subset=['Date Test'])

if df_sauts_modifiees_clean.shape[0] != df_original_sauts_compare.shape[0] or not df_sauts_modifiees_clean.equals(df_original_sauts_compare):
    st.warning(f"⚠️ {df_sauts_modifiees_clean.shape[0]} lignes de Sauts à sauvegarder/mettre à jour.")
    if st.button("💾 SAUVEGARDER L'HISTORIQUE DES SAUTS", key="save_sauts"):
        df_a_sauvegarder = calculer_metriques_sauts(df_sauts_modifiees_clean)
        df_final_sauts = pd.concat([df_historique_sans_joueuse_sauts, df_a_sauvegarder], ignore_index=True)
        if sauvegarder_df_global(df_final_sauts, FICHIER_SAUTS):
            st.success("✅ Historique des Sauts mis à jour.")
            st.cache_data.clear()
            st.rerun()
else:
    st.info("Aucune modification en attente (Sauts).")


# ----------------------------------------------------------------------
# 4. SUIVI DYNAMOMÉTRIE (NOUVEAU)
# ----------------------------------------------------------------------
st.markdown("---")
st.header("💪 4. Suivi de la Dynamométrie")

df_dynamo_joueuse = df_dynamo_historique[
    (df_dynamo_historique['Prénom'] == prenom_j) & 
    (df_dynamo_historique['NOM'] == nom_j)
].sort_values(by='Date Test', ascending=False).copy()

if not df_dynamo_joueuse.empty:
    df_dynamo_joueuse = calculer_metriques_dynamo(df_dynamo_joueuse)
else:
    colonnes_de_base_avec_metriques_dynamo = calculer_metriques_dynamo(df_dynamo_historique.head(0).copy()).columns
    all_cols_dynamo = list(set(list(df_dynamo_historique.columns) + list(colonnes_de_base_avec_metriques_dynamo)))
    df_dynamo_joueuse = pd.DataFrame(columns=all_cols_dynamo)

# Définition des colonnes attendues
colonnes_edition_dynamo = [
    'Date Test', 'Soléaire D', 'Soléaire G', 'Soléaire H barre', 'Sym soléaire',
    'Gastro D', 'Gastro G', 'Sym gastro',
    'Tibial post D', 'Tibial post G', 'Sym tibial post',
    'Fibulaire D', 'Fibulaire G', 'Sym fibulaire',
    'Abducteur D', 'Abducteur G', 'Sym abducteur',
    'Adducteur D', 'Adducteur G', 'Sym adducteur',
    'Ratio fibulaire / tibial post D', 'Ratio fibulaire / tibial post G',
    'Ratio ADD / ABD D', 'Ratio ADD / ABD G',
    'Prénom', 'NOM'
]

config_colonnes_dynamo = {
    'Date Test': st.column_config.DateColumn("Date Test", format="YYYY/MM/DD", required=True),
    'Soléaire D': st.column_config.NumberColumn("Soléaire D", format="%.1f", help="Force soléaire Droit"),
    'Soléaire G': st.column_config.NumberColumn("Soléaire G", format="%.1f", help="Force soléaire Gauche"),
    'Soléaire H barre': st.column_config.NumberColumn("Soléaire H barre", format="%.1f", help="Hauteur barre test mollet assis"),
    'Gastro D': st.column_config.NumberColumn("Gastro D", format="%.1f"),
    'Gastro G': st.column_config.NumberColumn("Gastro G", format="%.1f"),
    'Tibial post D': st.column_config.NumberColumn("Tibial post D", format="%.1f"),
    'Tibial post G': st.column_config.NumberColumn("Tibial post G", format="%.1f"),
    'Fibulaire D': st.column_config.NumberColumn("Fibulaire D", format="%.1f"),
    'Fibulaire G': st.column_config.NumberColumn("Fibulaire G", format="%.1f"),
    'Abducteur D': st.column_config.NumberColumn("Abducteur D", format="%.1f"),
    'Abducteur G': st.column_config.NumberColumn("Abducteur G", format="%.1f"),
    'Adducteur D': st.column_config.NumberColumn("Adducteur D", format="%.1f"),
    'Adducteur G': st.column_config.NumberColumn("Adducteur G", format="%.1f"),
    
    'Sym soléaire': st.column_config.NumberColumn("Sym soléaire (%)", format="%.2f", disabled=True),
    'Sym gastro': st.column_config.NumberColumn("Sym gastro (%)", format="%.2f", disabled=True),
    'Sym tibial post': st.column_config.NumberColumn("Sym tibial post (%)", format="%.2f", disabled=True),
    'Sym fibulaire': st.column_config.NumberColumn("Sym fibulaire (%)", format="%.2f", disabled=True),
    'Sym abducteur': st.column_config.NumberColumn("Sym abducteur (%)", format="%.2f", disabled=True),
    'Sym adducteur': st.column_config.NumberColumn("Sym adducteur (%)", format="%.2f", disabled=True),
    'Ratio fibulaire / tibial post D': st.column_config.NumberColumn("Ratio Fib/TibPost D", format="%.2f", disabled=True),
    'Ratio fibulaire / tibial post G': st.column_config.NumberColumn("Ratio Fib/TibPost G", format="%.2f", disabled=True),
    'Ratio ADD / ABD D': st.column_config.NumberColumn("Ratio ADD/ABD D (%)", format="%.2f", disabled=True),
    'Ratio ADD / ABD G': st.column_config.NumberColumn("Ratio ADD/ABD G (%)", format="%.2f", disabled=True),

    'Prénom': st.column_config.TextColumn("Prénom", disabled=True),
    'NOM': st.column_config.TextColumn("NOM", disabled=True),
}

colonnes_a_afficher_dynamo = [col for col in colonnes_edition_dynamo if col in df_dynamo_joueuse.columns]
config_dynamo_filtree = {k: v for k, v in config_colonnes_dynamo.items() if k in colonnes_a_afficher_dynamo}

st.write(f"Historique pour **{joueuse_selectionnee}** ({df_dynamo_joueuse.shape[0]} tests)")
df_dynamo_modifiees = st.data_editor(
    df_dynamo_joueuse[colonnes_a_afficher_dynamo],
    column_config=config_dynamo_filtree,
    hide_index=True,
    num_rows="dynamic",
    key="editor_dynamo"
)

# --- SAUVEGARDE DYNAMOMÉTRIE ---
df_dynamo_modifiees_clean = df_dynamo_modifiees.dropna(subset=['Date Test'])
if not df_dynamo_modifiees_clean.empty:
    df_dynamo_modifiees_clean.loc[:, 'Prénom'] = prenom_j
    df_dynamo_modifiees_clean.loc[:, 'NOM'] = nom_j

df_historique_sans_joueuse_dynamo = df_dynamo_historique[
    (df_dynamo_historique['Prénom'] != prenom_j) | 
    (df_dynamo_historique['NOM'] != nom_j)
].copy()

df_original_dynamo_compare = df_dynamo_joueuse[colonnes_a_afficher_dynamo].dropna(subset=['Date Test'])

if df_dynamo_modifiees_clean.shape[0] != df_original_dynamo_compare.shape[0] or not df_dynamo_modifiees_clean.equals(df_original_dynamo_compare):
    st.warning(f"⚠️ {df_dynamo_modifiees_clean.shape[0]} lignes de Dynamométrie à sauvegarder/mettre à jour.")
    if st.button("💾 SAUVEGARDER L'HISTORIQUE DE DYNAMOMÉTRIE", key="save_dynamo"):
        df_a_sauvegarder = calculer_metriques_dynamo(df_dynamo_modifiees_clean)
        
        # ------------------------------------------------------------------
        # CORRECTION du FutureWarning lors de la concaténation de DataFrames vides
        # ------------------------------------------------------------------
        dfs_to_concat = []
        
        # 1. Ajouter l'historique des autres joueuses si non vide
        if not df_historique_sans_joueuse_dynamo.empty:
            dfs_to_concat.append(df_historique_sans_joueuse_dynamo)
            
        # 2. Ajouter les données modifiées/ajoutées de la joueuse actuelle si non vide
        if not df_a_sauvegarder.empty:
            dfs_to_concat.append(df_a_sauvegarder)
            
        
        if dfs_to_concat:
            # Reconstruire le DataFrame global en concaténant uniquement les DFs non vides
            df_final_dynamo = pd.concat(dfs_to_concat, ignore_index=True)
            
            if sauvegarder_df_global(df_final_dynamo, FICHIER_DYNAMO):
                st.success("✅ Historique de Dynamométrie mis à jour.")
                st.cache_data.clear()
                st.rerun()
            else:
                 st.error("❌ Échec de la sauvegarde.")
        else:
             st.info("Aucune donnée à enregistrer.")
        
else:
    st.info("Aucune modification en attente (Dynamométrie).")

    