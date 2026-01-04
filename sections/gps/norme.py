import streamlit as st
import pandas as pd
import numpy as np # Ajout de numpy pour les opérations numériques
from sections.gps.pipeline import load_and_merge_rpe, add_columns_session_rpe
from sections.constantes import cols_num, cols_cumul_sum

def prepare_base_df(df_gps_brut):
    """
    Effectue la fusion RPE, le calcul des métriques RPE, le filtrage de la population,
    et le nettoyage/préparation du DataFrame.
    """
    
    # 1. Fusionner les données GPS avec le RPE
    df_gps_avec_rpe = load_and_merge_rpe(df_gps_brut.copy())
    
    # Correction des types pour RPE si nécessaire
    df_gps_avec_rpe['Cardio'] = pd.to_numeric(df_gps_avec_rpe['Cardio'], errors='coerce')
    df_gps_avec_rpe['Muscu'] = pd.to_numeric(df_gps_avec_rpe['Muscu'], errors='coerce')
    
    # 2. Ajouter les colonnes s.RPE
    df_gps_avec_rpe = add_columns_session_rpe(df=df_gps_avec_rpe)
    
    # 3. Créer la métrique "Sprint (m)"
    df_gps_avec_rpe['Sprint (m)'] = df_gps_avec_rpe['SPR Total Distance (m)'] + df_gps_avec_rpe['SPR + Total Distance (m)']

    # 4. Filtrage de la population pour les statistiques (S, hors Gardienne, hors Présence = P)
    df_statistiques = df_gps_avec_rpe.loc[
        (df_gps_avec_rpe['Activity Name'].str.startswith('S')) & 
        (df_gps_avec_rpe['Position Name'] != 'Gardienne') & 
        (df_gps_avec_rpe['Presence'] != 'P')
    ].copy() 
    
    # 5. Nettoyage final
    df_statistiques = df_statistiques.drop(['Start Time', 'End Time', 'VHSR + SPR effort', 'Presence'], axis=1)
    df_statistiques['Date'] = pd.to_datetime(df_statistiques['Date'], format='%d-%m-%Y', errors='coerce').dt.strftime('%Y-%m-%d')
    df_statistiques['Jour semaine'] = df_statistiques['Jour semaine'].astype(str).str.strip()
    
    return df_statistiques

# ----------------------------------------------------
# 2. FONCTION DE CUMUL PAR JOUEUSE (Déplacée de votre code)
# ----------------------------------------------------

def cumuler_moyennes_individuelles(df_list, jours_labels, cols_a_cumuler):
    """
    Joint plusieurs DataFrames de moyennes et somme les métriques spécifiées.
    """
    if not df_list:
        return pd.DataFrame()
        
    df_base = df_list[0].copy()
    
    # On gère l'accumulation sans renommer les colonnes de sortie pour le cumul
    # La logique de renommage temporaire est essentielle pour la jointure si les DF sont différents
    
    # 1. Initialiser le DataFrame cumulé avec le premier DataFrame (DF le plus ancien)
    df_base = df_list[0].rename(columns={
        col: f'{col}_moy_{jours_labels[0]}' for col in df_list[0].columns if col != 'Name'
    })
    
    # 2. Jointure séquentielle de tous les autres DataFrames
    for i, df_next in enumerate(df_list[1:], 1):
        jour_label = jours_labels[i]
        
        # Renommer les colonnes du DF suivant (étape temporaire nécessaire pour la jointure)
        df_next_renom = df_next.rename(columns={
            col: f'{col}_moy_{jour_label}' for col in df_next.columns if col != 'Name'
        })
        
        # Jointure externe
        df_base = pd.merge(df_base, df_next_renom, on='Name', how='outer')

    # Remplacer les NaN par 0 avant de sommer
    df_base = df_base.fillna(0)
    
    # 3. Calcul de la SOMME (le cumul)
    df_resultat = df_base[['Name']].copy()
    
    for col in cols_a_cumuler:
        # Identifier toutes les colonnes de moyennes à sommer pour cette métrique
        colonnes_somme = [c for c in df_base.columns if c.startswith(f'{col}_moy_')]
        
        if colonnes_somme:
            # Calculer la somme des colonnes de moyennes
            df_resultat[col] = df_base[colonnes_somme].sum(axis=1)
    
    # Suppression des colonnes non pertinentes APRÈS le cumul
    # S'assurer que les colonnes à supprimer existent, sinon on aura une erreur
    cols_to_drop = [col for col in ['V max', 'Meterage Per Minute', 'Cardio', 'Muscu'] if col in df_resultat.columns]
    df_resultat = df_resultat.drop(cols_to_drop, axis=1, errors='ignore')

    return df_resultat.round(2)


# ----------------------------------------------------
# 3. CALCUL DES MOYENNES INDIVIDUELLES PAR JOUR
# ----------------------------------------------------

def calculer_moyennes_individuelles_par_jour(df_statistiques, cols_num):
    """
    Calcule les DataFrames de moyennes individuelles par jour J-1, J-2, J-3, J-4.
    """
    resultats = {}
    jours = ['J-1', 'J-2', 'J-3', 'J-4']
    
    for jour in jours:
        # 1. Filtrer pour le jour J-X
        df_joueuse_jour = df_statistiques[df_statistiques['Jour semaine'] == jour]
        
        # 2. Déterminer les métriques valides
        metrics_to_mean = [col for col in cols_num if col in df_joueuse_jour.columns]
        
        if not metrics_to_mean:
            resultats[jour] = pd.DataFrame({'Name': df_statistiques['Name'].unique()})
            continue

        # 3. Calculer l'agrégation
        agg_dict_mean = {col: 'mean' for col in metrics_to_mean}
        df_mean_par_joueuse = df_joueuse_jour.groupby('Name').agg(agg_dict_mean).reset_index()
        
        # 4. Stocker le résultat
        resultats[jour] = df_mean_par_joueuse.round(2)
        
    return resultats


# ----------------------------------------------------
# 4. FONCTION MAÎTRESSE POUR LE CHARGEMENT DANS LA SESSION
# ----------------------------------------------------

def calculer_toutes_references(df_gps_brut):
    """
    Fonction principale pour enchaîner le pré-traitement, le calcul des moyennes, 
    le cumul, et le stockage final des DataFrames de référence dans st.session_state.
    """
    
    # Utiliser une clé unique pour vérifier si le travail a déjà été fait.
    if 'DF_REFERENCES_CUMULEES' in st.session_state:
        references_cumulees = st.session_state['DF_REFERENCES_CUMULEES']
        references_jour = st.session_state.get('DF_REFERENCES_BRUTES', {})
        # Pour les besoins de la page Statistiques (si elle utilise toujours les anciennes clés)
        if 'ref_j1_cumule' not in st.session_state or 'ref_j1_brute' not in st.session_state:
            st.session_state['ref_j1_cumule'] = references_cumulees.get('J-1')
            st.session_state['ref_j2_cumule'] = references_cumulees.get('J-2')
            st.session_state['ref_j3_cumule'] = references_cumulees.get('J-3')
            st.session_state['ref_j4_cumule'] = references_cumulees.get('J-4')
            st.session_state['ref_j1_brute'] = references_jour.get('J-1')
            st.session_state['ref_j2_brute'] = references_jour.get('J-2')
            st.session_state['ref_j3_brute'] = references_jour.get('J-3')
            st.session_state['ref_j4_brute'] = references_jour.get('J-4')
        return
    
    # 1. Préparer le DataFrame de base
    df_statistiques = prepare_base_df(df_gps_brut)
    
    # 2. Calculer les 4 DataFrames de moyennes individuelles
    # On doit s'assurer que cols_num est disponible (via l'import de constantes)
    moyennes_indiv = calculer_moyennes_individuelles_par_jour(df_statistiques, cols_num)
    
    df_j1_mean_par_joueuse = moyennes_indiv.get('J-1', pd.DataFrame())
    df_j2_mean_par_joueuse = moyennes_indiv.get('J-2', pd.DataFrame())
    df_j3_mean_par_joueuse = moyennes_indiv.get('J-3', pd.DataFrame())
    df_j4_mean_par_joueuse = moyennes_indiv.get('J-4', pd.DataFrame())

    # 3. Définir les colonnes à cumuler
    cols_a_cumuler = [col for col in cols_cumul_sum if col in df_j4_mean_par_joueuse.columns]

    # 4. Calculer les cumuls
    
    # J-3 + J-4
    df_list_j3_j4 = [df_j4_mean_par_joueuse, df_j3_mean_par_joueuse]
    jours_labels_j3_j4 = ['J-4', 'J-3'] 
    df_charge_j3_j4_cumulee = cumuler_moyennes_individuelles(df_list_j3_j4, jours_labels_j3_j4, cols_a_cumuler)

    # J-2 + J-3 + J-4
    df_list_j2_j3_j4 = [df_j4_mean_par_joueuse, df_j3_mean_par_joueuse, df_j2_mean_par_joueuse]
    jours_labels_j2_j3_j4 = ['J-4', 'J-3', 'J-2']
    df_charge_j2_j3_j4_cumulee = cumuler_moyennes_individuelles(df_list_j2_j3_j4, jours_labels_j2_j3_j4, cols_a_cumuler)
    
    # J-1 + J-2 + J-3 + J-4
    df_list_j1_j2_j3_j4 = [df_j4_mean_par_joueuse, df_j3_mean_par_joueuse, df_j2_mean_par_joueuse, df_j1_mean_par_joueuse]
    jours_labels_j1_j2_j3_j4 = ['J-4', 'J-3', 'J-2', 'J-1']
    df_charge_j1_j2_j3_j4_cumulee = cumuler_moyennes_individuelles(df_list_j1_j2_j3_j4, jours_labels_j1_j2_j3_j4, cols_a_cumuler)
    
    # 5. Stockage final (le dictionnaire unique + les 4 clés pour la page Stat)
    DF_REFERENCES_JOUR = {
        'J-1': df_j1_mean_par_joueuse,
        'J-2': df_j2_mean_par_joueuse,
        'J-3': df_j3_mean_par_joueuse,
        'J-4': df_j4_mean_par_joueuse
    }
    
    DF_REFERENCES_MOYENNES = {
        'J-1': df_charge_j1_j2_j3_j4_cumulee,
        'J-2': df_charge_j2_j3_j4_cumulee,
        'J-3': df_charge_j3_j4_cumulee,
        'J-4': df_j4_mean_par_joueuse
    }
    
    st.session_state['DF_REFERENCES_CUMULEES'] = DF_REFERENCES_MOYENNES 
    st.session_state['DF_REFERENCES_JOUR'] = DF_REFERENCES_JOUR
    st.session_state['ref_j1_cumule'] = df_charge_j1_j2_j3_j4_cumulee
    st.session_state['ref_j2_cumule'] = df_charge_j2_j3_j4_cumulee
    st.session_state['ref_j3_cumule'] = df_charge_j3_j4_cumulee
    st.session_state['ref_j4_cumule'] = df_j4_mean_par_joueuse
    st.session_state['ref_j1_brute'] = df_j1_mean_par_joueuse
    st.session_state['ref_j2_brute'] = df_j2_mean_par_joueuse
    st.session_state['ref_j3_brute'] = df_j3_mean_par_joueuse
    st.session_state['ref_j4_brute'] = df_j4_mean_par_joueuse

    # Le DataFrame de base peut aussi être utile à la page Statistique pour son affichage
    st.session_state['df_statistiques_base'] = df_statistiques


def calculer_prescription_ajustement(df_ecart_percent: pd.DataFrame | pd.Series, jour_a_ajuster: str, seuil_optimal: float = 0.10) -> pd.DataFrame:
    """
    Calcule l'ajustement en pourcentage pour une séance future en inversant l'écart actuel.
    Cette version utilise .index pour la robustesse (Series vs DataFrame).
    """
    
    if df_ecart_percent.empty:
        return pd.DataFrame()

    # --- Étape 1 : Obtenir les valeurs et les noms de métriques ---
    
    # Identifier les valeurs d'écart (première ligne si c'est un DF, ou directement les valeurs si c'est une Series)
    if isinstance(df_ecart_percent, pd.DataFrame):
        ecart_values = df_ecart_percent.iloc[0]
        # Si c'est un DF, les noms des métriques sont dans .columns
        metric_names = df_ecart_percent.columns
    else: # Supposons que c'est pd.Series
        ecart_values = df_ecart_percent
        # Si c'est une Series, les noms des métriques sont dans .index
        metric_names = df_ecart_percent.index
        
    # 🔑 CLÉ DE LA CORRECTION : S'assurer que 'ecart_series' est une Series Pandas.
    # Ceci garantit que toutes les opérations vectorielles suivantes fonctionneront.
    ecart_series = pd.Series(ecart_values, index=metric_names)

    # 2. Inverser l'écart
    ajustement = -ecart_series 
    
    # 3. Application du Seuil Optimal
    ajustement[ajustement.abs() <= (seuil_optimal * 100)] = 0
    
    # 4. Création du DataFrame de sortie
    df_ajustement = pd.DataFrame(ajustement).T
    df_ajustement.index = [f'Ajustement pour {jour_a_ajuster} (%)']
    
    return df_ajustement.round(1)

def calculer_prescription_groupe(df_charge_actuelle_cumulee, df_moyenne_cumulee, df_moyenne_future_brute, jour_reference):
    """
    Calcule la prescription brute corrigée pour toutes les joueuses du groupe.
    
    Args:
        df_charge_actuelle_cumulee (DataFrame): Charge cumulée actuelle de toutes les joueuses (jusqu'à J-X).
        df_moyenne_cumulee (DataFrame): Moyenne cumulée de référence pour toutes les joueuses (jusqu'à J-X).
        df_moyenne_future_brute (DataFrame): Moyenne BRUTE de la prochaine séance pour toutes les joueuses (J-X+1).
        jour_reference (str): Le jour de la séance évaluée ('J-4', 'J-3', etc.).
        
    Returns:
        DataFrame: Un tableau consolidé des prescriptions corrigées pour tout le groupe.
    """
    
    resultats_prescriptions = []
    
    # 1. Identifier toutes les joueuses
    joueuses_uniques = df_charge_actuelle_cumulee['Name'].unique()
    
    for nom_joueuse in joueuses_uniques:
        
        # A. Filtrer les 3 DataFrames par joueuse
        actuel_cumule_indiv = df_charge_actuelle_cumulee[df_charge_actuelle_cumulee['Name'] == nom_joueuse].iloc[0]
        moyenne_cumule_indiv = df_moyenne_cumulee[df_moyenne_cumulee['Name'] == nom_joueuse].iloc[0]
        moyenne_future_indiv = df_moyenne_future_brute[df_moyenne_future_brute['Name'] == nom_joueuse]
        
                # B. Calculer l'Écart Brut (df_ecart) pour la joueuse
        
        # 1. Identifier les colonnes métriques (celles qui sont numériques)
        cols_metriques = [col for col in actuel_cumule_indiv.index if col not in ['Name', 'Type de Charge']]
        
        # 2. Extraire les séries de valeurs
        actuel_values = actuel_cumule_indiv[cols_metriques]
        ref_values = moyenne_cumule_indiv[cols_metriques]
        
        # 3. Calcul de l'écart brut (la série de valeurs)
        ecart_values = actuel_values.subtract(ref_values, fill_value=0)
        
        # 4. Reformatage en DataFrame à une ligne pour l'utiliser dans la fonction appliquer_soustraction_brute
        df_ecart_brut = ecart_values.to_frame().T
        # IMPORTANT : Supprimer l'index par défaut qui est le nom de la colonne (ex: 'Total Time')
        df_ecart_brut.index = [0] 
        df_ecart_brut.insert(0, 'Type de Charge', 'Écart (Valeur)')
        
        # C. Appliquer la Soustraction Corrigée (Plafonnement)
        if not moyenne_future_indiv.empty:
            df_charge_corrigee = appliquer_soustraction_brute(
                df_ecart_brut=df_ecart_brut,
                df_moyenne_seance_future=moyenne_future_indiv
            )
            
            # D. Stocker le résultat
            resultats_prescriptions.append(df_charge_corrigee)

    # 2. Consolidation finale
    if resultats_prescriptions:
        df_prescriptions_groupe = pd.concat(resultats_prescriptions, ignore_index=True)
        return df_prescriptions_groupe
    else:
        return pd.DataFrame()

def formuler_recommendation_future(df_ajustement: pd.DataFrame, jour_a_ajuster: str) -> str:
    """Formule une phrase simple basée sur le Total Time pour la séance suivante."""
    if df_ajustement.empty or 'Total Time' not in df_ajustement.columns:
        return "Données insuffisantes pour formuler une recommandation."
        
    ajustement_tt = df_ajustement.iloc[0]['Total Time']
    
    if ajustement_tt == 0:
        return f"La charge actuelle est optimale (±10%). Maintenez la charge prévue pour **{jour_a_ajuster}**."
    elif ajustement_tt > 0:
        return f"Augmenter le **Volume Global (Total Time)** de la séance **{jour_a_ajuster}** d'environ **+{ajustement_tt:.1f}%** pour compenser la sous-charge au jour actuel."
    else:
        return f"Réduire le **Volume Global (Total Time)** de la séance **{jour_a_ajuster}** d'environ **{ajustement_tt:.1f}%** pour compenser la surcharge au jour actuel."
    

def appliquer_soustraction_brute(df_ecart_brut, df_moyenne_seance_future):
    """
    Applique la compensation d'écart brut (cumulé) à la séance future (brute) avec plafonnement.
    Conserve le nom de la joueuse et retourne un DataFrame formaté dans le bon ordre.
    """
    
    # Colonnes d'identification et non-métriques que nous savons exister dans certains DF
    COLS_IDENTIFICATION = ['Name', 'Type de Charge', 'Type de Prescription', 'Base Évaluation', 'Jour Prescription']
    COLS_A_SUPPRIMER_FINAL = ['Sprint (m)', 'V max', 'Meterage Per Minute', 'Cardio', 'Muscu']
    
    # 1. Préparation des variables et détermination des colonnes de calcul
    
    # Récupérer le nom de la joueuse (doit être fait avant de supprimer la colonne 'Name')
    # Ceci est la seule utilisation de la colonne 'Name' dans cette fonction
    nom_joueuse = df_moyenne_seance_future['Name'].iloc[0]
    
    # Déterminer la liste des colonnes NUMÉRIQUES à conserver et à calculer
    # On prend toutes les colonnes présentes dans le DF de moyenne future qui ne sont PAS des identifiants
    cols_numeriques = [
        col for col in df_moyenne_seance_future.columns 
        if col not in COLS_IDENTIFICATION and col not in COLS_A_SUPPRIMER_FINAL
    ]
    
    # 2. Isolez les valeurs numériques des deux DataFrames AVANT la soustraction
    
    # On filtre les DataFrames sur les colonnes NUMÉRIQUES et on récupère la ligne (Series)
    ecart_values_filtre = df_ecart_brut[cols_numeriques].iloc[0]
    moyenne_values_filtre = df_moyenne_seance_future[cols_numeriques].iloc[0]
    
    # 3. Soustraction et Plafonnement (Clipping)
    
    # 🚨 LA SOUSTRACTION EST MAINTENANT SÛRE 🚨
    # On soustrait uniquement les colonnes métriques filtrées (toutes sont numériques)
    resultat_soustrait = moyenne_values_filtre.subtract(ecart_values_filtre, fill_value=0)
    resultat_plafonne = np.maximum(0, resultat_soustrait) 
    
    # 4. Reconstruire le DataFrame de résultat
    
    # 4a. Conversion de la Series de résultat en DataFrame d'une seule ligne
    df_charge_corrigee = resultat_plafonne.to_frame().T
    
    # 4b. Insérer les métadonnées
    df_charge_corrigee.insert(0, 'Name', nom_joueuse)
    df_charge_corrigee.insert(1, 'Type de Prescription', 'Charge Corrigée')
    
    # 4c. Appliquer l'ordre des colonnes (basé sur les colonnes numériques originales)
    cols_ordre_numerique = [col for col in df_moyenne_seance_future.columns if col not in COLS_IDENTIFICATION]
    cols_ordre_final = ['Name', 'Type de Prescription'] + cols_ordre_numerique
    
    # On utilise reindex pour s'assurer que toutes les colonnes sont présentes dans le bon ordre
    df_charge_corrigee = df_charge_corrigee.reindex(columns=cols_ordre_final, fill_value=0)

    # 4d. Nettoyage final (supprimer la colonne descriptive)
    df_charge_corrigee = df_charge_corrigee.drop('Type de Prescription', axis=1)
    df_charge_corrigee = df_charge_corrigee.reset_index(drop=True)
    
    # 5. Retourner le résultat
    return df_charge_corrigee.round(2)

# 🎯 DÉTECTION AUTOMATIQUE DU JOUR DE RÉFÉRENCE
def detecter_dernier_jour_realise(session_state):
    """
    Détecte le dernier jour (J-X) pour lequel une charge 'actuelle' existe, 
    en utilisant les clés de session existantes pour les DataFrames cumulés de référence (qui servent d'actuel).
    """
    # Mappage des jours d'évaluation aux clés de session correspondantes (votre structure actuelle)
    jours_map = {
        'J-4': 'ref_j4_cumule',  
        'J-3': 'ref_j3_cumule',
        'J-2': 'ref_j2_cumule',
        'J-1': 'ref_j1_cumule'
    }
    
    # On parcourt les jours du plus récent au plus ancien
    jours_possibles = ['J-1', 'J-2', 'J-3', 'J-4'] # Note : L'ordre est inversé pour aller du plus récent au plus ancien

    for jour in jours_possibles:
        # On utilise le mapping pour obtenir le nom exact de la clé
        key_cumul = jours_map.get(jour) 
        
        # On vérifie si la clé existe ET que le DataFrame n'est pas vide
        if key_cumul in session_state and not session_state.get(key_cumul, pd.DataFrame()).empty:
            # Si on trouve des données non vides, ce jour devient notre jour de référence
            return jour 
            
    return None

def calculer_high_speed_distance_custom(df):
    """
    Calcule la métrique 'High Speed Distance' comme la somme de 'SPR' et 'SPR+' sur un DataFrame donné.
    Si les colonnes source sont manquantes, 'High Speed Distance' est défini à 0.
    """
    df = df.copy() # Travailler sur une copie
    
    # Vérifie si les colonnes nécessaires existent
    has_spr = 'SPR' in df.columns
    has_spr_plus = 'SPR+' in df.columns
    
    # Le calcul ne se fait que si les colonnes source existent
    if has_spr and has_spr_plus:
        df['High Speed Distance'] = df['SPR'] + df['SPR+']
    else:
        # Si une colonne manque, on s'assure que HSD est à 0 s'il est manquant
        if 'High Speed Distance' not in df.columns:
            df['High Speed Distance'] = 0
            
    # On supprime 'SPR' et 'SPR+' de la comparaison s'ils ne font pas partie de cols_cumul_sum initiales
    # (cela doit être géré via cols_cumul_sum pour être sûr)
            
    return df

def calculer_prescription_joueuse(nom_joueuse, df_rpe_complet, DF_REFERENCES_MOYENNES, cols_cumul_sum, jour_cible_analyse=None, date_analyse_cible_dt=None, df_id_joueuses=None):
    """
    Calcule la prescription. 
    - HSD est calculée comme SPR + SPR+.
    - Le nom final est formaté en NOM, Prénom (Format Catapult ID).
    """
    
    # 1. FILTRE DU DATAFRAME GLOBAL PAR JOUEUSE
    df_player_data = df_rpe_complet[df_rpe_complet['Name'] == nom_joueuse].copy()
    if df_player_data.empty or df_player_data.shape[0] < 1:
        return pd.DataFrame(), None, None
        
    df_player_data = df_player_data.sort_values(by='Date', ascending=False).reset_index(drop=True)
    df_player_data['Jour semaine'] = df_player_data['Jour semaine'].str.strip().str.upper()

    # ----------------------------------------------------------------------
    # 🚨 STEP A: RECALCUL DE 'High Speed Distance' DANS LE DATAFRAME DES JOUEUSES (BRUTES) 🚨
    # ----------------------------------------------------------------------
    df_player_data = calculer_high_speed_distance_custom(df_player_data)
        
    # 2. Définition du Jour de Référence pour la Logique
    if not jour_cible_analyse or jour_cible_analyse.strip().upper() not in ['J-4', 'J-3', 'J-2', 'J-1', 'J']:
        st.error(f"Erreur: Jour cible d'analyse '{jour_cible_analyse}' non valide.")
        return pd.DataFrame(), None, None
        
    jour_reference = jour_cible_analyse.strip().upper()
    
    # ----------------------------------------------------------------------
    # 🚨 STEP B: DÉFINITION DE COLS_A_COMPARER (on utilise cols_cumul_sum tel quel) 🚨
    # ----------------------------------------------------------------------
    cols_a_comparer = [col for col in cols_cumul_sum if col in df_player_data.columns]


    # ----------------------------------------------------------------------
    # 🎯 LOGIQUE CONDITIONNELLE : MOYENNE J-4 pour J-1/J 🎯
    # ----------------------------------------------------------------------
    
    if jour_reference in ['J-1', 'J']: 
        
        jour_a_ajuster = 'J-4' 
        key_future = 'ref_j4_brute' 
        
        if key_future not in st.session_state:
            st.warning(f"❌ Échec: Clé de moyenne J-4 '{key_future}' manquante dans session_state.")
            return pd.DataFrame(), None, None
        
        df_moyenne_seance_future = st.session_state.get(key_future)
        df_charge_corrigee = df_moyenne_seance_future[df_moyenne_seance_future['Name'] == nom_joueuse].copy()
        
        if df_charge_corrigee.empty:
            st.warning(f"❌ Échec pour {nom_joueuse}: Joueuse non trouvée dans la moyenne BRUTE {jour_a_ajuster}.")
            return pd.DataFrame(), None, None
            
        # 🚨 CORRECTIF 1: Recalculer HSD dans la référence J-4 Brute 🚨
        df_charge_corrigee = calculer_high_speed_distance_custom(df_charge_corrigee)
             
        # Colonnes de débogage
        df_charge_corrigee.insert(1, 'Type de Prescription', f'Moyenne BRUTE J-4') 
        df_charge_corrigee.insert(2, 'Base Évaluation', jour_reference) 
        df_charge_corrigee.insert(3, 'Jour Prescription', jour_a_ajuster)
        
    # ----------------------------------------------------------------------
    # LOGIQUE DE PRESCRIPTION AJUSTÉE (J-4, J-3, J-2)
    # ----------------------------------------------------------------------
    else:
        # 3. Filtrer la dernière séance de TRAVAIL (J-X) pour le calcul de l'Actuel
        df_travail_recent = df_player_data[df_player_data['Jour semaine'].str.startswith('J-')].copy()

        if df_travail_recent.empty:
            st.info(f"Débogage: {nom_joueuse} - Aucune séance de travail (J-X) récente trouvée.")
            return pd.DataFrame(), None, None
        
        # 4. VÉRIFICATION D'EXCLUSION
        if date_analyse_cible_dt is None:
             st.error(f"Erreur interne: date_analyse_cible_dt non fournie pour {nom_joueuse}.")
             return pd.DataFrame(), None, None

        df_seance_cible = df_player_data[
            (df_player_data['Jour semaine'] == jour_reference) &
            (df_player_data['Date'] == date_analyse_cible_dt) 
        ]
        
        if df_seance_cible.empty:
            st.warning(f"❌ Exclusion : **{nom_joueuse}** n'a pas participé à la séance cible (**{jour_reference}**) à la date du **{date_analyse_cible_dt.strftime('%Y-%m-%d')}**. Prescription ignorée.")
            return pd.DataFrame(), None, None
            
        # ... (Logique de cumul df_charge_actuelle) ...
        date_seance_cible = df_seance_cible['Date'].iloc[0]
        df_base_cumul = df_travail_recent[df_travail_recent['Date'] <= date_seance_cible].reset_index(drop=True)
        seances_requises = {'J-4': 1, 'J-3': 2, 'J-2': 3}.get(jour_reference, 1)
        
        cols_cumul_actuel = list(range(seances_requises))
        is_cumulative = jour_reference in ['J-3', 'J-2'] and df_base_cumul.shape[0] >= seances_requises
        
        if is_cumulative:
            df_cycle_actuel = df_base_cumul.iloc[cols_cumul_actuel][cols_a_comparer].copy()
            df_charge_actuelle = df_cycle_actuel.sum().to_frame().T
        else:
            df_charge_actuelle = df_base_cumul.iloc[[0]][cols_a_comparer].copy() 
            
        df_charge_actuelle['Name'] = nom_joueuse
        df_charge_actuelle.insert(1, 'Type de Charge', 'Actuelle')
        
        # 6. SÉLECTION DE LA RÉFÉRENCE HISTORIQUE ET CALCUL DE L'ÉCART
        if jour_reference not in DF_REFERENCES_MOYENNES:
             st.warning(f"❌ Échec pour {nom_joueuse}: Jour de référence '{jour_reference}' non trouvé dans DF_REFERENCES_MOYENNES.")
             return pd.DataFrame(), None, None

        df_reference_joueuse_globale = DF_REFERENCES_MOYENNES[jour_reference]
        ref_player = df_reference_joueuse_globale[df_reference_joueuse_globale['Name'] == nom_joueuse].copy()
                                 
        if ref_player.empty:
             st.warning(f"❌ Échec pour {nom_joueuse}: Joueuse non trouvée dans la référence CUMULÉE {jour_reference}.")
             return pd.DataFrame(), None, None

        # 🚨 CORRECTIF 2: Recalculer HSD dans la Référence CUMULÉE (ref_player) 🚨
        ref_player = calculer_high_speed_distance_custom(ref_player)
            
        ref_values = ref_player[cols_a_comparer].iloc[0] 
        actuel_values = df_charge_actuelle[cols_a_comparer].iloc[0]
        df_ecart_brut = (actuel_values - ref_values).to_frame().T
        df_ecart_brut.insert(0, 'Type de Charge', 'Écart (Valeur)') 
        df_ecart_brut['Name'] = nom_joueuse
        
        # Déterminer le jour à ajuster (J+1)
        jour_a_ajuster = {'J-4': 'J-3', 'J-3': 'J-2', 'J-2': 'J-1'}.get(jour_reference) 
        
        # 7. CALCUL DE LA PRESCRIPTION (df_charge_corrigee)
        jour_a_ajuster_key = jour_a_ajuster.split("(")[0].lower().replace("-", "")
        key_future = f'ref_{jour_a_ajuster_key}_brute'

        if key_future not in st.session_state:
             st.warning(f"❌ Échec: Clé de moyenne future '{key_future}' manquante dans session_state.")
             return pd.DataFrame(), None, None
             
        df_moyenne_seance_future = st.session_state.get(key_future)
        df_moyenne_joueuse = df_moyenne_seance_future[df_moyenne_seance_future['Name'] == nom_joueuse].copy() 
        if df_moyenne_joueuse.empty:
             st.warning(f"❌ Échec pour {nom_joueuse}: Joueuse non trouvée dans la moyenne BRUTE future {key_future}.")
             return pd.DataFrame(), None, None

        ecart_values_filtre = df_ecart_brut[['Name'] + cols_a_comparer].copy()
        
        # 🚨 CORRECTIF 3: Recalculer HSD dans la Référence Future Brute 🚨
        df_moyenne_joueuse = calculer_high_speed_distance_custom(df_moyenne_joueuse)
            
        moyenne_values_filtre = df_moyenne_joueuse[['Name'] + cols_a_comparer].copy()

        # Appel à la fonction de soustraction (on suppose appliquer_soustraction_brute existe)
        df_charge_corrigee = appliquer_soustraction_brute(
            df_ecart_brut=ecart_values_filtre,
            df_moyenne_seance_future=moyenne_values_filtre
        )
        
        # Colonnes de débogage
        df_charge_corrigee.insert(2, 'Base Évaluation', jour_reference) 
        df_charge_corrigee.insert(3, 'Jour Prescription', jour_a_ajuster)
    
    

    # ----------------------------------------------------------------------
    # 🚨 FORMATAGE FINAL CATAPULT (Inclut le formatage du nom NOM, Prénom) 🚨
    # ----------------------------------------------------------------------
    
    # 1. Extraction des métadonnées Catapult (ID, Position)
    if df_id_joueuses is None: 
        st.error("Erreur: Le DataFrame d'identification Catapult est manquant.")
        return pd.DataFrame(), None, None
        
    # Préparation du nom d'entrée pour la recherche normalisée (PRÉNOM NOM en MAJ)
    # 🚨 Ceci garantit que votre 'Madelynn ANDERSON' devient 'MADELYNN ANDERSON' 🚨
    nom_joueuse_upper = nom_joueuse.strip().upper()

    # Recherche dans la colonne normalisée
    meta_player = df_id_joueuses[df_id_joueuses['Normalized Name'] == nom_joueuse_upper]

    if meta_player.empty: 
        st.warning(f"❌ Formatage: Joueuse **{nom_joueuse}** non trouvée (Nom normalisé: **{nom_joueuse_upper}**) dans le fichier ID Catapult.")
        return pd.DataFrame(), None, None
        
    meta_player = meta_player.iloc[0]
    athlete_id = meta_player['Athlete ID']
    # Récupération du nom dans le format Catapult ORIGINAL (NOM, Prénom)
    athlete_name_catapult = meta_player['Athlete Name'] 
    position = meta_player['Position'] 
    
    # 2. Nettoyage
    
    # Supprimer la colonne de nom temporaire ('Name') qui est au mauvais format
    if 'Name' in df_charge_corrigee.columns:
        df_charge_corrigee = df_charge_corrigee.drop(columns=['Name'], errors='ignore')
    
    # Suppression des colonnes temporaires et de débogage
    cols_to_drop = [
        'Type de Prescription', 'Base Évaluation', 'Jour Prescription', 'Type de Charge'
    ]
    df_charge_corrigee = df_charge_corrigee.drop(
        columns=[col for col in cols_to_drop if col in df_charge_corrigee.columns],
        errors='ignore'
    )
    
    # 3. Insertion des colonnes d'identification dans l'ORDRE STRCIT Catapult
    
    # a. Athlete ID (1er)
    df_charge_corrigee.insert(0, 'Athlete ID', athlete_id) 
    
    # b. Athlete Name (2ème) - Utilisation du nom formaté Catapult (NOM, Prénom)
    df_charge_corrigee.insert(
        1, 
        'Athlete Name', 
        athlete_name_catapult 
    )
    
    # c. Position (3ème)
    df_charge_corrigee.insert(
        2, 
        'Position', 
        position 
    )

    # 4. S'assurer que les métriques sont dans le bon ordre
    cols_id = ['Athlete ID', 'Athlete Name', 'Position']
    cols_final_order = cols_id + [col for col in cols_a_comparer if col in df_charge_corrigee.columns]

    df_final_csv = df_charge_corrigee.reindex(columns=cols_final_order)

    df_final_csv['Very High Speed Distance'] = df_final_csv['SPR Total Distance (m)'] + df_final_csv['SPR + Total Distance (m)']
    df_final_csv = df_final_csv.drop(['SPR Total Distance (m)', 'SPR + Total Distance (m)', 's.Cardio', 's.Muscu'], axis=1)

    dict_renomme = {
        'Total Distance (m)': 'Total Distance',
        'HSR Total distance (m)': 'Velocity Distance Band 4',
        'VHSR Total Distance (m)': 'Velocity Distance Band 5',
        'VHSR effort': 'Velocity Effort Band 5',
        'Sprint effort': 'High Speed Effort',
        'Accel >2m.s²': 'Acceleration Bands 2-3 Efforts', 
        'Decel >2m.s²': 'Deceleration Bands 2-3 Efforts',
    }

    df_final_csv.rename(columns=dict_renomme, inplace=True, errors='ignore')
    
    return df_final_csv, jour_reference, jour_a_ajuster


def calculer_prescription_groupe_auto(df_rpe_complet, DF_REFERENCES_MOYENNES, cols_cumul_sum, date_analyse_cible, df_id_joueuses): # 👈 AJOUTÉ
    """
    Itère sur toutes les joueuses de champ pour générer un DataFrame de prescription de groupe 
    basé sur une date cible et la logique conditionnelle.
    """
    
    # 1. Obtenir la liste des joueuses de champ à analyser (Exclusion des Gardiennes)
    joueuses_uniques = df_rpe_complet['Name'].unique()
    df_positions = df_rpe_complet[['Name', 'Position Name']].drop_duplicates()
    GARDENNE_POSTE = 'GARDIENNE' 
    
    noms_a_exclure = df_positions[
        df_positions['Position Name'].str.upper().str.strip() == GARDENNE_POSTE
    ]['Name'].unique()
    
    joueuses_a_analyser = [
        nom for nom in joueuses_uniques 
        if nom not in noms_a_exclure
    ]

    # ----------------------------------------------------------------------
    # 🎯 ÉTAPE CLÉ : DÉTERMINER LE JOUR DE RÉFÉRENCE CYCLIQUE À PARTIR DE LA DATE CIBLE 🎯
    # ----------------------------------------------------------------------
    
    df_rpe_complet['Jour semaine'] = df_rpe_complet['Jour semaine'].str.strip().str.upper()

    df_cible_jour = df_rpe_complet[
        (df_rpe_complet['Date'] == date_analyse_cible) & 
        (df_rpe_complet['Jour semaine'].str.startswith('J-') | df_rpe_complet['Jour semaine'].str.startswith('J'))
    ]
    
    if df_cible_jour.empty:
        st.warning(f"Aucune séance de travail (J-X ou J) trouvée à la date cible {date_analyse_cible.strftime('%Y-%m-%d')}.")
        return pd.DataFrame(), None, None
        
    jour_reference_cible = df_cible_jour['Jour semaine'].mode().iloc[0].strip()
    st.success(f"Jour de référence du microcycle détecté à la date cible : **{jour_reference_cible}**")

    # ----------------------------------------------------------------------

    resultats_prescriptions = []
    jour_ref_final = jour_reference_cible
    jour_ajust_final = None 
    
    # 2. Itérer sur les joueuses de champ
    for nom_joueuse in joueuses_a_analyser:
        
        df_presc, jour_ref, jour_ajust = calculer_prescription_joueuse(
            nom_joueuse, 
            df_rpe_complet, 
            DF_REFERENCES_MOYENNES, 
            cols_cumul_sum,
            jour_cible_analyse=jour_reference_cible,
            date_analyse_cible_dt=date_analyse_cible,
            df_id_joueuses=df_id_joueuses # 👈 TRANSMISSION DU DATAFRAME ID
        )
        
        if not df_presc.empty:
            resultats_prescriptions.append(df_presc)
            if jour_ajust_final is None:
                jour_ajust_final = jour_ajust
            
    # 3. Concaténer le résultat final
    if resultats_prescriptions:
        df_final = pd.concat(resultats_prescriptions, ignore_index=True)
        return df_final, jour_ref_final, jour_ajust_final
    else:
        return pd.DataFrame(), None, None