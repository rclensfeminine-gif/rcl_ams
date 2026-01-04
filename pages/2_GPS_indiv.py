import streamlit as st
import pandas as pd
import os
import numpy as np
from sections.menu.menu import custom_sidebar_menu
from sections.gps.norme import calculer_prescription_ajustement, appliquer_soustraction_brute
from sections.visualisation.viz import creer_graph_dt, creer_graph_vhsr, creer_graph_spr, creer_graph_accel_charge
from sections.gps.pipeline import filtrer_dataframe_joueuse_indiv, choisir_une_joueuse_une_date, regrouper_par_semaine_civile, add_columns_session_rpe, generer_alertes_charge
from sections.constantes import ONGLET_GPS_SAISON, ONGLET_GPS_TYPE, LIST_ORGANISATION_COLS_INDIV, cols_cumul_sum

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

# initialisation 
if 'all_gps_session' not in st.session_state:
    st.error("Veuillez d'abord charger les données GPS.")
    st.stop()
df_gps = st.session_state['all_gps_session']

if 'ref_j1_cumule' not in st.session_state or 'ref_j1_brute' not in st.session_state:
    st.error("Les DataFrames de référence (J-1 à J-4) n'ont pas été calculés ou chargés. Veuillez visiter la page 4_GPS_statistiques.")
    st.stop()

DF_REFERENCES_MOYENNES = {
    'J-1': st.session_state['ref_j1_cumule'],
    'J-2': st.session_state['ref_j2_cumule'],
    'J-3': st.session_state['ref_j3_cumule'],
    'J-4': st.session_state['ref_j4_cumule'],
}

if 'DF_REFERENCES_MOYENNES' not in st.session_state:
    st.session_state['DF_REFERENCES_MOYENNES'] = DF_REFERENCES_MOYENNES

DF_REFERENCES_JOUR = {
    'J-1': st.session_state['ref_j1_brute'],
    'J-2': st.session_state['ref_j2_brute'],
    'J-3': st.session_state['ref_j3_brute'],
    'J-4': st.session_state['ref_j4_brute'],
}
if 'DF_REFERENCES_JOUR' not in st.session_state:
    st.session_state['DF_REFERENCES_JOUR'] = DF_REFERENCES_JOUR

# Main
st.title("Données GPS - Joueuses")

### Par jour d'entraînement (Bloc RPE) ###

FILE_PATH = "data/rpe.csv"
if not os.path.exists("data"):
    os.makedirs("data")

@st.cache_data
def load_and_merge_rpe(df_base):
    try:
        df_rpe = pd.read_csv(FILE_PATH)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        df_rpe = pd.DataFrame(columns=['Name', 'Date', 'Jour semaine', 'Cardio', 'Muscu', 'Presence'])
    
    # Conversion de la colonne 'Date' dans les deux DataFrames pour éviter l'erreur de fusion
    df_base['Date'] = pd.to_datetime(df_base['Date'], dayfirst=True)
    df_rpe['Date'] = pd.to_datetime(df_rpe['Date'], dayfirst=True, errors='coerce')
    
    df_merged = pd.merge(df_base, df_rpe, on=['Name', 'Date', 'Jour semaine'], how='left')
    df_merged['Presence'] = df_merged['Presence'].fillna('C')
    return df_merged

# Initialisation du DataFrame complet si ce n'est pas déjà fait
if 'df_joueuse_rpe_complete' not in st.session_state:
    st.session_state['df_joueuse_rpe_complete'] = load_and_merge_rpe(df_gps.copy())

selection_annee = st.segmented_control("Saison", ONGLET_GPS_SAISON, selection_mode="single", default=ONGLET_GPS_SAISON[0])
selection_type = st.segmented_control("Type séance", ONGLET_GPS_TYPE, selection_mode="multi")
filtered_df_rpe = filtrer_dataframe_joueuse_indiv(st.session_state['df_joueuse_rpe_complete'], selection_annee, selection_type)

# Gestion de la joueuse sélectionnée
available_players = filtered_df_rpe['Name'].unique().tolist()
first_player = available_players[0] if available_players else None

if 'selected_player' not in st.session_state or st.session_state['selected_player'] not in available_players:
    st.session_state['selected_player'] = first_player

# Bloc RPE
if not filtered_df_rpe.empty:
    filtered_df_rpe_display = filtered_df_rpe[['Name', 'Date', 'Jour semaine', 'Cardio', 'Muscu', 'Presence']]
    st.write('Inscrire les RPE :')
    edited_df = st.data_editor(filtered_df_rpe_display, key="rpe_editor", hide_index=True)

    if st.button("Sauvegarder les notes"):
        try:
            existing_notes = pd.read_csv(FILE_PATH)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            existing_notes = pd.DataFrame(columns=['Name', 'Date', 'Jour semaine', 'Cardio', 'Muscu', 'Presence'])
        
        existing_notes['Date'] = pd.to_datetime(existing_notes['Date'], dayfirst=True, errors='coerce')
        edited_notes = edited_df[['Name', 'Date', 'Jour semaine', 'Cardio', 'Muscu', 'Presence']]
        edited_notes['Date'] = pd.to_datetime(edited_notes['Date'], dayfirst=True, errors='coerce')

        updated_notes_df = pd.concat([existing_notes, edited_notes]).drop_duplicates(
            subset=['Name', 'Date', 'Jour semaine'], keep='last'
        )
        
        updated_notes_df['Date'] = updated_notes_df['Date'].dt.strftime('%d-%m-%Y')
        updated_notes_df.to_csv(FILE_PATH, index=False)
        
        st.success("Notes sauvegardées avec succès !")
        
        df_gps['Date'] = pd.to_datetime(df_gps['Date'], dayfirst=True)
        updated_notes_df['Date'] = pd.to_datetime(updated_notes_df['Date'], dayfirst=True)
        
        st.session_state['df_joueuse_rpe_complete'] = pd.merge(df_gps, updated_notes_df, on=['Name', 'Date', 'Jour semaine'], how='left')
        
        st.session_state['filtered_df_rpe'] = filtrer_dataframe_joueuse_indiv(st.session_state['df_joueuse_rpe_complete'], selection_annee, selection_type)
        st.rerun()

# --- Nouveau tableau avec toutes les données ---
st.markdown("---")
st.subheader("Données GPS individuelles")

## Par jour par joueuse
if not filtered_df_rpe.empty:
    filtered_df_rpe = add_columns_session_rpe(df=filtered_df_rpe)

    # 1. Filtre la joueuse et la date (produit df_joueuse_indiv)
    df_joueuse_indiv = choisir_une_joueuse_une_date(df=filtered_df_rpe, key_prefix='daily_view', default_player=st.session_state['selected_player'])

    # 2. Si des données existent, on affiche les données du jour
    if not df_joueuse_indiv.empty:
        st.subheader("Données par jour")
        df_joueuse_indiv_display = df_joueuse_indiv[LIST_ORGANISATION_COLS_INDIV]
        st.dataframe(df_joueuse_indiv_display, hide_index=True)

        st.header("Analyse de Charge Instantanée")

        # 3. IDENTIFICATION ET TRI DES SÉANCES
        df_player_data = df_joueuse_indiv.copy()
        df_player_data = df_player_data.sort_values(by='Date', ascending=False).reset_index(drop=True)
        
        if df_player_data.shape[0] < 1:
            st.info("Moins de 1 séance enregistrée pour cette joueuse.")
            st.stop() # Arrête le script ici si aucune donnée n'est trouvée pour la joueuse/filtres
            
        derniere_seance_info = df_player_data.iloc[0]
        jour_semaine_actuel = derniere_seance_info['Jour semaine'].strip()
        activity_name_actuel = derniere_seance_info['Activity Name']
        
        # Préparation des variables pour la comparaison
        cols_a_comparer = [col for col in cols_cumul_sum if col in df_player_data.columns]
        cols_cumul_actuel = [] # Indices des séances à sommer
        alerte_sequence = "N/A"
        
        # --- LOGIQUE DE VÉRIFICATION DE LA SÉQUENCE ET DÉTERMINATION DU CUMUL ---
        
        if jour_semaine_actuel in ['J-1', 'J-2', 'J-3', 'J-4']:
            
            if jour_semaine_actuel == 'J-1':
                seances_requises = 4
                sequence_attendue = ['J-2', 'J-3', 'J-4']
            elif jour_semaine_actuel == 'J-2':
                seances_requises = 3
                sequence_attendue = ['J-3', 'J-4']
            elif jour_semaine_actuel == 'J-3':
                seances_requises = 2
                sequence_attendue = ['J-4']
            elif jour_semaine_actuel == 'J-4':
                seances_requises = 1
                sequence_attendue = []

            
            # Vérification
            if df_player_data.shape[0] < seances_requises:
                alerte_sequence = f"⚠ Séquence incomplète: Moins de {seances_requises} séances nécessaires pour le {jour_semaine_actuel}."
            else:
                sequence_actuelle = [df_player_data.iloc[i]['Jour semaine'].strip() for i in range(1, seances_requises)]
                
                if sequence_actuelle == sequence_attendue:
                    alerte_sequence = f"✅ Séquence {jour_semaine_actuel} confirmée."
                    cols_cumul_actuel = list(range(seances_requises))
                else:
                    trouve = ', '.join(sequence_actuelle)
                    attendu = ', '.join(sequence_attendue)
                    alerte_sequence = f"❌ Séquence Invalide {jour_semaine_actuel}: Attendu: {attendu} - Trouvé: {trouve}."
            
        else:
            seances_requises = 1
            cols_cumul_actuel = [0]
            alerte_sequence = f"ℹ️ Jour non critique ({jour_semaine_actuel}). Comparaison séance individuelle."

        
        # Affichage de l'état de la séquence
        st.markdown(f"**Séance évaluée :** `{activity_name_actuel}` ({jour_semaine_actuel})")
        st.info(alerte_sequence)

        # 4. CALCUL DE LA CHARGE ACTUELLE CUMULÉE (ou séance individuelle)
        
        # Logique de cumul si la séquence est valide ET que c'est un jour de microcycle
        is_cumulative = alerte_sequence.startswith("✅") and jour_semaine_actuel in ['J-1', 'J-2', 'J-3']
        
        if is_cumulative:
            df_cycle_actuel = df_player_data.iloc[cols_cumul_actuel].copy()
            df_charge_actuelle = df_cycle_actuel[cols_a_comparer].sum().to_frame().T
            df_charge_actuelle['Name'] = st.session_state['selected_player']
            
            st.markdown(f"**Charge évaluée :** Cumul des charges ({jour_semaine_actuel} + jours précédents)")
            jour_reference = jour_semaine_actuel
            reference_dict = DF_REFERENCES_MOYENNES
            
        else:
            # Séquence invalide, J-4 (non cumulatif par design), ou autre jour
            if not alerte_sequence.startswith("ℹ️"):
                 st.warning(f"Analyse de charge cumulée non réalisée (Séquence invalide ou Jour non cumulé). Comparaison {jour_semaine_actuel} vs Moyenne {jour_semaine_actuel} individuelle.")

            df_charge_actuelle = df_player_data.iloc[[0]].copy() # Séance individuelle
            jour_reference = jour_semaine_actuel
            reference_dict = DF_REFERENCES_MOYENNES # Utilise la référence individuelle J-X/J

        # 5. SÉLECTION DE LA RÉFÉRENCE HISTORIQUE
        if jour_reference in reference_dict:
            df_reference_joueuse_globale = reference_dict[jour_reference]
            ref_player = df_reference_joueuse_globale[df_reference_joueuse_globale['Name'] == st.session_state['selected_player']]
            
            if ref_player.empty:
                st.warning(f"Pas de référence historique ({jour_reference}) pour cette joueuse.")
            else:
                # 6. GÉNÉRATION DES ALERTES (MAIS NON AFFICHÉES)
                # La fonction generer_alertes_charge est toujours appelée pour générer le statut,
                # mais le résultat (df_alertes_player) n'est pas utilisé/affiché.
                generer_alertes_charge(
                    df_charge_actuelle, 
                    ref_player, 
                    cols_a_comparer, 
                    seuil_pourcentage=0.10
                )
                
                # ==========================================================
                # --- NOUVEAU BLOC : Affichage du comparatif Actuel vs Moyen (Format Large) ---
                # ==========================================================
                st.subheader(f"Charge cumulée analyse à ({jour_reference})")
                
                # 1. Préparation des valeurs numériques (sans 'Name')
                ref_values = ref_player[cols_a_comparer].iloc[0] # La référence moyenne
                actuel_values = df_charge_actuelle[cols_a_comparer].iloc[0] # La charge actuelle
                
                # 2. Calcul de l'écart en valeur absolue
                # Écart = Actuel - Moyenne
                df_ecart = (actuel_values - ref_values).to_frame().T
                df_ecart.insert(0, 'Type de Charge', 'Écart (Valeur)')

                # 3. Préparation des DataFrames Actuelle et Moyenne (mêmes que précédemment)
                df_actuel_display = actuel_values.to_frame().T
                df_actuel_display.insert(0, 'Type de Charge', 'Actuelle')

                df_moyen_display = ref_values.to_frame().T
                df_moyen_display.insert(0, 'Type de Charge', f'Moyenne {jour_reference}')

                # 4. Concaténation des trois lignes
                df_comparatif_final = pd.concat([df_actuel_display, df_moyen_display, df_ecart], ignore_index=True)
                
                # 5. Préparation des valeurs numériques (sans 'Name')
                ref_values = ref_player[cols_a_comparer].iloc[0] # La référence moyenne
                actuel_values = df_charge_actuelle[cols_a_comparer].iloc[0] # La charge actuelle
                
                # 5a. CALCUL DE L'ÉCART EN POURCENTAGE (POUR LA COULEUR)
                # (Actuel - Moyen) / Moyen * 100. Utilisation de .replace(0, np.nan) pour éviter la division par zéro.
                ecart_percent_values = ((actuel_values - ref_values) / ref_values.replace(0, np.nan) * 100)
                
                # 5b. CALCUL DE L'ÉCART EN VALEUR ABSOLUE (POUR L'AFFICHAGE)
                df_ecart = (actuel_values - ref_values).to_frame().T
                df_ecart.insert(0, 'Type de Charge', 'Écart (Valeur)')

                # =========================================================================
                # 🔑 AJOUT DU CALCUL DE L'AJUSTEMENT EN POURCENTAGE (df_ajustement_recommande)
                # =========================================================================

                # 1. Convertir la Série d'écart en Pourcentage en un DataFrame pour la fonction de calcul
                df_ecart_percent_pour_calc = ecart_percent_values.to_frame().T
                # L'index du DataFrame (qui représente l'écart en %) n'a pas d'importance ici.

                # 2. Calculer l'ajustement en Pourcentage
                # df_ajustement_recommande est maintenant défini.
                df_ajustement_recommande = calculer_prescription_ajustement(
                    df_ecart_percent=df_ecart_percent_pour_calc, 
                    jour_a_ajuster=jour_reference, # On utilise le jour de la séance évaluée pour le label
                    seuil_optimal=0.10
                )

                # 5c. Recréation du DataFrame final (seulement 3 lignes : Actuelle, Moyenne, Écart Valeur)
                df_actuel_display = actuel_values.to_frame().T
                df_actuel_display.insert(0, 'Type de Charge', 'Actuelle')
                
                df_moyen_display = ref_values.to_frame().T
                df_moyen_display.insert(0, 'Type de Charge', f'Moyenne {jour_reference}')
                                
                df_comparatif_final = pd.concat([df_actuel_display, df_moyen_display, df_ecart], ignore_index=True)


                # 6. Affichage avec formatage et coloration ciblée
                cols_affichage = ['Type de Charge'] + cols_a_comparer

                # Fonction de style qui applique les règles Vert/Orange/Rouge basées sur le pourcentage calculé
                def apply_color_on_row(row):
                    # Initialise une liste de styles vides pour toutes les colonnes
                    styles = [''] * len(cols_affichage)
                    
                    # On applique la couleur uniquement si la ligne est 'Écart (Valeur)'
                    if row['Type de Charge'] == 'Écart (Valeur)':
                        
                        # On itère sur les colonnes de métriques pour appliquer le style conditionnel
                        for i, col in enumerate(cols_a_comparer):
                            # On récupère le pourcentage (la clé de la couleur) pour la colonne actuelle
                            val_percent = pd.to_numeric(ecart_percent_values[col], errors='coerce')
                            
                            style_index = i + 1
                            
                            if pd.isna(val_percent):
                                styles[style_index] = '' 
                            
                            # --- LOGIQUE DE COULEUR PAR PLAGE ---
                            elif -2 <= val_percent <= 2:
                                # VERT : Optimal (Écart entre -2% et +2%)
                                styles[style_index] = 'color: green; font-weight: bold'
                            
                            elif (-5 < val_percent < -2) or (2 < val_percent < 5):
                                # ORANGE : Écart léger (Écart entre 2-5% ou -5--2%)
                                styles[style_index] = 'color: orange; font-weight: bold'
                                
                            else:
                                # ROUGE : Risque élevé (Écart au-delà de -5% ou +5%)
                                styles[style_index] = 'color: red; font-weight: bold'
                    
                    return styles

                # 6a. Application du style
                
                # 1. Utiliser .format pour l'arrondi (2 décimales pour toutes les colonnes métriques)
                df_styled = df_comparatif_final[cols_affichage].style.format(precision=2, subset=cols_a_comparer)

                # 2. Utiliser .apply(axis=1) pour la coloration ciblée sur la ligne Écart (Valeur)
                df_styled = df_styled.apply(
                    apply_color_on_row, 
                    axis=1, 
                    subset=cols_affichage 
                )
                
                # 6b. Affichage
                st.dataframe(
                df_styled,
                use_container_width=True, 
                hide_index=True
                )

                # ---------------------------------------------------------------------------------
                # 1. DÉFINITION DE LA CIBLE (J+1)
                # ---------------------------------------------------------------------------------
                jour_a_ajuster = None
                df_moyenne_seance_future = None

                # Définition des cibles (Moyenne BRUTE J-X)
                # NOTE: Ces clés DOIVENT contenir la charge MOYENNE INDIVIDUELLE de la SÉANCE BRUTE J-X.
                if jour_reference == 'J-4':
                    jour_a_ajuster = 'J-3'
                    # On veut la moyenne BRUTE du J-3
                    df_moyenne_seance_future = st.session_state.get('ref_j3_brute') # Assurez-vous que cette clé contient la moyenne J-3 NON cumulée
                    
                elif jour_reference == 'J-3':
                    jour_a_ajuster = 'J-2'
                    # On veut la moyenne BRUTE du J-2
                    df_moyenne_seance_future = st.session_state.get('ref_j2_brute') 
                    
                elif jour_reference == 'J-2':
                    jour_a_ajuster = 'J-1'
                    # On veut la moyenne BRUTE du J-1
                    df_moyenne_seance_future = st.session_state.get('ref_j1_brute')
                    
                elif jour_reference == 'J-1':
                    jour_a_ajuster = 'Prochain microcycle (J-4)'
                    # On veut la moyenne BRUTE du J-4 pour commencer le prochain cycle
                    df_moyenne_seance_future = st.session_state.get('ref_j4_brute')


                # ---------------------------------------------------------------------------------
                # 2. FILTRAGE ET CALCUL
                # ---------------------------------------------------------------------------------
                if jour_a_ajuster and df_moyenne_seance_future is not None:
                    
                    # 🎯 CORRECTION : Utiliser le nom de la joueuse qui est actuellement affichée (df_joueuse_indiv)
                    if not df_joueuse_indiv.empty:
                        joueuse_selectionnee_reelle = df_joueuse_indiv['Name'].iloc[0]
                    else:
                        # Solution de repli, même si logiquement df_joueuse_indiv ne devrait pas être vide ici
                        joueuse_selectionnee_reelle = st.session_state.get('selected_player', 'Nom Inconnu')

                    # ⚠️ Filtre crucial : Ne garder que la ligne de la joueuse sélectionnée dans le DF de moyenne future
                    df_moyenne_joueuse = df_moyenne_seance_future[df_moyenne_seance_future['Name'] == joueuse_selectionnee_reelle]
                    
                    
                    # Calculer la Charge Corrigée (uniquement si l'ajustement n'est pas vide)
                    if not df_ajustement_recommande.empty and not df_moyenne_joueuse.empty:
                        
                        df_charge_corrigee = appliquer_soustraction_brute(
                            df_ecart_brut=df_ecart, 
                            df_moyenne_seance_future=df_moyenne_joueuse # La moyenne BRUTE du jour J+1
                        )

                        # --------------------------------------------------------------------------
                        # 3. AFFICHAGE FINAL (Le seul tableau que vous voulez)
                        # --------------------------------------------------------------------------
                        st.markdown("---")
                        st.header("🎯 Prescription : Séance Corrigée")
                        st.subheader(f"Charge corrigée séance **{jour_a_ajuster}**")
                        
                        st.dataframe(
                            df_charge_corrigee.style
                                    .format(precision=1)
                                    .highlight_max(axis=0, props='font-weight: bold; background-color: #e0f7fa;'),
                                use_container_width=True, # Important pour remplir la colonne
                                hide_index=True
                        )

                        # Affichage de la recommandation verbale (elle utilise df_ajustement_recommande)
                        # st.markdown(f"**Recommandation Globale :** {formuler_recommendation_future(df_ajustement_recommande, jour_a_ajuster)}")
                        
                    else:
                        st.warning(f"Données manquantes ou ajustement non nécessaire pour {jour_a_ajuster}.")
        else:
            st.error(f"Le jour de référence '{jour_reference}' n'a pas de DataFrame de référence défini dans le dictionnaire.")


        # ======================================================================
        # 7. BLOC GRAPHIQUES ET TABLEAUX PAR SEMAINE (INDENTATION CORRIGÉE)
        # ======================================================================

        st.markdown("---")
        st.subheader("Données par semaine")
        
        df_par_semaine_joueuse = regrouper_par_semaine_civile(df_joueuse_indiv.copy())
        df_par_semaine_joueuse = df_par_semaine_joueuse.sort_values('Semaine', ascending=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            creer_graph_dt(
                df=df_par_semaine_joueuse,
                x_col='Semaine',
                y_col='Total Distance (m)',
                titre='Distance totale par semaine',
                couleur_barre='gold',
                x_label='Semaine',
                y_label='Total Distance (m)'
            )
        
        couleurs_spr = {
            'SPR Total Distance (m)': 'firebrick',
            'SPR + Total Distance (m)': 'red'
        }

        with col3:
            creer_graph_spr(
            df= df_par_semaine_joueuse, 
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
            df=df_par_semaine_joueuse,
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
            df=df_par_semaine_joueuse,
            x_col='Semaine',
            y_cols=['Accel >2m.s²', 'Decel >2m.s²'],
            titre='Nombre accel et decel par semaine',
            couleur_barre=couleurs_accel,
            x_label='Semaine',
            y_label='Unité arbitraire'
            )
        
        couleurs_rpe = {
            's.Cardio': 'grey', 
            's.Muscu': '#FF1493' 
        }

        with col2:
            creer_graph_accel_charge(
            df=df_par_semaine_joueuse,
            x_col='Semaine',
            y_cols=['s.Cardio', 's.Muscu'],
            titre='Charge int par semaine',
            couleur_barre=couleurs_rpe,
            x_label='Semaine',
            y_label='Unité arbitraire'
            )

        with col3:
            df_par_semaine_joueuse_display = df_par_semaine_joueuse[['Semaine', 'Total Time', 'Field Time', 'V max']]
            st.dataframe(
                df_par_semaine_joueuse_display.style
                    .format(precision=1)
                    .highlight_max(axis=0, props='font-weight: bold; background-color: #e0f7fa;'),
                use_container_width=True,
                hide_index=True 
            )

    else:
        st.info("Aucune donnée disponible pour cette joueuse dans cette période.")
else:
    st.info("Aucune donnée disponible pour cette sélection.")