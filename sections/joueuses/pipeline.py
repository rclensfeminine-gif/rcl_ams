import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
import unicodedata
from pandas.io.formats.style import Styler
from datetime import date
from plotly.subplots import make_subplots

#################################
# Fiche joueuse #
#################################

# Liste des colonnes de plis à comparer (À ADAPTER si cette liste n'est pas correcte)
PLIS_COLUMNS = ['Biceps', 'Triceps', 'Sous-Scap', 'Sup-Iliaque', 'Sub-Spinal', 'Abdo', 'Mollet']


def preparer_donnees_plis_comparaison(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforme le DataFrame du format large au format long (melt) pour les plis cutanés.
    Ne conserve que 'Date' comme variable d'identification.
    """
    df_long = df.melt(
        id_vars=['Date'], # <-- CORRIGÉ : Retrait de 'Nom Complet'
        value_vars=[col for col in PLIS_COLUMNS if col in df.columns],
        var_name='Pli Cutané',
        value_name='Mesure (mm)'
    ).dropna(subset=['Mesure (mm)'])
    return df_long

def creer_graph_comparatif_et_somme_plis(df_anthropo_plis_j: pd.DataFrame, titre: str):
    """
    Crée un graphique combiné (Barres + Ligne) avec double axe Y.
    - Barres (Axe Y1): Plis individuels comparés par date.
    - Ligne/Points (Axe Y2): Somme des plis (∑ Plis).
    """
    
    # 1. Préparation des données pour les barres (format long)
    df_plis_long = preparer_donnees_plis_comparaison(df_anthropo_plis_j)

    # 2. Préparation des données pour la ligne (format initial)
    df_somme_plis = df_anthropo_plis_j[['Date', '∑ Plis']].dropna(subset=['∑ Plis']).sort_values(by='Date')

    # 3. Création de la figure avec double axe Y
    fig = make_subplots(
        specs=[[{"secondary_y": True}]],
        figure=go.Figure(
            layout=go.Layout(
                # Ajustement des marges pour laisser de la place au texte des barres
                margin=dict(t=50, b=50, l=50, r=80), 
                title_text=f"<b>{titre}</b>", 
                title_font_size=20
            )
        )
    )

    # Couleurs pour les plis individuels (utilisant la palette Plotly par défaut)
    colors = px.colors.qualitative.Plotly

    # 4. Ajout des Barres (Plis Individuels) sur l'axe Y principal (Barres groupées)
    for i, pli in enumerate(PLIS_COLUMNS):
        if pli in df_anthropo_plis_j.columns:
            df_subset = df_plis_long[df_plis_long['Pli Cutané'] == pli]
            
            fig.add_trace(
                go.Bar(
                    x=df_subset['Date'],
                    y=df_subset['Mesure (mm)'],
                    name=pli,
                    marker_color=colors[i % len(colors)],
                    width=0.1, # Largeur réduite pour un graphique groupé
                    texttemplate='<b>%{y:.1f}</b>', # Valeurs des plis en gras avec 1 décimale
                    textposition='outside',
                    hovertemplate=f"<b>{pli}</b><br>Date: %{{x|%d-%m-%Y}}<br>Mesure: %{{y:.1f}} mm<extra></extra>"
                ),
                secondary_y=False,
            )

    # 5. Ajout de la Ligne et des Points (Somme des Plis) sur le DEUXIÈME axe Y
    # Pour le texte au-dessus de la ligne, nous devons maintenant nous assurer d'utiliser 'Date' 
    # comme base pour le tri et l'affichage des points.
    fig.add_trace(
        go.Scatter(
            x=df_somme_plis['Date'],
            y=df_somme_plis['∑ Plis'],
            name='∑ Plis (mm)',
            mode='lines+markers+text', # Afficher la ligne, les points, ET le texte
            marker=dict(color='black', size=10, symbol='circle'),
            line=dict(color='black', width=3),
            text=df_somme_plis['∑ Plis'].round(0).astype(int).astype(str).apply(lambda x: f"<b>{x}</b>"), # Texte en gras sans décimale
            textposition="top center",
            hovertemplate=f"<b>Somme Plis</b><br>Date: %{{x|%d-%m-%Y}}<br>∑ Plis: %{{y:.0f}} mm<extra></extra>"
        ),
        secondary_y=True,
    )

    # 6. Mise en page et configuration des axes
    
    # Configuration de l'axe Y principal (Barres - Plis individuels)
    fig.update_yaxes(
        title_text="<b>Mesure Plis Individuels (mm)</b>", 
        secondary_y=False,
        showgrid=True,
        gridcolor='lightgray'
    )

    # Configuration du deuxième axe Y (Ligne - Somme des plis)
    fig.update_yaxes(
        title_text="<b>Somme des Plis (mm)</b>", 
        secondary_y=True,
        showgrid=False, # Désactiver la grille pour le 2e axe Y
        range=[df_somme_plis['∑ Plis'].min() * 0.9, df_somme_plis['∑ Plis'].max() * 1.02], # Ajustement dynamique de la plage
        color='red'
    )
    
    # Configuration de l'axe X
    fig.update_xaxes(
        title_text="<b>Date de Mesure</b>",
        type='category', # Important pour que les barres se groupent correctement
        tickformat="%d-%m-%Y"
    )
    
    # Rendre le graphique réactif à la colonne de Streamlit
    fig.update_layout(
        # Ajuster la taille en fonction du conteneur Streamlit (la colonne large)
        autosize=True,
        height=550, 
        barmode='group', # Barres côte à côte
        legend_title_text='Pli Mesuré',
        # Augmenter l'espace entre le graphique et le bas pour le texte 'outside'
        margin=dict(b=70) 
    )

    # 7. Affichage
    st.plotly_chart(fig, use_container_width=True)


# Iscocinétisme 
#########################################

def colorer_seuil_difference(valeur):
    """
    Applique une couleur de fond basée sur la valeur absolue de la différence (en pourcentage):
    - Vert : < 10%
    - Orange : [10%, 15%]
    - Rouge : > 15%
    """
    if pd.isna(valeur):
        return ''
    
    # On travaille avec la valeur absolue de la différence
    valeur_abs = abs(valeur)
    
    # 3. Seuil Rouge (Critique) : > 15
    if valeur_abs > 15:
        # Rouge vif pour indiquer un déséquilibre critique
        return 'background-color: #ff6666; color: black; font-weight: bold;' 
    
    # 2. Seuil Orange (Alerte) : [10, 15]
    elif 10 <= valeur_abs <= 15:
        # Orange pour une surveillance rapprochée
        return 'background-color: #ffb84d; color: black; font-weight: bold;' 
    
    # 1. Seuil Vert (Optimal) : < 10
    elif 0 <= valeur_abs < 10:
        # Vert clair pour indiquer l'équilibre
        return 'background-color: #d1ffc9; color: black;' 
        
    return

RATIO_60 = ['Ratio IJ/Q60° D', 'Ratio IJ/Q60° G']
RATIO_240 = ['Ratio IJ/Q240° D', 'Ratio IJ/Q240° G']
RATIO_MIXTE = ['Ratio Mixte D', 'Ratio Mixte G']

def colorer_seuil_ratio(serie):
    """
    Applique une couleur de fond basée sur la valeur du ratio et la colonne, 
    selon les seuils spécifiés par l'utilisateur (Vert/Rouge).
    
    CORRECTION: Ajout d'une vérification de type au début pour éviter 
    l'erreur 'float' object has no attribute 'items' si un NaN est passé seul.
    """
    # Si l'entrée n'est pas une Series (par exemple, si c'est un float/NaN), 
    # cela signifie qu'il n'y a pas de données à traiter, donc on retourne une liste vide.
    if not isinstance(serie, pd.Series):
        return [''] * len(serie) if isinstance(serie, list) else ['']
    
    styles = []
    
    for col, valeur in serie.items():
        # Gérer les valeurs manquantes à l'intérieur de la Series (ligne)
        if pd.isna(valeur):
            styles.append('')
            continue
        
        is_vert = False
        
        # Logique pour Ratio IJ/Q60° D/G : vert si [0.5, 0.7]
        if col in RATIO_60:
            if 0.5 <= valeur <= 0.7:
                is_vert = True
        
        # Logique pour Ratio IJ/Q240° D/G : vert si [0.65, 0.85]
        elif col in RATIO_240:
            if 0.65 <= valeur <= 0.85:
                is_vert = True
                
        # Logique pour Ratio Mixte D/G : vert si [1, 1.4]
        elif col in RATIO_MIXTE:
            if 1.0 <= valeur <= 1.4:
                is_vert = True
        
        # Appliquer le style
        if is_vert:
            styles.append('background-color: #d1ffc9; color: black;') # Vert clair
        else:
            styles.append('background-color: #ff6666; color: black; font-weight: bold;') # Rouge vif
            
    return styles


COLONNES_RATIO_FIB_TIB = [
    'Ratio fibulaire / tibial post D', 
    'Ratio fibulaire / tibial post G' 
]

COLONNES_RATIO_ADD_ABD = [
    'Ratio ADD / ABD D', 
    'Ratio ADD / ABD G'
]

def colorer_seuil_ratio_fib_abd(serie):
    """
    Applique une couleur de fond basée sur la valeur du ratio et la colonne, 
    selon les seuils spécifiés par l'utilisateur (Vert/Rouge).
    
    CORRECTION: Ajout d'une vérification de type au début pour éviter 
    l'erreur 'float' object has no attribute 'items' si un NaN est passé seul.
    """
    # Si l'entrée n'est pas une Series (par exemple, si c'est un float/NaN), 
    # cela signifie qu'il n'y a pas de données à traiter, donc on retourne une liste vide.
    if not isinstance(serie, pd.Series):
        return [''] * len(serie) if isinstance(serie, list) else ['']
    
    styles = []
    
    for col, valeur in serie.items():
        # Gérer les valeurs manquantes à l'intérieur de la Series (ligne)
        if pd.isna(valeur):
            styles.append('')
            continue
        
        is_vert = False
        
        # Logique pour Ratio IJ/Q60° D/G : vert si [0.5, 0.7]
        if col in COLONNES_RATIO_FIB_TIB:
            if 0.8 <= valeur <= 1.3:
                is_vert = True
        
        # Logique pour Ratio IJ/Q240° D/G : vert si [0.65, 0.85]
        elif col in COLONNES_RATIO_ADD_ABD:
            if 80 <= valeur:
                is_vert = True
        
        # Appliquer le style
        if is_vert:
            styles.append('background-color: #d1ffc9; color: black;') # Vert clair
        else:
            styles.append('background-color: #ff6666; color: black; font-weight: bold;') # Rouge vif
            
    return styles

def style_performance_improvements(df: pd.DataFrame) -> Styler:
    """
    Applique une mise en forme conditionnelle à un DataFrame pour mettre en évidence 
    une amélioration (> 10% en vert) ou une régression (> 10% en rouge) par rapport
    à la mesure précédente de chaque colonne, et limite les décimales à zéro.
    Exclut explicitement les colonnes 'Date Test' et 'Remarque' du formatage numérique.

    Args:
        df: DataFrame contenant les colonnes de performance.

    Returns:
        Un objet Styler de Pandas avec les styles appliqués.
    """

    df_data = df.copy()
    
    # Définir les colonnes à exclure des calculs et du formatage numérique
    EXCLUDE_COLS = ['Date Test', 'Remarque']
    
    date_column_name = 'Date Test' if 'Date Test' in df_data.columns else None
    remark_column_name = 'Remarque' if 'Remarque' in df_data.columns else None
    
    # 1. PRÉPARATION DES DONNÉES ET CALCUL DU CHANGEMENT RELATIF
    
    # Séparer les colonnes de données numériques
    data_cols = [col for col in df_data.columns if col not in EXCLUDE_COLS]
    
    # Créer le DataFrame des données numériques pour les calculs
    df_numeric = df_data[data_cols].copy()
    
    # CONVERSION CRUCIALE : Assurer que les colonnes de données sont numériques
    # Les valeurs non convertibles deviendront NaN, empêchant l'erreur 'f' sur 'str'.
    df_numeric = df_numeric.apply(pd.to_numeric, errors='coerce')
    
    # Calculer le changement relatif uniquement sur les données numériques
    df_previous = df_numeric.shift(1)
    df_relative_change = (df_numeric - df_previous) / df_previous

    # Remplacer les Inf et les NaN par 0 dans les changements relatifs pour le stylage
    df_relative_change = df_relative_change.replace([float('inf'), float('-inf')], 0).fillna(0)

    # 2. APPLICATION DU STYLE DE COULEUR
    
    # Créer un DataFrame de styles (vides par défaut) aligné sur df_data pour le style final
    df_styles = pd.DataFrame('', index=df_data.index, columns=df_data.columns)
    
    SEUIL_CHGT = 0.10
    
    # Définir les styles CSS
    STYLE_VERT = 'background-color: #D4EDDA; color: #155724; font-weight: bold;' # Vert clair (Amélioration)
    STYLE_ROUGE = 'background-color: #F8D7DA; color: #721C24; font-weight: bold;' # Rouge clair (Régression)

    # Appliquer les styles uniquement aux colonnes de données numériques
    for col in data_cols:
        # Masque pour l'amélioration (> +10%)
        mask_amelioration = df_relative_change[col] > SEUIL_CHGT
        
        # Masque pour la régression (< -10%)
        mask_regression = df_relative_change[col] < -SEUIL_CHGT
        
        # Appliquer le style Vert
        df_styles.loc[mask_amelioration, col] = STYLE_VERT
        
        # Appliquer le style Rouge
        df_styles.loc[mask_regression, col] = STYLE_ROUGE
        
    # 3. RETOURNER LE STYLER ET APPLIQUER LE FORMATAGE
    
    # Appliquer les styles au DataFrame original (qui contient 'Date Test' et 'Remarque')
    styled_df = df_data.style.apply(lambda x: df_styles, axis=None)

    # 3b. Limiter les décimales à 0 pour les colonnes NUMÉRIQUES SEULEMENT
    format_mapping = {}
    
    # Ajouter le format numérique pour les colonnes de performance
    for col in data_cols:
        # Utiliser 'na_rep' dans .format() plus tard pour les NaN, mais le format est bien '{:.0f}'
        format_mapping[col] = '{:.0f}'
    
    # Ajouter les formats de chaîne pour les colonnes exclues (si elles existent)
    if date_column_name:
        format_mapping[date_column_name] = '{}' # Format string simple
    if remark_column_name:
        format_mapping[remark_column_name] = '{}' # Format string simple
    
    # Appliquer le formatage
    styled_df = styled_df.format(format_mapping, na_rep='-')

    return styled_df




#################################
# Suivi joueuse #
#################################

# Liste complète des noms de plis
PLIS_NOMS = ['Biceps', 'Triceps', 'Sous-Scap', 'Sup-Illiaque', 'Sub-Spinal', 'Abdo', 'Quadriceps', 'Mollet']
# Générer les noms des colonnes pour les prises individuelles
PLIS_PRISES_COLS = [f"{p}_1" for p in PLIS_NOMS] + [f"{p}_2" for p in PLIS_NOMS]

### Sauvegarde

DOSSIER_DATA = 'data/suivi'
FICHIER_ANTHROPO_SUIVI = 'data/suivi/anthropo_suivi.csv'
FICHIER_ANTHROPO_FIXE = 'data/suivi/anthropo_fixes.csv'

def sauvegarder_suivi_global(df):
    """
    Sauvegarde le DataFrame mis à jour dans le fichier CSV.
    """
    try:
        # Créer une copie pour la manipulation des dates sans modifier l'original en place
        df_to_save = df.copy()
        
        # Convertir les objets date.date en string pour une écriture CSV propre
        if 'Date' in df_to_save.columns:
            # Utilise .dt.strftime('%Y-%m-%d') pour une gestion propre de l'objet date
            # Appliquer seulement si ce n'est pas déjà None ou NaN
            df_to_save['Date'] = df_to_save['Date'].apply(
                lambda x: x.isoformat() if pd.notna(x) else None
            ) 
            
        # Écriture robuste: index=False est essentiel
        df_to_save.to_csv(
            FICHIER_ANTHROPO_SUIVI, 
            index=False, 
            na_rep='', # Représenter les valeurs manquantes comme vides
            encoding='utf-8'
        )
        return True
    except Exception as e:
        st.error(f"Échec de la sauvegarde dans le fichier CSV : {e}")
        return False


def sauvegarder_fixes(df_modifie):
    """Sauvegarde les données fixes (Taille, EIAS, Poignet) par écrasement."""
    if not os.path.exists(DOSSIER_DATA): os.makedirs(DOSSIER_DATA)
    try:
        colonnes_a_sauver = ['Prénom', 'NOM', 'Taille (cm)', 'EIAS - Maléole D', 'EIAS - Maléole G', 'Tour Poignet (cm)']
        # Filtrer le DataFrame pour ne garder que les colonnes et les lignes valides
        df_modifie = df_modifie.dropna(subset=['Prénom', 'NOM']).copy()
        df_modifie = df_modifie[colonnes_a_sauver]
        df_modifie.to_csv(FICHIER_ANTHROPO_FIXE, index=False, encoding='utf-8')
        return True
    except Exception as e:
        st.error(f"Erreur Sauvegarde Fixe : {e}")
        return False
    
def sauvegarder_df_global(df_final, chemin_fichier):
    """
    Sauvegarde l'intégralité du DataFrame df_final dans le fichier spécifié
    par chemin_fichier. Utilisée pour l'Upsert (Update/Insert) et la Suppression.
    """
    if not os.path.exists(DOSSIER_DATA): 
        os.makedirs(DOSSIER_DATA)
        
    try:
        df_final.to_csv(chemin_fichier, index=False, encoding='utf-8')
        return True 
    except Exception as e:
        st.error(f"Erreur Sauvegarde Globale ({chemin_fichier}): {e}")
        return False


# --- Fonction d'initialisation ---
def get_initial_value_suivi(colonne, mesure_a_modifier: pd.DataFrame):
    """
    Retourne la valeur de la mesure à modifier pour la date sélectionnée, sinon None.
    Ceci est pour les champs numériques (Poids, Plis).
    """
    # S'il y a des données pour la date sélectionnée (mode MODIFICATION)
    if not mesure_a_modifier.empty and colonne in mesure_a_modifier.columns:
        # Utiliser .iloc[0] pour accéder à la ligne unique
        valeur = mesure_a_modifier.iloc[0].get(colonne)
        if pd.notna(valeur):
            try:
                # Retourne la valeur en float
                return float(valeur) 
            except ValueError:
                return None
    return None # Retourne None si aucune donnée n'est trouvée (mode AJOUT)

def init_session_state_poids_plis(mesure_a_modifier: pd.DataFrame, date_a_comparer: date):
    """
    Initialise st.session_state pour le poids, la remarque et tous les plis.
    """
    
    POIDS_KEY = "poids_input"
    REMARQUE_KEY = "remarque_input" # <- Nouvelle clé pour la remarque
    
    # Vérifie si la date a changé (pour déclencher le rechargement/réinitialisation)
    is_new_date_selected = st.session_state.get('last_date_checked') != date_a_comparer
    
    if is_new_date_selected:
        # --- BLOC DE RÉINITIALISATION/CHARGEMENT LORSQUE LA DATE CHANGE ---
        st.session_state['last_date_checked'] = date_a_comparer # Marque la date vérifiée
        
        # 1. Gestion du Poids
        initial_poids = get_initial_value_suivi('Poids (kg)', mesure_a_modifier)
        st.session_state[POIDS_KEY] = initial_poids
        
        # 2. Gestion de la Remarque (NOUVEAU)
        initial_remarque = None
        if not mesure_a_modifier.empty and 'Remarque' in mesure_a_modifier.columns:
            # Récupère la chaîne de caractères brute
            remarque_val = mesure_a_modifier.iloc[0].get('Remarque')
            if pd.notna(remarque_val):
                initial_remarque = str(remarque_val)
                
        # Initialise la remarque à la valeur chargée ou à une chaîne vide
        st.session_state[REMARQUE_KEY] = initial_remarque or ""
        
        # 3. Gestion des Plis Cutanés 
        for pli_name in PLIS_NOMS:
            # Récupère la moyenne (colonne 'Biceps') pour rétro-compatibilité
            initial_avg_value = get_initial_value_suivi(pli_name, mesure_a_modifier)
            
            for prise_num in [1, 2]:
                key_pli_session = f"pli_{pli_name}_{prise_num}"
                key_pli_db = f"{pli_name}_{prise_num}"
                
                # Tente de charger la valeur individuelle (Biceps_1 ou Biceps_2)
                initial_value_db = get_initial_value_suivi(key_pli_db, mesure_a_modifier)
                
                if initial_value_db is not None:
                    st.session_state[key_pli_session] = initial_value_db
                elif initial_avg_value is not None:
                    st.session_state[key_pli_session] = initial_avg_value
                else:
                    st.session_state[key_pli_session] = None

def generate_pli_inputs_optimized(pli_name):
    """
    Crée les deux inputs pour une seule localisation de pli et gère l'état via st.session_state.
    """
    st.markdown(f"**{pli_name}**")
    
    # Clés basées sur les noms de colonne dans session_state
    key_1 = f"pli_{pli_name}_1"
    key_2 = f"pli_{pli_name}_2"
    
    col_p1, col_p2 = st.columns(2)
    
    with col_p1:
        st.number_input(
            "Prise 1 (mm)", 
            value=st.session_state.get(key_1), 
            min_value=0.0, 
            format="%.1f", 
            key=key_1,
            label_visibility="collapsed"
        )
        
    with col_p2:
        st.number_input(
            "Prise 2 (mm)", 
            value=st.session_state.get(key_2),
            min_value=0.0, 
            format="%.1f", 
            key=key_2,
            label_visibility="collapsed"
        )
        
    st.markdown("---") # Séparateur visuel entre les plis
                        

### Calcules 
def calculer_ratios_isocinetisme(df):
    """Calcule les 6 ratios (IJ/Q à 60° et 240°, et Mixte IJExc/Q240°)."""
    df_temp = df.copy()

    # S'assurer que les colonnes sont numériques pour le calcul, les valeurs non numériques deviennent NaN
    cols_numeriques = [
        'Q60° D', 'Q60° G', 'IJ60° D', 'IJ60° G', 
        'Q240° D', 'Q240° G', 'IJ240° D', 'IJ240° G', 
        'IJExc D', 'IJExc G'
    ]
    for col in cols_numeriques:
        if col in df_temp.columns:
            df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce')

    # Liste des calculs à effectuer : (Nom de la colonne Ratio, Numérateur D, Dénominateur D, Numérateur G, Dénominateur G)
    ratios = [
        # 1. Ratio IJ/Q 60°
        ('Ratio IJ/Q60° D', 'IJ60° D', 'Q60° D', 'Ratio IJ/Q60° G', 'IJ60° G', 'Q60° G'),
        
        # 2. Ratio IJ/Q 240°
        ('Ratio IJ/Q240° D', 'IJ240° D', 'Q240° D', 'Ratio IJ/Q240° G', 'IJ240° G', 'Q240° G'),
        
        # 3. Ratio Mixte (IJ Exc / Q 240°)
        ('Ratio Mixte D', 'IJExc D', 'Q240° D', 'Ratio Mixte G', 'IJExc G', 'Q240° G')
    ]
    
    # Exécution des calculs
    for ratio_d_col, num_d_col, den_d_col, ratio_g_col, num_g_col, den_g_col in ratios:
        
        # Ratio Droit
        if num_d_col in df_temp.columns and den_d_col in df_temp.columns:
            # Calcul : (Numérateur / Dénominateur), arrondi à 2 décimale
            df_temp[ratio_d_col] = (df_temp[num_d_col] / df_temp[den_d_col]).round(2)
            
        # Ratio Gauche
        if num_g_col in df_temp.columns and den_g_col in df_temp.columns:
            df_temp[ratio_g_col] = (df_temp[num_g_col] / df_temp[den_g_col]).round(2)

    return df_temp

def calculer_metriques_hop_test(df):
    """
    Calcule la moyenne (Mean), le maximum (Max) et l'indice de symétrie 
    (Sym = |(Max G - Max D)| / Max D * 100) pour les tests SHT, THT, et CHT.
    """
    # Si le DF est vide, retourne un DF vide sans planter
    if df.empty:
        # Créer un DF avec les colonnes de métriques si possible pour l'initialisation
        cols_metriques = [
            'Mean SHT D', 'Mean SHT G', 'Max SHT D', 'Max SHT G', 'Sym SHT', 
            'Mean THT D', 'Mean THT G', 'Max THT D', 'Max THT G', 'Sym THT', 
            'Mean CHT D', 'Mean CHT G', 'Max CHT D', 'Max CHT G', 'Sym CHT'
        ]
        return pd.DataFrame(columns=cols_metriques)
        
    df_temp = df.copy()

    # Liste des tests et leurs colonnes
    tests = [
        ('SHT', ['SHT D1', 'SHT D2', 'SHT D3'], ['SHT G1', 'SHT G2', 'SHT G3']),
        ('THT', ['THT D1', 'THT D2', 'THT D3'], ['THT G1', 'THT G2', 'THT G3']),
        ('CHT', ['CHT D1', 'CHT D2', 'CHT D3'], ['CHT G1', 'CHT G2', 'CHT G3']),
    ]
    
    # Assurer que toutes les colonnes de mesure sont numériques
    all_hop_cols = [col for _, d_cols, g_cols in tests for col in d_cols + g_cols]
    for col in all_hop_cols:
        if col in df_temp.columns:
            # Force la conversion, les valeurs non valides deviennent NaN (float)
            df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce')

    for prefixe, d_cols, g_cols in tests:
        mean_d_col = f'Mean {prefixe} D'
        mean_g_col = f'Mean {prefixe} G'
        max_d_col = f'Max {prefixe} D'
        max_g_col = f'Max {prefixe} G'
        sym_col = f'Sym {prefixe}'
        
        # 1. Calcul de la MOYENNE et du MAX
        
        # Vérification si les colonnes existent avant de calculer la moyenne/max
        d_cols_exist = [col for col in d_cols if col in df_temp.columns]
        g_cols_exist = [col for col in g_cols if col in df_temp.columns]

        if d_cols_exist:
            df_temp[mean_d_col] = df_temp[d_cols_exist].mean(axis=1).round(2)
            df_temp[max_d_col] = df_temp[d_cols_exist].max(axis=1).round(2)
        else:
            df_temp[mean_d_col] = np.nan
            df_temp[max_d_col] = np.nan
            
        if g_cols_exist:
            df_temp[mean_g_col] = df_temp[g_cols_exist].mean(axis=1).round(2)
            df_temp[max_g_col] = df_temp[g_cols_exist].max(axis=1).round(2)
        else:
            df_temp[mean_g_col] = np.nan
            df_temp[max_g_col] = np.nan
            
        # 2. Calcul de la SYMÉTRIE/DÉSÉQUILIBRE (Sym)
        # On utilise .get() pour s'assurer que les colonnes existent
        max_d = df_temp.get(max_d_col)
        max_g = df_temp.get(max_g_col)
        
        if max_d is not None and max_g is not None:
            # Remplacement des 0 par NaN pour éviter une division par zéro
            max_d_safe = max_d.replace(0, np.nan)
            
            # Calcul de l'indice de déséquilibre : |(Max G - Max D)| / Max D * 100
            # Si le Max D est NaN ou 0, le résultat sera NaN, ce qui est correct.
            df_temp[sym_col] = (abs((max_g - max_d) / max_d_safe) * 100).round(2)
        else:
            df_temp[sym_col] = np.nan

    return df_temp.reset_index(drop=True)

def calculer_metriques_sauts(df):
    """
    Calcule la valeur maximale (Max) pour différents tests de saut :
    CMJ, CMJ Bras, CMJ 1 Jambe Droit/Gauche, SRJT 5 Mean, SRJT 5 RSI.
    """
    # Si le DF est vide, retourne un DF vide avec les colonnes de sortie
    tests_max_cols = [
        'Max CMJ', 'Max CMJ Bras', 'Max CMJ 1J D', 'Max CMJ 1J G', 
        'Max SRJT 5 Mean', 'Max SRJT 5 RSI'
    ]
    if df.empty:
        return pd.DataFrame(columns=tests_max_cols)
        
    df_temp = df.copy()

    # Définition des tests et des colonnes sources
    tests_max = {
        'Max CMJ': ['CMJ 1', 'CMJ 2', 'CMJ 3'],
        'Max CMJ Bras': ['CMJ Bras 1', 'CMJ Bras 2', 'CMJ Bras 3'],
        'Max CMJ 1J D': ['CMJ 1J D1', 'CMJ 1J D2', 'CMJ 1J D3'],
        'Max CMJ 1J G': ['CMJ 1J G1', 'CMJ 1J G2', 'CMJ 1J G3'],
        'Max SRJT 5 Mean': ['SRJT 5 Mean 1', 'SRJT 5 Mean 2', 'SRJT 5 Mean 3'],
        'Max SRJT 5 RSI': ['SRJT 5 RSI 1', 'SRJT 5 RSI 2', 'SRJT 5 RSI 3']
    }
    
    # 1. Assurer que toutes les colonnes sources sont numériques
    all_jump_cols = [col for cols in tests_max.values() for col in cols]
    for col in all_jump_cols:
        if col in df_temp.columns:
            df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce')

    # 2. Calculer le Maximum pour chaque test
    for output_col, source_cols in tests_max.items():
        # Vérifier que les colonnes sources existent dans le DF avant le calcul
        source_cols_exist = [col for col in source_cols if col in df_temp.columns]
        
        if source_cols_exist:
            # Calculer le maximum sur les colonnes existantes
            df_temp[output_col] = df_temp[source_cols_exist].max(axis=1).round(2)
        else:
            df_temp[output_col] = np.nan
            
    return df_temp.reset_index(drop=True)

def calculer_metriques_dynamo(df):
    """
    Calcule l'indice de Symétrie (Sym = G / D * 100) et les Ratios fonctionnels 
    pour les tests de dynamométrie.
    """
    # Colonnes de métriques attendues en sortie
    cols_metriques_dynamo = [
        'Sym soléaire', 'Sym gastro', 'Sym tibial post', 'Sym fibulaire', 
        'Sym abducteur', 'Sym adducteur', 
        'Ratio fibulaire / tibial post D', 'Ratio fibulaire / tibial post G', 
        'Ratio ADD / ABD D', 'Ratio ADD / ABD G'
    ]
    if df.empty:
        return pd.DataFrame(columns=cols_metriques_dynamo)

    df_temp = df.copy()

    # Définition des tests et des colonnes
    muscles = [
        ('Soléaire', 'Soléaire D', 'Soléaire G', 'Sym soléaire'),
        ('Gastro', 'Gastro D', 'Gastro G', 'Sym gastro'),
        ('Tibial post', 'Tibial post D', 'Tibial post G', 'Sym tibial post'),
        ('Fibulaire', 'Fibulaire D', 'Fibulaire G', 'Sym fibulaire'),
        ('Abducteur', 'Abducteur D', 'Abducteur G', 'Sym abducteur'),
        ('Adducteur', 'Adducteur D', 'Adducteur G', 'Sym adducteur'),
    ]
    
    # 1. Assurer que toutes les colonnes sources sont numériques
    all_muscles_cols = [col_d for _, col_d, col_g, _ in muscles] + [col_g for _, col_d, col_g, _ in muscles]
    for col in all_muscles_cols:
        if col in df_temp.columns:
            df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce')

    # 2. Calcul des Indices de Symétrie (Sym = G / D * 100)
    for _, col_d, col_g, sym_col in muscles:
        if col_d in df_temp.columns and col_g in df_temp.columns:
            val_g_safe = df_temp[col_g].replace(0, np.nan)
            val_d_safe = df_temp[col_d].replace(0, np.nan)
            df_temp[sym_col] = (abs((val_g_safe - val_d_safe) / val_d_safe)*100).round(2)
        else:
            df_temp[sym_col] = np.nan

    # 3. Calcul des Ratios Fonctionnels
    
    # Ratios 1 : Fibulaire / Tibial Post * 100 (Ratio d'eversion/inversion)
    fib_d = df_temp.get('Fibulaire D')
    tib_d = df_temp.get('Tibial post D')
    fib_g = df_temp.get('Fibulaire G')
    tib_g = df_temp.get('Tibial post G')
    
    if fib_d is not None and tib_d is not None:
        tib_d_safe = tib_d.replace(0, np.nan)
        df_temp['Ratio fibulaire / tibial post D'] = (fib_d / tib_d_safe).replace([np.inf, -np.inf], np.nan).round(2)
        
    if fib_g is not None and tib_g is not None:
        tib_g_safe = tib_g.replace(0, np.nan)
        df_temp['Ratio fibulaire / tibial post G'] = (fib_g / tib_g_safe).replace([np.inf, -np.inf], np.nan).round(2)

    # Ratios 2 : Adducteur / Abducteur * 100 (Ratio d'adduction/abduction)
    add_d = df_temp.get('Adducteur D')
    abd_d = df_temp.get('Abducteur D')
    add_g = df_temp.get('Adducteur G')
    abd_g = df_temp.get('Abducteur G')
    
    if add_d is not None and abd_d is not None:
        abd_d_safe = abd_d.replace(0, np.nan)
        # Note : Le ratio ADD/ABD est souvent exprimé en pourcentage
        df_temp['Ratio ADD / ABD D'] = (add_d * 100 / abd_d_safe ).replace([np.inf, -np.inf], np.nan).round(2)
        
    if add_g is not None and abd_g is not None:
        abd_g_safe = abd_g.replace(0, np.nan)
        df_temp['Ratio ADD / ABD G'] = (add_g * 100 / abd_g_safe ).replace([np.inf, -np.inf], np.nan).round(2)
            
    return df_temp.reset_index(drop=True)


########  Rapport de match indiv #########
def get_poste_all_match(df_raw):
            """
            Optimisation : Utilisation d'assign pour éviter les SettingWithCopyWarning
            et filtrage direct des colonnes pour l'agrégation.
            """
            # Calcul VHSR effort plus efficace
            if all(col in df_raw.columns for col in ['VHSR + SPR effort', 'Sprint effort']):
                df_raw = df_raw.assign(**{'VHSR effort': df_raw['VHSR + SPR effort'] - df_raw['Sprint effort']})

            id_cols = ['Player Name', 'Position Name', 'Activity Name', 'Type match', 'Saison']
            
            agg_logic = {
                'Field Time': 'sum',
                'Total Distance (m)': 'sum',
                'VHSR Total Distance (m)': 'sum',
                'Sprint (m)': 'sum',
                'VHSR effort': 'sum',
                'Sprint effort': 'sum',
                'Accel >2m.s²': 'sum',
                'Decel >2m.s²': 'sum',
                'V max': 'max',
                'Meterage Per Minute': 'mean'
            }
            
            # Intersection rapide des colonnes
            actual_agg = {k: v for k, v in agg_logic.items() if k in df_raw.columns}
            
            return df_raw.groupby(id_cols, as_index=False).agg(actual_agg)

def calculate_position_benchmarks(df):
            """
            Optimisation : Vectorisation de l'aplatissement des colonnes.
            """
            # Conversion groupée pour plus de rapidité
            df['Field Time'] = pd.to_numeric(df['Field Time'], errors='coerce')
            df_filtered = df[df['Field Time'] >= 60].copy()
            
            metrics = [
                'Total Distance (m)', 'VHSR Total Distance (m)', 'Sprint (m)', 
                'VHSR effort', 'Sprint effort', 'Accel >2m.s²', 
                'Decel >2m.s²', 'V max', 'Meterage Per Minute'
            ]
            
            available_metrics = [m for m in metrics if m in df_filtered.columns]
            
            if not available_metrics:
                return pd.DataFrame(), df_filtered

            # Agrégation groupée
            benchmarks = df_filtered.groupby('Position Name')[available_metrics].agg(['mean', 'max'])
            
            # Aplatissement vectorisé des colonnes
            benchmarks.columns = [f"{col[0]}_{'moyenne' if col[1] == 'mean' else 'max'}" for col in benchmarks.columns]
            
            return benchmarks.reset_index(), df_filtered

def normalize_text(text):
            """Version optimisée de la normalisation."""
            if not isinstance(text, str): return ""
            # Normalisation et mise en minuscule en une étape
            text = "".join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
            text = text.lower().strip()
            return text[:-1] if text.endswith('e') else text

def get_player_benchmarks(benchmarks_df, player_profile):
            """
            Récupère les benchmarks pour le poste de la joueuse.
            Correction : Gestion du cas où player_profile est une Series ou un Dict.
            """
            # CORRECTION : On vérifie si benchmarks_df est vide et si player_profile existe
            # Pour une Series Pandas, on utilise .empty, pour un dict on utilise la vérité booléenne
            is_profile_empty = player_profile.empty if hasattr(player_profile, 'empty') else not player_profile
            
            if benchmarks_df.empty or is_profile_empty:
                return None

            # 1. Normalisation du poste recherché
            # On utilise .get() si c'est un dict, ou l'accès classique si c'est une Series
            if hasattr(player_profile, 'get'):
                raw_poste = player_profile.get('1er Poste', '')
            else:
                raw_poste = player_profile['1er Poste'] if '1er Poste' in player_profile else ''
                
            target = normalize_text(raw_poste)
            if not target:
                return None

            # 2. Matching du poste
            # On crée une copie locale pour ne pas modifier le DF original
            df = benchmarks_df.copy()
            df['norm_pos'] = df['Position Name'].astype(str).apply(normalize_text)
            
            # Filtrage
            mask = df['norm_pos'].str.contains(target, na=False) | (target == df['norm_pos'])
            result = df[mask]

            if result.empty:
                return None

            # 3. Extraction et Pivot (Formatage pour l'affichage)
            # On prend la première ligne trouvée
            row = result.iloc[0].drop(['Position Name', 'norm_pos'])
            
            # Conversion en DataFrame temporaire pour pivoter
            temp_df = row.to_frame(name='value').reset_index()
            
            # On sépare le nom de la métrique et le type de stat (moyenne/max)
            # split sur le dernier '_' trouvé
            def split_metric_stat(x):
                if '_moyenne' in x:
                    return x.replace('_moyenne', ''), 'moyenne'
                if '_max' in x:
                    return x.replace('_max', ''), 'max'
                return x, 'moyenne'

            temp_df[['metric', 'stat']] = temp_df['index'].apply(lambda x: pd.Series(split_metric_stat(x)))
            
            # Pivot pour avoir l'index [moyenne, max]
            final_df = temp_df.pivot(index='stat', columns='metric', values='value')
            
            # Sécurité pour s'assurer que les deux lignes existent
            return final_df.reindex(["moyenne", "max"])

def display_stats_joueuse_cards(stats_joueuse):
            """
            Rendu optimisé : Séparation du CSS et boucle de rendu simplifiée.
            """
            if stats_joueuse is None or stats_joueuse.empty:
                st.warning("Données de statistiques indisponibles.")
                return
            
            # On injecte le CSS une seule fois
            st.markdown(get_css_styles(), unsafe_allow_html=True)

            def render_group(title, metrics_config, cols_count):
                st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)
                cols = st.columns(cols_count)
                
                # Filtrer uniquement les métriques présentes dans le DF pour éviter les erreurs
                valid_metrics = [m for m in metrics_config if m[0] in stats_joueuse.columns]
                
                for idx, (col_name, label, unit) in enumerate(valid_metrics):
                    m_val = stats_joueuse.at["moyenne", col_name]
                    x_val = stats_joueuse.at["max", col_name]
                    
                    # Formatage conditionnel
                    fmt = lambda v: f"{v:,.0f}" if v > 100 else f"{v:.1f}"
                    
                    with cols[idx % cols_count]:
                        st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-title">{label}</div>
                                <div class="metric-avg">{fmt(m_val)}<span class="metric-unit"> {unit}</span></div>
                                <div class="metric-max-label">Record : {fmt(x_val)} {unit}</div>
                            </div>
                        """, unsafe_allow_html=True)

            # Configuration des groupes
            groups = [
                ("📊 Volume", [("Total Distance (m)", "Distance Totale", "m"), ("Meterage Per Minute", "Rythme", "m/min")], 2),
                ("⚡ Intensité", [
                    ("VHSR Total Distance (m)", "Distance Haute Intensité", "m"),
                    ("Sprint (m)", "Dist. Sprint", "m"),
                    ("VHSR effort", "Efforts Haute Intensité", "nb"),
                    ("Sprint effort", "Efforts Sprint", "nb"),
                    ("V max", "Vitesse Max", "km/h")
                ], 3)
            ]

            for title, configs, count in groups:
                render_group(title, configs, count)

def get_css_styles():
            return """
            <style>
                .metric-card {
                    background-color: #ffffff; border: 1px solid #eef2f6; border-radius: 12px;
                    padding: 20px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 15px;
                }
                .metric-title { color: #64748b; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; margin-bottom: 8px; }
                .metric-avg { color: #2563eb; font-size: 1.75rem; font-weight: 800; line-height: 1; }
                .metric-unit { font-size: 0.85rem; color: #94a3b8; font-weight: 400; }
                .metric-max-label { margin-top: 10px; padding-top: 10px; border-top: 1px solid #f1f5f9; font-size: 0.8rem; color: #e11d48; font-weight: 600; }
                .section-header { font-size: 1.1rem; font-weight: 700; color: #1e293b; margin: 0 0 10px 0; padding-bottom: 5px; border-bottom: 2px solid #3b82f6; display: inline-block; }
            </style>
            """

def afficher_comparaison_match(joueuse_stats, stats_df):
        """
        Affiche les performances avec deux logiques distinctes :
        1. Delta : Comparaison à la MOYENNE (Vert ou Orange)
        2. Bloc Record : Proximité du MAXIMUM (Rouge, Orange ou Vert)
        """
        
        # CSS pour le style des indicateurs
        st.markdown("""
            <style>
            .delta-positive { color: #27ae60 !important; font-weight: bold; font-size: 0.9rem; }
            .delta-negative { color: #e67e22 !important; font-weight: bold; font-size: 0.9rem; }
            [data-testid="stMetricValue"] { font-size: 1.8rem !important; }
            </style>
        """, unsafe_allow_html=True)

        volume_metrics = [
            ('Field Time', 'Temps de Jeu', 'min'),
            ('Total Distance (m)', 'Distance Totale', 'm'),
            ('Meterage Per Minute', 'Rythme', 'm/min')
        ]
        
        intensite_metrics = [
            ('VHSR Total Distance (m)', 'Distance Haute Intensité 19-23km/h', 'm'),
            ('Sprint (m)', 'Distance Sprint 23km/h', 'm'),
            ('VHSR effort', 'Haute Intensité effort', 'nb'),
            ('Sprint effort', 'Sprint effort', 'nb'),
            ('V max', 'Vitesse Max', 'km/h')
        ]

        def render_metric_block(metrics, section_label):
            if not metrics:
                return
                
            st.markdown(f"#### {section_label}")
            cols = st.columns(len(metrics))
            
            for i, (key, label, unit) in enumerate(metrics):
                if key in joueuse_stats.index:
                    valeur_match = joueuse_stats[key]
                    
                    with cols[i]:
                        if key == 'Field Time':
                            st.metric(label=label, value=f"{valeur_match:.1f} {unit}")
                        
                        elif key in stats_df.index:
                            moyenne = stats_df.loc[key, 'Moyenne']
                            maximum = stats_df.loc[key, 'Maximum']
                            delta_vs_moy = valeur_match - moyenne
                            percent_max = (valeur_match / maximum * 100) if maximum > 0 else 0
                            
                            # --- LOGIQUE 1 : DELTA (Moyenne) ---
                            if delta_vs_moy > 0:
                                delta_class = "delta-positive"
                                symbol = "▲"
                            else:
                                delta_class = "delta-negative"
                                symbol = "▼"
                            
                            # --- LOGIQUE 2 : BLOC RECORD (Maximum) ---
                            if percent_max < 70:
                                record_color = "#969696" # Gris
                            elif percent_max < 90:
                                record_color = "#f6b243" # Jaune
                            else:
                                record_color = "#2ecc71" # Vert

                            # Affichage de la métrique principale
                            st.metric(label=label, value=f"{valeur_match:.1f} {unit}", delta=None)
                            
                            # Affichage du Delta personnalisé
                            st.markdown(f"""
                                <div style="margin-top: -15px; margin-bottom: 10px;">
                                    <span class="{delta_class}">{symbol} {delta_vs_moy:+.1f} {unit} vs moyenne</span>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            # Affichage du bloc Record avec sa propre couleur
                            st.markdown(f"""
                                <div style="padding: 6px; border-radius: 4px; border-left: 4px solid {record_color}; background-color: #f8f9fa; line-height:1.2;">
                                    <span style="color:#2c3e50; font-weight:bold; font-size:0.7rem;">RECORD : {maximum:.1f} {unit}</span><br>
                                    <span style="font-size:0.7rem; color:#666;"><b>{percent_max:.1f}% du max</b></span>
                                </div>
                            """, unsafe_allow_html=True)

        render_metric_block(volume_metrics, "🏃‍♂️ Volume")
        st.markdown("<div style='margin: 20px 0;'></div>", unsafe_allow_html=True)
        render_metric_block(intensite_metrics, "⚡ Intensité")


def afficher_radar_performance(dernier_match_stats, stats_globales):
        """
        Radar chart avec étiquettes de catégories ET pourcentages de l'axe en gras.
        Métriques : Distance Totale, m/min, VHSR Distance, Sprint Distance, VHSR effort, Sprint effort.
        """
        # Mapping des noms techniques vers noms d'affichage en gras
        mapping_categories = {
            'Total Distance (m)': '<b>Distance Total</b>',
            'Meterage Per Minute': '<b>Rythme</b>',
            'VHSR Total Distance (m)': '<b>Distance VHSR</b>',
            'VHSR effort': '<b>Haute Intensité effort</b>',
            'Sprint (m)': '<b>Distance Sprint</b>',
            'Sprint effort': '<b>Sprint Effort</b>'
        }
        
        # Filtrage des catégories présentes
        categories_tech = [c for c in mapping_categories.keys() 
                        if c in dernier_match_stats.index and c in stats_globales.index]
        
        categories_display = [mapping_categories[c] for c in categories_tech]
        
        match_values = []
        moyenne_values = []
        
        for cat in categories_tech:
            max_val = stats_globales.loc[cat, 'Maximum']
            match_perc = (dernier_match_stats[cat] / max_val * 100) if max_val > 0 else 0
            moy_perc = (stats_globales.loc[cat, 'Moyenne'] / max_val * 100) if max_val > 0 else 0
            
            match_values.append(match_perc)
            moyenne_values.append(moy_perc)

        if categories_display:
            categories_close = categories_display + [categories_display[0]]
            match_values += [match_values[0]]
            moyenne_values += [moyenne_values[0]]

            fig = go.Figure()

            # Guide Record 100%
            fig.add_trace(go.Scatterpolar(
                r=[100]*len(categories_close),
                theta=categories_close,
                name='Record',
                line=dict(color='rgba(46, 204, 113, 0.4)', width=2, dash='dash'),
                fill='none',
                hoverinfo='skip'
            ))

            # Zone Moyenne
            fig.add_trace(go.Scatterpolar(
                r=moyenne_values,
                theta=categories_close,
                fill='toself',
                name='Moyenne',
                line_color='goldenrod',
                fillcolor='rgba(218, 165, 32, 0.1)',
                hoverinfo='skip'
            ))

            # Zone Match
            fig.add_trace(go.Scatterpolar(
                r=match_values,
                theta=categories_close,
                fill='toself',
                name='Match du jour',
                line=dict(color='#1f77b4', width=3),
                fillcolor='rgba(31, 119, 180, 0.3)',
                hoverinfo='skip'
            ))

            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True, 
                        range=[0, 115], 
                        ticksuffix="%",
                        # MISE EN GRAS DES POURCENTAGES AU MILIEU
                        tickfont=dict(
                            size=11,
                            color="grey",
                            family="Arial Black, sans-serif" # Force un aspect gras
                        ),
                        gridcolor="rgba(0,0,0,0.1)"
                    ),
                    angularaxis=dict(
                        tickfont=dict(size=12),
                        rotation=90,
                        direction="clockwise",
                        gridcolor="rgba(0,0,0,0.1)"
                    ),
                    bgcolor="white"
                ),
                height=500,
                margin=dict(l=80, r=80, t=40, b=40),
                legend=dict(
                    orientation="h", 
                    yanchor="bottom", 
                    y=1.05, 
                    xanchor="center", 
                    x=0.5
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Données insuffisantes pour générer le radar.")

def get_player_all_match(df_raw):
        """
        1. Agrège les mi-temps par match (Activity Name).
        2. Calcule les statistiques (Moyenne & Max) sur les matchs complets.
        """
        # --- PARTIE 1 : AGRÉGATION DES MI-TEMPS ---
        # Calcul de VHSR effort si les colonnes sources existent
        if 'VHSR + SPR effort' in df_raw.columns and 'Sprint effort' in df_raw.columns:
            df_raw['VHSR effort'] = df_raw['VHSR + SPR effort'] - df_raw['Sprint effort']

        # Identifiants uniques d'un match
        id_cols = ['Position Name', 'Activity Name', 'Type match', 'Saison']
        
        # Logique d'agrégation
        agg_logic = {
            'Field Time': 'sum',
            'Total Distance (m)': 'sum',
            'VHSR Total Distance (m)': 'sum',
            'Sprint (m)': 'sum',
            'VHSR effort': 'sum',
            'Sprint effort': 'sum',
            'Accel >2m.s²': 'sum',
            'Decel >2m.s²': 'sum',
            'V max': 'max',            # Vitesse de pointe sur le match entier
            'Meterage Per Minute': 'mean' # Moyenne des deux mi-temps
        }
        
        # Filtrer agg_logic pour ne garder que les colonnes présentes dans le DF
        agg_logic = {k: v for k, v in agg_logic.items() if k in df_raw.columns}

        # Groupement par match
        df_matchs = df_raw.groupby(id_cols).agg(agg_logic).reset_index()

        df_matchs = df_matchs.drop(['Position Name'], axis=1)

        return df_matchs

##### Meilleur match + moyenne 60' ######
def get_player_summary_stats(df_raw, min_time=60):
        """
        1. Agrège les mi-temps par match (Activity Name).
        2. Calcule les statistiques (Moyenne & Max) sur les matchs complets.
        """
        # --- PARTIE 1 : AGRÉGATION DES MI-TEMPS ---
        # Calcul de VHSR effort si les colonnes sources existent
        if 'VHSR + SPR effort' in df_raw.columns and 'Sprint effort' in df_raw.columns:
            df_raw['VHSR effort'] = df_raw['VHSR + SPR effort'] - df_raw['Sprint effort']

        # Identifiants uniques d'un match
        id_cols = ['Position Name', 'Activity Name', 'Type match', 'Saison']
        
        # Logique d'agrégation
        agg_logic = {
            'Field Time': 'sum',
            'Total Distance (m)': 'sum',
            'VHSR Total Distance (m)': 'sum',
            'Sprint (m)': 'sum',
            'VHSR effort': 'sum',
            'Sprint effort': 'sum',
            'Accel >2m.s²': 'sum',
            'Decel >2m.s²': 'sum',
            'V max': 'max',            # Vitesse de pointe sur le match entier
            'Meterage Per Minute': 'mean' # Moyenne des deux mi-temps
        }
        
        # Filtrer agg_logic pour ne garder que les colonnes présentes dans le DF
        agg_logic = {k: v for k, v in agg_logic.items() if k in df_raw.columns}

        # Groupement par match
        df_matchs = df_raw.groupby(id_cols).agg(agg_logic).reset_index()
        
        # --- PARTIE 2 : CALCUL DES STATS DE SYNTHÈSE ---
        # Filtrage par temps de jeu minimum (ex: 60 min)
        df_filtered = df_matchs[df_matchs['Field Time'] >= min_time].copy()
        
        if df_filtered.empty:
            return None, 0

        # Sélection des colonnes numériques pour le calcul final
        numeric_cols = df_filtered.select_dtypes(include=['number']).columns.tolist()
        to_exclude = ['Saison', 'Unnamed: 0'] 
        cols_to_process = [c for c in numeric_cols if c not in to_exclude]

        # Calcul Max et Moyenne
        stats_summary = pd.DataFrame({
            "Moyenne": df_filtered[cols_to_process].mean(),
            "Maximum": df_filtered[cols_to_process].max()
        })

        return stats_summary, len(df_filtered)