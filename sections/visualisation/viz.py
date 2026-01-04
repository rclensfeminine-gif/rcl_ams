import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots


###Graphique charge
def creer_graph_dt(
    df: pd.DataFrame, 
    x_col: str, 
    y_col: str, 
    titre: str, 
    couleur_barre: str = 'gold',
    x_label: str = 'Semaine',
    y_label: str = 'Total Distance (m)'
):
    """
    Crée et affiche un graphique à barres interactif avec Plotly.
    Args:
        df (pd.DataFrame): Le DataFrame contenant les données.
        x_col (str): Le nom de la colonne pour l'axe des x (généralement une date).
        y_col (str): Le nom de la colonne pour l'axe des y (la métrique).
        titre (str): Le titre principal du graphique.
        couleur_barre (str): La couleur des barres du graphique.
        x_label (str): Le label de l'axe des x.
        y_label (str): Le label de l'axe des y.
    """
    
    # Créer le graphique avec Plotly Express
    fig = px.bar(
        df,
        x=x_col,
        y=y_col,
        title=titre,
        height=500,
        width=800
    )

    # Mettre à jour la couleur des barres
    fig.update_traces(marker_color=couleur_barre)

    # NOUVEAU : Personnaliser le survol pour afficher des nombres entiers sans 'k'
    fig.update_traces(hovertemplate=f"{x_label}: %{{x|%d-%m-%Y}}<br>{y_label}: %{{y:.0f}}")

    # Mettre à jour la mise en page
    fig.update_layout(
        xaxis_title={
            "text": f"<b>{x_label}</b>",
            "font": {"size": 16, "color": "black"}
        },
        yaxis_title={
            "text": f"<b>{y_label}</b>",
            "font": {"size": 16, "color": "black"}
        },
        xaxis=dict(
            tickvals=df[x_col],
            tickformat="%d-%m-%Y",
            tickfont=dict(
                family="Open Sans, sans-serif",
                size=12,
                color="black"
            )
        ),
        yaxis=dict(
            tickfont=dict(
                family="Open Sans, sans-serif",
                size=12,
                color="black"
            )
        )
    )

    # Afficher le graphique
    st.plotly_chart(fig)

def creer_graph_vhsr(
    df: pd.DataFrame, 
    x_col: str, 
    y_col: str, # y_col représente maintenant les barres
    line_col: str, 
    titre: str, 
    couleur_barre: str = 'blue',
    x_label: str = 'Semaine',
    y1_label: str = 'VHSR Total Distance (m)',
    y2_label: str = 'Nombre VHSR'
):
    """
    Crée et affiche un graphique combiné (barres et ligne) avec Plotly.
    
    Args:
        df (pd.DataFrame): Le DataFrame contenant les données.
        x_col (str): Le nom de la colonne pour l'axe des x.
        y_col (str): Le nom de la colonne pour les barres.
        line_col (str): Le nom de la colonne pour le graphique en ligne.
        titre (str): Le titre principal.
        couleur_barre (str): La couleur des barres.
        x_label (str): Le label de l'axe des x.
        y1_label (str): Le label de l'axe y principal.
        y2_label (str): Le label de l'axe y secondaire.
    """
    
    # Créer un objet Figure avec deux axes y
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Ajouter le graphique à barres (trace de barres)
    fig.add_trace(
        go.Bar(
            x=df[x_col],
            y=df[y_col],
            name=y1_label,
            marker_color=couleur_barre,
            hovertemplate=f"<b>{x_label}</b>: %{{x|%d-%m-%Y}}<br><b>{y1_label}</b>: %{{y:.0f}}<extra></extra>"
        ),
        secondary_y=False # Placer sur le premier axe y
    )

    # Ajouter la ligne de points (trace de scatter)
    fig.add_trace(
        go.Scatter(
            x=df[x_col],
            y=df[line_col],
            mode='lines+markers',
            name=y2_label,
            marker=dict(color='black', size=10),
            line=dict(color='black', width=3),
            hovertemplate=f"<b>{x_label}</b>: %{{x|%d-%m-%Y}}<br><b>{y2_label}</b>: %{{y:.0f}}<extra></extra>"
        ),
        secondary_y=True # Placer sur le second axe y
    )

    # Mettre à jour la mise en page
    fig.update_layout(
        title=titre,
        xaxis_title={
            "text": f"<b>{x_label}</b>",
            "font": {"size": 16, "color": "black"}
        },
        xaxis=dict(
            tickvals=df[x_col],
            tickformat="%d-%m-%Y",
            tickfont=dict(
                family="Open Sans, sans-serif",
                size=12,
                color="black"
            )
        ),
        showlegend=False
    )

    # Mettre à jour les titres des deux axes Y
    fig.update_yaxes(title_text=f"<b>{y1_label}</b>", secondary_y=False)
    fig.update_yaxes(title_text=f"<b>{y2_label}</b>", secondary_y=True)

    fig.update_yaxes(
        range=[0, None],
        secondary_y=True
    )

    # Afficher le graphique
    st.plotly_chart(fig)

def creer_graph_spr(
    df: pd.DataFrame, 
    x_col: str, 
    y_cols: list, # Maintenant une liste de colonnes
    line_col: str, # Colonne pour le nombre de sprints
    titre: str, 
    couleur_barre: dict, # Utilise un dictionnaire pour les couleurs
    x_label: str = 'Semaine',
    y1_label: str = 'Distance (m)',
    y2_label: str = 'Nombre de sprints'
):
    """
    Crée et affiche un graphique combiné (barres empilées et ligne) pour les sprints.
    
    Args:
        df (pd.DataFrame): Le DataFrame contenant les données.
        x_col (str): Le nom de la colonne pour l'axe des x (généralement une date).
        y_cols (list): La liste des colonnes pour les barres empilées.
        line_col (str): La colonne pour le graphique en ligne (ex: 'Total Sprint').
        titre (str): Le titre principal du graphique.
        couleurs (dict): Dictionnaire pour les couleurs des barres.
        x_label (str): Le label de l'axe des x.
        y1_label (str): Le label de l'axe y principal.
        y2_label (str): Le label de l'axe y secondaire.
    """
    
    # 1. Créer la figure avec un deuxième axe y
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Transformer le DataFrame en format 'long' pour les barres empilées
    df_long = pd.melt(df, id_vars=[x_col], value_vars=y_cols, var_name='Metrique', value_name='Valeur')

    # 2. Ajouter les traces de barres sur le premier axe y
    for col in y_cols:
        trace_data = df_long[df_long['Metrique'] == col]
        fig.add_trace(go.Bar(
            x=trace_data[x_col],
            y=trace_data['Valeur'],
            name=col,
            marker_color=couleur_barre.get(col, 'blue'),
            hovertemplate=f"<b>{x_label}</b>: %{{x|%d-%m-%Y}}<br><b>{col}</b>: %{{y:.0f}}<extra></extra>"
        ), secondary_y=False)
    
    # 3. Ajouter la trace de la ligne sur le deuxième axe y
    fig.add_trace(go.Scatter(
        x=df[x_col],
        y=df[line_col],
        mode='lines+markers',
        name=y2_label,
        marker=dict(color='black', size=10),
        line=dict(color='black', width=3),
        hovertemplate=f"<b>{x_label}</b>: %{{x|%d-%m-%Y}}<br><b>{y2_label}</b>: %{{y:.0f}}<extra></extra>"
    ), secondary_y=True)

    # 4. Mettre à jour la mise en page
    fig.update_layout(
        barmode='stack', # Pour empiler les barres
        title=titre,
        xaxis_title={
            "text": f"<b>{x_label}</b>",
            "font": {"size": 16, "color": "black"}
        },
        xaxis=dict(
            tickvals=df[x_col],
            tickformat="%d-%m-%Y",
            tickfont=dict(
                family="Open Sans, sans-serif",
                size=12,
                color="black"
            )
        ),
        # Mettre à jour les titres des deux axes Y
        yaxis=dict(
            title=f"<b>{y1_label}</b>",
            tickfont=dict(
                family="Open Sans, sans-serif",
                size=12,
                color="black"
            )
        ),
        yaxis2=dict(
            title=f"<b>{y2_label}</b>",
            overlaying='y',
            range=[0, None],
            side='right',
            tickfont=dict(
                family="Open Sans, sans-serif",
                size=12,
                color="black"
            )
        ),
        showlegend=False
    )

    # 5. Afficher le graphique
    st.plotly_chart(fig)

def creer_graph_accel_charge(
    df: pd.DataFrame, 
    x_col: str, 
    y_cols: list, # Maintenant une liste de colonnes
    titre: str, 
    couleur_barre: dict, # Utilise un dictionnaire pour les couleurs
    x_label: str = 'Semaine',
    y_label: str = 'Unité arbitraire'
):
    """
    Crée et affiche un graphique à barres groupées (côte à côte) pour les accélérations et décélérations.
    
    Args:
        df (pd.DataFrame): Le DataFrame contenant les données.
        x_col (str): Le nom de la colonne pour l'axe des x.
        y_cols (list): La liste des colonnes pour les barres groupées.
        titre (str): Le titre principal du graphique.
        couleur_barre (dict): Dictionnaire pour les couleurs des barres.
        x_label (str): Le label de l'axe des x.
        y_label (str): Le label de l'axe y principal.
    """

    # 1. Créer la figure vide (plus besoin de make_subplots si pas de 2ème axe Y)
    fig = go.Figure()
    
    # 2. Ajouter une trace go.Bar pour chaque colonne y_cols (Accel et Decel)
    for col in y_cols:
        fig.add_trace(go.Bar(
            x=df[x_col],
            y=df[col],
            name=col, # Utilise le nom de la colonne comme nom de la trace (pour la légende)
            marker_color=couleur_barre.get(col, 'gray'), # Applique la couleur depuis le dictionnaire
            hovertemplate=f"<b>{x_label}</b>: %{{x|%d-%m-%Y}}<br><b>{col}</b>: %{{y:.0f}}<extra></extra>"
        ))

    # 3. Mettre à jour la mise en page
    fig.update_layout(
        barmode='group', # Barres côte à côte
        title=titre,
        xaxis_title={
            "text": f"<b>{x_label}</b>",
            "font": {"size": 16, "color": "black"}
        },
        yaxis_title={
            "text": f"<b>{y_label}</b>",
            "font": {"size": 16, "color": "black"}
        },
        xaxis=dict(
            tickvals=df[x_col],
            tickformat="%d-%m-%Y",
            tickfont=dict(
                family="Open Sans, sans-serif",
                size=12,
                color="black"
            )
        ),
        yaxis=dict(
            tickfont=dict(
                family="Open Sans, sans-serif",
                size=12,
                color="black"
            )
        ),
        showlegend=False
    )

    # 4. Afficher le graphique
    st.plotly_chart(fig)

###Graphique match
def jauge_distance(
    valeur_match: float,
    valeur_moyenne: float,
    valeur_max: float,
    couleur_barre: str,
    titre: str
):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        
        value = valeur_match,
        
        domain = {'x': [0, 1], 'y': [0, 1]},
        
        title = {'text': titre, 'font': {'size': 24}},
        
        delta = {'reference': valeur_max, 'increasing': {'color': "RebeccaPurple"}},
        
        gauge = {
            'axis': {'range': [valeur_max*0.8, valeur_max*1.02], 'tickwidth': 2, 'tickcolor': "gray"},
            'bar': {'color': couleur_barre},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [valeur_moyenne*0.98, valeur_moyenne*1.02], 'color': 'silver'},
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': valeur_max}}))

    fig.update_layout(font = {'color': "black", 'family': "Arial"})

    fig.update_layout(
        height=280, # Hauteur du graphique
        width=250,  # Largeur, essayez d'augmenter cette valeur
        margin={'t': 50, 'b': 20, 'l': 0, 'r': 0}
    )

    st.plotly_chart(fig, use_container_width=True)

def jauge_intensite(
    valeur_match: float,
    valeur_moyenne: float,
    valeur_max: float,
    couleur_barre: str,
    titre: str
):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        
        value = valeur_match,
        
        domain = {'x': [0, 1], 'y': [0, 1]},
        
        title = {'text': titre, 'font': {'size': 24}},
        
        delta = {'reference': valeur_max, 'increasing': {'color': "RebeccaPurple"}},

        gauge = {
            'axis': {'range': [valeur_max*0.5, valeur_max*1.02], 'tickwidth': 2, 'tickcolor': "gray"},
            'bar': {'color': couleur_barre},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [valeur_moyenne*0.98, valeur_moyenne*1.02], 'color': 'silver'},
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': valeur_max}}))

    fig.update_layout(
        font = {'color': "black", 'family': "Arial"},
        height=280, # Hauteur du graphique
        width=250,  # Largeur, essayez d'augmenter cette valeur
        margin={'t': 50, 'b': 20, 'l': 0, 'r': 0}
    )

    st.plotly_chart(fig)

def jauge_nbr(
    valeur_match: float,
    valeur_moyenne: float,
    valeur_max: float,
    couleur_barre: str,
    titre: str
):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        
        value = valeur_match,
        
        domain = {'x': [0, 1], 'y': [0, 1]},
        
        title = {'text': titre, 'font': {'size': 24}},
        
        delta = {'reference': valeur_max, 'increasing': {'color': "RebeccaPurple"}},

        gauge = {
            'axis': {'range': [valeur_max*0.5, valeur_max*1.02], 'tickwidth': 2, 'tickcolor': "gray"},
            'bar': {'color': couleur_barre},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [valeur_moyenne*0.98, valeur_moyenne*1.02], 'color': 'silver'},
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': valeur_max}}))

    fig.update_layout(
        font = {'color': "black", 'family': "Arial"},
        height=165, # Hauteur du graphique
        width=250,  # Largeur, essayez d'augmenter cette valeur
        margin={'t': 50, 'b': 20, 'l': 0, 'r': 0}
    )

    st.plotly_chart(fig)

def jauge_barre(
    valeur_match: float,
    valeur_moyenne: float,
    valeur_max: float,
    couleur_barre: str,
    titre: str
):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        
        value = valeur_match,
        
        domain = {'x': [0, 1], 'y': [0, 1]},
        
        title = {'text': titre, 'font': {'size': 20}},
        
        delta = {'reference': valeur_max, 'increasing': {'color': "RebeccaPurple"}},

        gauge = {
            'shape': "bullet",
            'axis': {'range': [valeur_max*0.9, valeur_max*1.02], 'tickwidth': 2, 'tickcolor': "gray"},
            'bar': {'color': couleur_barre},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [valeur_moyenne*0.99, valeur_moyenne*1.01], 'color': 'silver'},
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': valeur_max}}))

    fig.update_layout(
        font = {'color': "black", 'family': "Arial"},
        height=110, # Hauteur du graphique
        width=250,  # Largeur, essayez d'augmenter cette valeur
        margin={'t': 50, 'b': 20, 'l': 0, 'r': 0}
    )

    st.plotly_chart(fig)


##########################
# Graph fiche joueuse
##########################
def creer_graph_poids(
    df: pd.DataFrame, 
    x_col: str, 
    y_col: str, 
    titre: str, 
    x_label: str = 'Date mesure',
    y_label: str = 'Poids (kg)', 
    remarque_col: str = 'Remarque'
):
    """
    Crée et affiche un graphique d'évolution du Poids avec Plotly.
    Les étiquettes de données s'adaptent à la densité des points :
    1. Le décalage vertical s'ajuste pour éloigner les étiquettes de la ligne.
    2. Les étiquettes sont filtrées (une sur N) si la densité est trop élevée.
    
    Args:
        df (pd.DataFrame): Le DataFrame contenant les données.
        x_col (str): Le nom de la colonne pour l'axe des x (Date).
        y_col (str): Le nom de la colonne pour l'axe des y (Poids).
        titre (str): Le titre principal du graphique.
        x_label (str): Le label de l'axe des x.
        y_label (str): Le label de l'axe des y.
        remarque_col (str): Le nom de la colonne contenant les remarques pour le survol.
    """
    
    if df.empty:
        st.info(f"Aucune donnée de {y_col} disponible pour le graphique.")
        return
        
    # S'assurer que la colonne de date est de type datetime
    df[x_col] = pd.to_datetime(df[x_col], errors='coerce')
    
    # Préparation des données personnalisées pour le survol
    if remarque_col in df.columns:
        customdata = df[[remarque_col]].values
        hover_remarque_template = f"<br><b>Remarque</b>: %{{customdata[0]}}"
    else:
        customdata = None
        hover_remarque_template = ""

    # 1. Calcul des métriques pour l'adaptation
    y_min = df[y_col].min()
    y_max = df[y_col].max()
    y_range_amplitude = y_max - y_min
    n_points = len(df)
    
    # --- DÉCALAGE VERTICAL ADAPTATIF (offset_y) ---
    
    # Facteur de base (proportion de l'amplitude verticale pour le décalage)
    base_offset_factor = 0.05 
    
    if n_points <= 10:
        offset_multiplier = 1.0 # Décalage standard pour peu de points
    else:
        # Augmentation du multiplicateur au-delà de 10 points (logarithmique pour douceur)
        offset_multiplier = 1.0 + (np.log(n_points) / np.log(10) - 1.0) 
        offset_multiplier = max(1.0, offset_multiplier)

    offset_y = y_range_amplitude * base_offset_factor * offset_multiplier
    
    # S'assurer que le décalage est raisonnable
    offset_y = np.clip(offset_y, 0.2, 0.7) 
    
    # Calcul des nouvelles coordonnées Y décalées pour le texte
    y_shifted = df[y_col] + offset_y

    # --- FILTRAGE ADAPTATIF DES ÉTIQUETTES (texte_etiquettes) ---
    seuil_filtration = 15 # Début de la filtration à partir de 15 points
    
    if n_points <= seuil_filtration:
        # Afficher toutes les étiquettes
        intervalle = 1
    else:
        # Calcul de l'intervalle k pour afficher 1 étiquette sur k points
        # L'intervalle augmente avec la densité
        intervalle = max(2, int(n_points / seuil_filtration)) 
    
    # Génération des étiquettes à afficher/masquer
    texte_etiquettes = []
    for i, poids in enumerate(df[y_col]):
        if i % intervalle == 0:
            # Afficher l'étiquette pour les points sélectionnés
            texte_etiquettes.append(f"<b>{poids:.1f}<b> kg")
        else:
            # Ne pas afficher l'étiquette (chaîne vide)
            texte_etiquettes.append("")
            
    # 2. Créer la figure
    fig = go.Figure()

    # Trace 1: Ligne et points
    fig.add_trace(
        go.Scatter(
            x=df[x_col],
            y=df[y_col],
            mode='lines+markers',
            name=y_label,
            customdata=customdata,
            marker=dict(color='black', size=8, symbol='circle'), 
            line=dict(color='green', width=3),
            hovertemplate=(
                f"<b>{x_label}</b>: %{{x|%d-%m-%Y}}<br>"
                f"<b>{y_label}</b>: %{{y:.1f}} kg"
                f"{hover_remarque_template}<extra></extra>"
            )
        )
    )

    # 3. Trace 2: Étiquettes de texte
    fig.add_trace(
        go.Scatter(
            x=df[x_col],
            y=y_shifted,
            mode='text',
            text=texte_etiquettes, # Utilisation de la liste filtrée
            textposition="middle center", 
            textfont=dict(
                size=12,
                color='black'
            ),
            showlegend=False,
            hoverinfo='none' 
        )
    )

    # 4. Calcul de l'ajustement final de l'axe Y 
    # Le padding doit être au moins égal au décalage calculé
    y_range_top_padding = max(offset_y, y_range_amplitude * 0.30) 

    y_axis_range = [y_min - y_range_amplitude * 0.1, y_max + y_range_top_padding] 

    # 5. Mettre à jour la mise en page (Layout)
    fig.update_layout(
        title={
            'text': titre,
            'font': {'size': 20, 'color': '#333'}
        },
        xaxis_title={
            "text": f"<b>{x_label}</b>",
            "font": {"size": 14, "color": "black"}
        },
        yaxis_title={
            "text": f"<b>{y_label}</b>",
            "font": {"size": 14, "color": "black"}
        },
        yaxis_range=y_axis_range, 
        
        # Configuration de l'axe X pour afficher correctement les dates
        xaxis=dict(
            tickvals=df[x_col],
            tickformat="%d/%m/%Y", 
            tickangle=-45,
            showgrid=True,
            gridcolor='#e6e6e6',
            tickfont=dict(
                family="Arial, sans-serif", 
                size=12,
                color="black",
            )
        ),
        # Configuration de l'axe Y
        yaxis=dict(
            zeroline=False,
            showgrid=True,
            gridcolor='#e6e6e6'
        ),
        template="plotly_white", 
        showlegend=False,
        height=400
    )
    
    # 6. Afficher le graphique
    st.plotly_chart(fig, use_container_width=True)


def creer_graph_poids_blessure(
    df_poids: pd.DataFrame, 
    df_blessure: pd.DataFrame, # DataFrame des blessures (correspond à votre df_blessure_j)
    x_col: str, # Doit être 'Date' pour le poids
    y_col: str, 
    titre: str, 
    x_label: str = 'Date mesure',
    y_label: str = 'Poids (kg)',
    remarque_col: str = 'Remarque'
):
    """
    Crée et affiche un graphique d'évolution du Poids (ligne et points) pour une seule joueuse.
    Superpose les événements de blessure musculaire (df_blessure) comme des points rouges.
    
    Le décalage vertical et la filtration des étiquettes de poids sont désormais ADAPTATIFS
    en fonction du nombre de points (densité) de poids.

    Args:
        df_poids (pd.DataFrame): Le DataFrame contenant les données de Poids. x_col doit être 'Date'.
        df_blessure (pd.DataFrame): Le DataFrame contenant les données de Blessure. 
                                     Doit contenir la colonne 'Date blessure' et 'Type Blessure'.
        x_col (str): Le nom de la colonne pour l'axe des x du Poids (Date).
        y_col (str): Le nom de la colonne pour l'axe des y (Poids).
        titre (str): Le titre principal du graphique.
        x_label (str): Le label de l'axe des x.
        y_label (str): Le label de l'axe des y.
        remarque_col (str): Le nom de la colonne contenant les remarques pour le poids.
    """
    
    # 1. Préparation des données de Poids
    if df_poids.empty:
        st.info(f"Aucune donnée de {y_col} disponible pour le graphique.")
        return
    
    # S'assurer que la colonne de date du poids est de type datetime
    df_poids[x_col] = pd.to_datetime(df_poids[x_col], errors='coerce')
    y_min_poids = df_poids[y_col].min()
    y_max_poids = df_poids[y_col].max() 
    n_points = len(df_poids) # Nombre de points de poids
    
    # Préparation des données personnalisées pour le survol du poids
    if remarque_col in df_poids.columns:
        customdata_poids = df_poids[[remarque_col]].values
        hover_remarque_template = f"<br><b>Remarque</b>: %{{customdata[0]}}"
    else:
        customdata_poids = None
        hover_remarque_template = ""

    # --- CALCULS ADAPTATIFS DE DÉCALAGE ET FILTRAGE POUR LES ÉTIQUETTES DE POIDS ---
    
    y_range_amplitude = y_max_poids - y_min_poids
    
    # 1. DÉCALAGE VERTICAL ADAPTATIF (offset_y)
    base_offset_factor = 0.05 
    
    if n_points <= 10:
        offset_multiplier = 1.0 
    else:
        # Augmentation du multiplicateur basée sur la densité
        offset_multiplier = 1.0 + (np.log(n_points) / np.log(10) - 1.0) 
        offset_multiplier = max(1.0, offset_multiplier)

    offset_y = y_range_amplitude * base_offset_factor * offset_multiplier
    offset_y = np.clip(offset_y, 0.1, 0.5) # Limiter le décalage entre 0.1 et 0.5 kg
    
    # Calcul des nouvelles coordonnées Y décalées pour le texte
    y_shifted = df_poids[y_col] + offset_y

    # 2. FILTRAGE ADAPTATIF DES ÉTIQUETTES
    seuil_filtration = 15 
    
    if n_points <= seuil_filtration:
        intervalle = 1 # Afficher toutes les étiquettes
    else:
        # Augmentation de l'intervalle avec la densité
        intervalle = max(2, int(n_points / seuil_filtration)) 
    
    # Génération des étiquettes à afficher/masquer
    texte_etiquettes = []
    for i, poids in enumerate(df_poids[y_col]):
        if i % intervalle == 0:
            texte_etiquettes.append(f"<b>{poids:.1f}<b> kg")
        else:
            texte_etiquettes.append("") # Masquer

    # 3. Préparation des données de Blessure
    df_blessure_musculaire = pd.DataFrame()
    blessure_date_col = 'Date blessure' 
    blessure_type_col = 'Type Blessure'
    y_min_graph = y_min_poids 
    message_blessure_warning = ""

    if not df_blessure.empty and blessure_date_col in df_blessure.columns and blessure_type_col in df_blessure.columns:
        df_blessure[blessure_date_col] = pd.to_datetime(df_blessure[blessure_date_col], errors='coerce')
        
        # Filtrer uniquement les blessures musculaires (contenant 'MUSC')
        df_blessure_musculaire = df_blessure[
            (df_blessure[blessure_type_col].str.contains('MUSC', case=False, na=False))
        ].copy() 
        
        if not df_blessure_musculaire.empty:
            # Position Y fixe pour les marqueurs de blessure (5% sous le poids min)
            y_range_poids = y_max_poids - y_min_poids
            y_fixed_blessure = y_min_poids - y_range_poids * 0.05 

            df_blessure_musculaire['y_plot'] = y_fixed_blessure
            y_min_graph = min(y_min_poids, y_fixed_blessure) # Définit le min du graphique
            
        else:
            message_blessure_warning = "Aucune blessure musculaire (contenant 'MUSC') trouvée dans les données filtrées."
            
    else:
        message_blessure_warning = f"Le DataFrame des blessures n'est pas utilisé car il est vide ou ne contient pas les colonnes '{blessure_date_col}' et '{blessure_type_col}' nécessaires."
    
    if message_blessure_warning:
        st.warning(message_blessure_warning)


    # 4. Créer la figure
    fig = go.Figure()

    # Trace 1: Ligne et points du Poids (vert/noir)
    fig.add_trace(
        go.Scatter(
            x=df_poids[x_col],
            y=df_poids[y_col],
            mode='lines+markers',
            name=y_label,
            customdata=customdata_poids,
            marker=dict(color='black', size=8, symbol='circle'),
            line=dict(color='green', width=3),
            hovertemplate=(
                f"<b>{x_label}</b>: %{{x|%d-%m-%Y}}<br>"
                f"<b>{y_label}</b>: %{{y:.1f}} kg"
                f"{hover_remarque_template}<extra></extra>"
            )
        )
    )

    # Trace 2: Événements de Blessure Musculaire (Rouge 'X')
    if not df_blessure_musculaire.empty:
        customdata_blessure = df_blessure_musculaire[[blessure_type_col]].values

        fig.add_trace(
            go.Scatter(
                x=df_blessure_musculaire[blessure_date_col], 
                y=df_blessure_musculaire['y_plot'],
                mode='markers',
                name='Blessure Musculaire',
                customdata=customdata_blessure,
                marker=dict(
                    color='red', 
                    size=12, 
                    symbol='x-thin-open', 
                    line=dict(width=2)
                ),
                hovertemplate=(
                    f"<b>Date Blessure</b>: %{{x|%d-%m-%Y}}<br>"
                    f"<b>Blessure</b>: %{{customdata[0]}}"
                    f"<extra></extra>"
                )
            )
        )

    # Trace 3: Étiquettes de texte pour le Poids (avec filtration et décalage adaptatifs)
    fig.add_trace(
        go.Scatter(
            x=df_poids[x_col],
            y=y_shifted,
            mode='text',
            text=texte_etiquettes, # Utilisation de la liste filtrée et décalée
            textposition="middle center",
            textfont=dict(
                size=12,
                color='black'
            ),
            showlegend=False,
            hoverinfo='none' 
        )
    )

    # 5. Calcul de l'ajustement de l'axe Y 
    y_range_data = y_max_poids - y_min_graph 
    
    # Le padding supérieur doit au moins inclure le décalage adaptatif
    y_range_top_padding = max(offset_y, y_range_data * 0.5) 
    # Padding inférieur pour inclure les blessures
    y_range_bottom_padding = y_max_poids - y_min_graph + y_range_data * 0.1 

    y_axis_range = [y_min_graph - y_range_bottom_padding * 0.1, y_max_poids + y_range_top_padding] 

    # 6. Mettre à jour la mise en page (Layout)
    fig.update_layout(
        title={
            'text': titre,
            'font': {'size': 20, 'color': '#333'}
        },
        xaxis_title={
            "text": f"<b>{x_label}</b>",
            "font": {"size": 14, "color": "black"}
        },
        yaxis_title={
            "text": f"<b>{y_label}</b>",
            "font": {"size": 14, "color": "black"}
        },
        yaxis_range=y_axis_range, 
        xaxis=dict(
            tickvals=df_poids[x_col],
            tickformat="%d/%m/%Y", 
            tickangle=-45,
            showgrid=True,
            gridcolor='#e6e6e6',
            tickfont=dict(
                family="Arial, sans-serif", 
                size=12,
                color="black",
            )
        ),
        yaxis=dict(
            zeroline=False,
            showgrid=True,
            gridcolor='#e6e6e6'
        ),
        template="plotly_white", 
        showlegend=True, 
        # --- NOUVELLE CONFIGURATION POUR LA LÉGENDE (EN BAS ET CENTRÉE) ---
        legend=dict(
            orientation="h",       # Horizontale
            yanchor="top",         # Ancrer en haut de la boîte de légende
            y=-0.45,                # Placer sous le graphique (valeur ajustée)
            xanchor="center",      # Ancrer au centre de la boîte de légende
            x=0.5,                 # Centrer horizontalement
            bgcolor="rgba(255, 255, 255, 0)", 
            bordercolor="rgba(0, 0, 0, 0)",
            borderwidth=0
        ),
        height=400 # Garder la hauteur standard
    )
    
    # 7. Afficher le graphique
    st.plotly_chart(fig, use_container_width=True)

def creer_MM(
    df: pd.DataFrame, 
    x_col: str, 
    y_col: str, 
    titre: str, 
    x_label: str = 'Date mesure',
    y_label: str = 'Masse musculaire',
    y_range_min: float = None # NOUVEAU PARAMÈTRE : Limite basse de l'axe Y
):
    """
    Crée et affiche un graphique d'évolution (nuage de points reliés) pour une seule série de données.
    
    Le décalage vertical et la filtration des étiquettes sont désormais ADAPTATIFS
    en fonction du nombre de points (densité).
    
    Args:
        df (pd.DataFrame): Le DataFrame contenant les données.
        x_col (str): Le nom de la colonne pour l'axe des x (Date).
        y_col (str): Le nom de la colonne pour l'axe des y (Poids ou % MG).
        titre (str): Le titre principal du graphique.
        x_label (str): Le label de l'axe des x.
        y_label (str): Le label de l'axe des y.
        y_range_min (float, optional): La valeur minimale pour l'axe Y. 
                                        Si None (défaut), Plotly choisit. 
                                        Utilisé pour éviter la distorsion des petites variations.
    """
    
    if df.empty:
        st.info(f"Aucune donnée de {y_col} disponible pour le graphique.")
        return

    # S'assurer que la colonne de date est de type datetime
    df[x_col] = pd.to_datetime(df[x_col], errors='coerce')
    
    # 1. Calcul des métriques pour l'adaptation
    y_min = df[y_col].min()
    y_max = df[y_col].max()
    y_range_amplitude = y_max - y_min
    n_points = len(df)
    
    # --- DÉCALAGE VERTICAL ADAPTATIF (offset_y) ---
    
    # Facteur de base (proportion de l'amplitude verticale pour le décalage)
    base_offset_factor = 0.05 
    
    if n_points <= 10:
        offset_multiplier = 1.0 # Décalage standard pour peu de points
    else:
        # Augmentation du multiplicateur au-delà de 10 points (logarithmique pour douceur)
        offset_multiplier = 1.0 + (np.log(n_points) / np.log(10) - 1.0) 
        offset_multiplier = max(1.0, offset_multiplier)

    offset_y = y_range_amplitude * base_offset_factor * offset_multiplier
    
    # S'assurer que le décalage est raisonnable
    # Utiliser np.clip pour fixer une plage de valeurs, par exemple entre 0.05 et 0.3
    offset_y = np.clip(offset_y, 0.2, 1) 
    
    # Calcul des nouvelles coordonnées Y décalées pour le texte
    y_shifted = df[y_col] + offset_y

    # --- FILTRAGE ADAPTATIF DES ÉTIQUETTES (texte_etiquettes) ---
    seuil_filtration = 15 # Début de la filtration à partir de 15 points
    
    if n_points <= seuil_filtration:
        # Afficher toutes les étiquettes
        intervalle = 1
    else:
        # Calcul de l'intervalle k pour afficher 1 étiquette sur k points
        intervalle = max(2, int(n_points / seuil_filtration)) 
    
    # Génération des étiquettes à afficher/masquer
    texte_etiquettes = []
    for i, valeur in enumerate(df[y_col]):
        if i % intervalle == 0:
            # Afficher l'étiquette pour les points sélectionnés
            texte_etiquettes.append(f"<b>{valeur:.1f}<b> kg")
        else:
            # Ne pas afficher l'étiquette (chaîne vide)
            texte_etiquettes.append("")
            
    # 2. Créer la figure
    fig = go.Figure()
    
    # Couleur pour la Masse Musculaire
    line_color = "#B40C0C"

    # Trace 1: Ligne et points
    fig.add_trace(
        go.Scatter(
            x=df[x_col],
            y=df[y_col],
            mode='lines+markers',
            name=y_label,
            marker=dict(color='black', size=8, symbol='circle'), # Style des points
            line=dict(color=line_color, width=3), # Couleur de la ligne
            # Formatage du survol (hover)
            hovertemplate=f"<b>{x_label}</b>: %{{x|%d-%m-%Y}}<br><b>{y_label}</b>: %{{y:.1f}} kg<extra></extra>"
        )
    )

    # 3. Trace 2: Étiquettes de texte (les valeurs de poids)
    fig.add_trace(
        go.Scatter(
            x=df[x_col],
            y=y_shifted,
            mode='text',
            text=texte_etiquettes, # Utilisation de la liste filtrée et décalée
            textposition="middle center", 
            textfont=dict(
                size=12,
                color='black'
            ),
            showlegend=False,
            hoverinfo='none' 
        )
    )

    # 4. Calcul de l'ajustement de l'axe Y pour faire de la place pour les étiquettes
    
    # Le padding supérieur doit au moins être égal au décalage maximal utilisé
    y_range_top_padding = max(offset_y, y_range_amplitude * 0.5) 
    
    # Définition de la limite haute
    y_max_final = y_max + y_range_top_padding
    
    # Définition de la limite basse
    if y_range_min is not None:
        y_min_final = y_range_min
    else:
        # Marge légère en bas si aucune valeur min n'est imposée
        y_min_final = y_min - y_range_amplitude * 0.05 
        
    y_axis_range = [y_min_final, y_max_final] 

    # --- NOUVELLE LOGIQUE POUR L'AXE Y (ajustée) ---
    yaxis_update = dict(
        zeroline=False,
        showgrid=True,
        gridcolor='#e6e6e6'
    )
    yaxis_update['range'] = y_axis_range
    
    if y_range_min is not None and y_range_min == 0:
         yaxis_update['zeroline'] = True

    # 5. Mettre à jour la mise en page (Layout)
    fig.update_layout(
        title={
            'text': titre,
            'font': {'size': 20, 'color': '#333'}
        },
        xaxis_title={
            "text": f"<b>{x_label}</b>",
            "font": {"size": 14, "color": "black"}
        },
        yaxis_title={
            "text": f"<b>{y_label}</b>",
            "font": {"size": 14, "color": "black"}
        },
        
        # Configuration de l'axe X pour afficher correctement les dates
        xaxis=dict(
            tickvals=df[x_col],
            tickformat="%d/%m/%Y", # Format des étiquettes de date
            tickangle=-45,
            showgrid=True,
            gridcolor='#e6e6e6',
            tickfont=dict(
                family="Arial, sans-serif", 
                size=12,
                color="black",
            )
        ),
        # Utilisation des mises à jour de l'axe Y (avec le range adapté)
        yaxis=yaxis_update,
        template="plotly_white", # Thème plus propre
        showlegend=False,
        height=400
    )
    
    # 6. Afficher le graphique
    st.plotly_chart(fig, use_container_width=True)

def creer_percent_MG(
    df: pd.DataFrame, 
    x_col: str, 
    y_col: str, 
    titre: str, 
    x_label: str = 'Date mesure',
    y_label: str = '% Masse grasse',
    y_range_min: float = None # Limite basse de l'axe Y
):
    """
    Crée et affiche un graphique d'évolution (nuage de points reliés) pour une seule série de données.
    
    Le décalage vertical et la filtration des étiquettes sont désormais ADAPTATIFS
    en fonction du nombre de points (densité).
    
    Args:
        df (pd.DataFrame): Le DataFrame contenant les données.
        x_col (str): Le nom de la colonne pour l'axe des x (Date).
        y_col (str): Le nom de la colonne pour l'axe des y (Poids ou % MG).
        titre (str): Le titre principal du graphique.
        x_label (str): Le label de l'axe des x.
        y_label (str): Le label de l'axe des y.
        y_range_min (float, optional): La valeur minimale pour l'axe Y. 
                                        Si None (défaut), Plotly choisit. 
    """
    
    if df.empty:
        st.info(f"Aucune donnée de {y_col} disponible pour le graphique.")
        return

    # S'assurer que la colonne de date est de type datetime
    df[x_col] = pd.to_datetime(df[x_col], errors='coerce')
    
    # 1. Calcul des métriques pour l'adaptation
    y_min = df[y_col].min()
    y_max = df[y_col].max()
    y_range_amplitude = y_max - y_min
    n_points = len(df)
    
    # --- DÉCALAGE VERTICAL ADAPTATIF (offset_y) ---
    
    # Facteur de base pour le décalage (doit être plus petit pour les pourcentages)
    base_offset_factor = 0.02 
    
    if n_points <= 10:
        offset_multiplier = 1.0 
    else:
        # Augmentation du multiplicateur au-delà de 10 points
        offset_multiplier = 1.0 + (np.log(n_points) / np.log(10) - 1.0) 
        offset_multiplier = max(1.0, offset_multiplier)

    offset_y = y_range_amplitude * base_offset_factor * offset_multiplier
    
    # Limiter le décalage pour qu'il soit raisonnable pour les pourcentages (ex: entre 0.1 et 0.5%)
    offset_y = np.clip(offset_y, 0.5, 2) 
    
    # Calcul des nouvelles coordonnées Y décalées pour le texte
    y_shifted = df[y_col] + offset_y

    # --- FILTRAGE ADAPTATIF DES ÉTIQUETTES (texte_etiquettes) ---
    seuil_filtration = 15 
    
    if n_points <= seuil_filtration:
        intervalle = 1
    else:
        # Calcul de l'intervalle k
        intervalle = max(2, int(n_points / seuil_filtration)) 
    
    # Génération des étiquettes à afficher/masquer
    texte_etiquettes = []
    for i, valeur in enumerate(df[y_col]):
        if i % intervalle == 0:
            # Formatage avec 1 décimale + "%"
            texte_etiquettes.append(f"<b>{valeur:.1f}<b> %")
        else:
            texte_etiquettes.append("")
            
    # 2. Créer la figure
    fig = go.Figure()
    
    # Couleur pour le % Masse Grasse
    line_color = '#6A0DAD'

    # Trace 1: Ligne et points
    fig.add_trace(
        go.Scatter(
            x=df[x_col],
            y=df[y_col],
            mode='lines+markers',
            name=y_label,
            marker=dict(color='black', size=8, symbol='circle'), 
            line=dict(color=line_color, width=3), 
            # Formatage du survol (hover)
            hovertemplate=f"<b>{x_label}</b>: %{{x|%d-%m-%Y}}<br><b>{y_label}</b>: %{{y:.1f}} %<extra></extra>"
        )
    )

    # 3. Trace 2: Étiquettes de texte
    fig.add_trace(
        go.Scatter(
            x=df[x_col],
            y=y_shifted,
            mode='text',
            text=texte_etiquettes, # Utilisation de la liste filtrée et décalée
            textposition="middle center", 
            textfont=dict(
                size=12,
                color='black'
            ),
            showlegend=False,
            hoverinfo='none' 
        )
    )

    # 4. Calcul de l'ajustement de l'axe Y pour faire de la place pour les étiquettes
    
    # Le padding supérieur doit au moins être égal au décalage maximal utilisé
    y_range_top_padding = max(offset_y, y_range_amplitude * 1) 
    
    # Définition de la limite haute
    y_max_final = y_max + y_range_top_padding
    
    # Définition de la limite basse
    if y_range_min is not None:
        y_min_final = y_range_min
    else:
        # Marge légère en bas si aucune valeur min n'est imposée
        y_min_final = y_min - y_range_amplitude * 0.05 
        
    y_axis_range = [y_min_final, y_max_final] 

    # --- Configuration de l'Axe Y ---
    yaxis_update = dict(
        zeroline=False,
        showgrid=True,
        gridcolor='#e6e6e6',
        range=y_axis_range # Application du range adapté
    )
    
    if y_range_min is not None and y_range_min == 0:
         yaxis_update['zeroline'] = True

    # 5. Mettre à jour la mise en page (Layout)
    fig.update_layout(
        title={
            'text': titre,
            'font': {'size': 20, 'color': '#333'}
        },
        xaxis_title={
            "text": f"<b>{x_label}</b>",
            "font": {"size": 14, "color": "black"}
        },
        yaxis_title={
            "text": f"<b>{y_label}</b>",
            "font": {"size": 14, "color": "black"}
        },
        
        # Configuration de l'axe X pour afficher correctement les dates
        xaxis=dict(
            tickvals=df[x_col],
            tickformat="%d/%m/%Y", 
            tickangle=-45,
            showgrid=True,
            gridcolor='#e6e6e6',
            tickfont=dict(
                family="Arial, sans-serif", 
                size=12,
                color="black",
            )
        ),
        yaxis=yaxis_update,
        template="plotly_white", 
        showlegend=False,
        height=400
    )
    
    # 6. Afficher le graphique
    st.plotly_chart(fig, use_container_width=True)

def creer_graph_pli(
    df: pd.DataFrame, 
    x_col: str, 
    y_col: str, 
    titre: str, 
    couleur_barre: str = 'gold',
    x_label: str = 'Date mesure',
    y_label: str = 'Pli (mm)'
):
    """
    Crée et affiche un graphique à barres interactif avec Plotly.
    Affiche la valeur Y au-dessus de la barre, en gras.
    """
    
    # Créer le graphique avec Plotly Express
    fig = px.bar(
        df,
        x=df[x_col],
        y=y_col,
        title=titre,
        height=500,
        width=800
    )

    # Mettre à jour la couleur des barres
    fig.update_traces(
        marker_color=couleur_barre
        )

    # CRUCIAL: Mettre à jour le texte pour l'afficher en gras (<b>) au-dessus (outside) et sans décimale (:.0f)
    fig.data[0].update(
        texttemplate='<b>%{y:.0f}</b>', 
        textposition='outside',
        hovertemplate=f"{x_label}: %{{x|%d-%m-%Y}}<br>{y_label}: %{{y:.0f}}",
        # Utiliser le dictionnaire textfont pour forcer la couleur à un noir pur (#000000)
        textfont=dict(color='#000000', size=12) # J'ai augmenté la taille pour mieux contraster
    )

    # Mettre à jour la mise en page
    fig.update_layout(
        xaxis_title={
            "text": f"<b>{x_label}</b>",
            "font": {"size": 16, "color": "black"}
        },
        yaxis_title={
            "text": f"<b>{y_label}</b>",
            "font": {"size": 16, "color": "black"}
        },
        xaxis=dict(
            tickvals=df[x_col],
            tickformat="%d-%m-%Y",
            tickfont=dict(
                family="Open Sans, sans-serif",
                size=12,
                color="black"
            )
        ),
        yaxis=dict(
            tickfont=dict(
                family="Open Sans, sans-serif",
                size=12,
                color="black"
            )
        ),
        # Pour forcer le texte des valeurs à s'adapter
        uniformtext_minsize=8,
        uniformtext_mode='hide' 
    )

    # Afficher le graphique
    st.plotly_chart(fig)

def create_sauts_performance(df, y_cols_primary, title, y_cols_secondary=None, 
                        yaxis_primary_label="Performance", yaxis_secondary_label=""): 
        """
        Crée un nuage de points relié (line chart) en forçant l'axe X à n'afficher 
        que les dates de test réelles (type 'category').
        """
        
        all_y_cols = y_cols_primary + (y_cols_secondary if y_cols_secondary else [])
        
        # Vérifier que toutes les colonnes requises existent
        missing_cols = [col for col in all_y_cols if col not in df.columns]
        if missing_cols:
            st.warning(f"Impossible de créer le graphique '{title}'. Colonnes manquantes : {', '.join(missing_cols)}")
            return None

        # Créer la colonne de date formatée comme une chaîne de caractères (pour l'axe X)
        # C'est ce qui force l'axe X à traiter les dates comme des catégories discrètes.
        df['Date Label'] = df['Date Test'].dt.strftime('%d %b %Y')

        plot_df = df[['Date Label', 'Date Test'] + all_y_cols].melt(
            id_vars=['Date Label', 'Date Test'], # Conserver les deux colonnes de date pour le tri et l'affichage
            value_vars=all_y_cols,
            var_name='Métrique',
            value_name='Performance'
        ).dropna(subset=['Performance']) 
        
        if plot_df.empty:
            st.info(f"Aucune donnée valide trouvée pour le graphique : {title}")
            return None

        # Triage de la DataFrame par la date réelle (pour assurer une ligne correcte)
        plot_df = plot_df.sort_values(by='Date Test').reset_index(drop=True)
        
        # Création du graphique. Utiliser 'Date Label' (la chaîne de caractères) pour l'axe X
        fig = px.line(
            plot_df, 
            x='Date Label', # Utilise la chaîne de date pour l'affichage de l'axe
            y='Performance',
            color='Métrique',
            title=title,
            markers=True,
            line_shape='linear',
            template="plotly_white",
            # Conserver les données originales pour l'infobulle
            custom_data=[plot_df['Date Label'], plot_df['Métrique'], plot_df['Performance']] 
        )

        # -----------------------------------------------------------
        # CONFIGURATION DES AXES X ET Y 
        # -----------------------------------------------------------
        
        # Définir le type de l'axe X comme 'category' pour n'afficher que les points existants
        fig.update_xaxes(
            title_text="", # Rétablit le titre
            showgrid=True,
            tickangle=-45, 
            type='category', # C'est le changement clé
            # Assure que l'ordre est correct
            categoryorder='array',
            categoryarray=plot_df['Date Label'].unique() 
        )
        
        # Si y_cols_secondary est fourni, configurer le second axe Y
        if y_cols_secondary:
            for trace in fig.data:
                if trace.name in y_cols_secondary:
                    trace.update(yaxis="y2")
            
            fig.update_layout(
                yaxis2=dict(
                    title=yaxis_secondary_label,
                    overlaying='y', 
                    side='right', 
                    showgrid=False
                )
            )

        fig.update_layout(
            yaxis_title=yaxis_primary_label, 
            legend_title="",
            hovermode="x unified",
            margin=dict(t=50, l=10, r=10, b=100),
            legend=dict(
                orientation="h", 
                yanchor="top",
                y=-0.3, 
                xanchor="center",
                x=0.5 
            )
        )
        
        # Mise à jour du template d'infobulle (utilise les dates non-décalées)
        fig.update_traces(
            hovertemplate=(
                "<b>Date Test:</b> %{customdata[0]}<br>"
                "<b>Métrique:</b> %{customdata[1]}<br>"
                "<b>Performance:</b> %{customdata[2]:.2f} <extra></extra>" 
            )
        )

        return fig

def create_CMJ_unilat_blessure(df, bar_cols, point_col, title, y1_label, y2_label):
    """
    Crée un graphique combiné avec des barres groupées (Y1) pour les hauteurs de saut 
    et une ligne/points (Y2) pour l'indicateur de Symétrie/Asymétrie.
    
    Les valeurs numériques sont affichées en gras au-dessus des barres (sans décimale).
    """
    
    # Prétraitement : Ne garder que les colonnes nécessaires et trier
    required_cols = ['Date Test'] + bar_cols + [point_col]
    temp_df = df[required_cols].dropna(subset=bar_cols, how='all').sort_values(by='Date Test')
    
    if temp_df.empty:
        st.info(f"Aucune donnée valide trouvée pour le graphique : {title}")
        return None
        
    # Créer la colonne de date formatée comme une chaîne de caractères (pour l'axe X)
    temp_df['Date Label'] = temp_df['Date Test'].dt.strftime('%d %b %Y')

    fig = go.Figure()

    # Définir une ligne de seuil visuelle pour la symétrie à 10%
    SEUIL_ASYMETRIE = 10 

    # Fonction pour formater le texte dans la barre
    def format_bar_text(series):
        return series.apply(lambda x: f"{x:.1f}" if pd.notna(x) else "")
    
    text_font_style = dict(family='Arial', size=12, color='black', weight='bold')

    
    # 1. Traces en Barres (Axe Y Primaire: Hauteur en cm)
    bar_colors = ['#8B0000', '#FFD700'] # Bleu pour Droit, Orange pour Gauche
    
    for i, col in enumerate(bar_cols):
        # La trace n'inclut plus les arguments text et textfont ici, 
        # car ils seront gérés globalement par update_traces.
        fig.add_trace(go.Bar(
            x=temp_df['Date Label'],
            y=temp_df[col],
            name=col.replace('Max CMJ 1J ', 'CMJ 1J '), 
            marker_color=bar_colors[i % len(bar_colors)],
            yaxis='y1',
            text=format_bar_text(temp_df[bar_cols[0]]),
            textposition='outside',
            textfont=text_font_style, 
            # Note: hovertemplate conserve 2 décimales pour la précision au survol
            hovertemplate=(
                f"<b>{col}:</b> %{{y:.2f}} cm<br>"
                "<b>Date Test:</b> %{x}<extra></extra>"
            )
        ))

    # --- NOUVEAU: Application du format de texte aux traces de barres ---
    # Applique un modèle de texte en gras et le positionnement extérieur pour toutes les barres
    # Nous utilisons ici fig.update_traces pour appliquer cette mise à jour à toutes les traces de type 'bar'
    fig.update_traces(
        selector=dict(type='bar'), # Important pour n'affecter que les traces de barres
        texttemplate='<b>%{y:.1f}</b>', # Affichage en gras sans décimale
        textposition='outside',
        textfont=dict(size=14, color='black') # Optionnel: Ajuster la taille de la police pour le texte de la barre
    )
    # ----------------------------------------------------------------------
    
    # 2. Trace en Ligne/Points (Axe Y Secondaire: Symétrie %)
    if temp_df[point_col].dropna().any():
        sym_name = point_col.replace('Sym D/G', 'Asymétrie CMJ 1J (%)')
        
        fig.add_trace(go.Scatter(
            x=temp_df['Date Label'],
            y=temp_df[point_col],
            name=sym_name,
            mode='lines+markers',
            marker=dict(color='black', size=10),
            line=dict(dash='dash', color='black', width=3),
            yaxis='y2',
            hovertemplate=(
                f"<b>{point_col}:</b> %{{y:.2f}} %<br>"
                "<b>Date Test:</b> %{x}<extra></extra>"
            )
        ))

    # 3. Configuration de la Mise en Page
    # Calcul des ranges: Augmenter y1_max pour laisser de la place au texte 'outside'
    y2_max = max(temp_df[point_col].max() * 1.2 if not temp_df[point_col].empty and temp_df[point_col].max() > 0 else 20, SEUIL_ASYMETRIE * 1.5)
    y1_max = temp_df[bar_cols].max().max() * 1.15 if not temp_df[bar_cols].empty else 30 

    fig.update_layout(
        title={
            'text': title,
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=20)
        },
        template="plotly_white",
        barmode='group', # Groupement des barres par date
        xaxis=dict(
            title="",
            type='category', # Pour n'afficher que les dates réelles
            tickangle=-45,
            showgrid=True
        ),
        # Axe Y Primaire (Barres)
        yaxis=dict(
            title=y1_label,
            showgrid=True,
            zeroline=False,
            range=[0, y1_max] # Range ajusté
        ),
        # Axe Y Secondaire (Points) - Ajout d'une ligne de référence pour le seuil
        yaxis2=dict(
            title=y2_label,
            overlaying='y',
            side='right',
            showgrid=False,
            zeroline=False,
            range=[0, y2_max], # Range ajusté
        ),
        
        # Ajout de la ligne de référence pour le seuil de risque d'asymétrie
        shapes=[
            dict(
                type='line',
                yref='y2', y0=SEUIL_ASYMETRIE, y1=SEUIL_ASYMETRIE,
                xref='paper', x0=0, x1=1,
                line=dict(color='red', width=2, dash='dot'),
                name=f'Seuil {SEUIL_ASYMETRIE}%'
            )
        ],

        legend=dict(
            orientation="h", 
            yanchor="top",
            y=-0.2, 
            xanchor="center",
            x=0.5,
            traceorder="normal"
        ),
        height=550,
        margin=dict(t=50, l=80, r=80, b=100),
    )
    
    # Ajout d'annotation pour le seuil
    fig.add_annotation(
        yref='y2', y=SEUIL_ASYMETRIE,
        xref='paper', x=1.05,
        text=f"",
        showarrow=False,
        font=dict(color="red", size=10),
        align="left"
    )

    return fig

def create_LHT_chart(df):
    """
    Crée un graphique combiné pour le Long Hop Test (LHT) : 
    Barres (distance Droit/Gauche) + Ligne/Points (Asymétrie).
    
    Args:
        df (pd.DataFrame): DataFrame contenant les données des Hop Tests.
    """
    
    # Paramètres fixes pour le LHT
    bar_cols = ['LHT D', 'LHT G']
    point_col = 'Sym LHT'
    test_name = "Latéral Hop Test (LHT)"
    
    title = f"{test_name}"
    
    # Prétraitement : Ne garder que les colonnes nécessaires et trier
    required_cols = ['Date Test'] + bar_cols + [point_col]
    temp_df = df[required_cols].dropna(subset=bar_cols, how='all').sort_values(by='Date Test')
    
    if temp_df.empty:
        st.info(f"Aucune donnée valide trouvée pour le graphique : {title}")
        return None
        
    # Créer la colonne de date formatée comme une chaîne de caractères (pour l'axe X)
    temp_df['Date Label'] = temp_df['Date Test'].dt.strftime('%d %b %Y')

    fig = go.Figure()

    # Définir une ligne de seuil visuelle pour la symétrie à 10%
    SEUIL_ASYMETRIE = 10 
    
    # 1. Traces en Barres (Axe Y Primaire: Distance en cm)
    bar_colors = ['#8B0000', '#FFD700'] # Bleu pour Droit, Orange pour Gauche
    
    # Déterminer les noms de légende
    name_droit = f"Nombre Droite"
    name_gauche = f"Nombre Gauche"
    
    # Fonction pour formater le texte dans la barre
    def format_bar_text(series):
        return series.apply(lambda x: f"{x:.1f}" if pd.notna(x) else "")
    
    text_font_style = dict(family='Arial', size=12, color='black', weight='bold')

    # Trace Droit
    fig.add_trace(go.Bar(
        x=temp_df['Date Label'],
        y=temp_df[bar_cols[0]],
        name=name_droit, 
        marker_color=bar_colors[0],
        yaxis='y1',
        text=format_bar_text(temp_df[bar_cols[0]]),
        textposition='outside',
        textfont=text_font_style, 
        hovertemplate=(
            f"<b>{name_droit}:</b> %{{y:.1f}}<br>"
            "<b>Date Test:</b> %{x}<extra></extra>"
        )
    ))
    
    # Trace Gauche
    fig.add_trace(go.Bar(
        x=temp_df['Date Label'],
        y=temp_df[bar_cols[1]],
        name=name_gauche, 
        marker_color=bar_colors[1],
        yaxis='y1',
        text=format_bar_text(temp_df[bar_cols[1]]),
        textposition='outside',
        textfont=text_font_style, 
        hovertemplate=(
            f"<b>{name_gauche}:</b> %{{y:.1f}}<br>"
            "<b>Date Test:</b> %{x}<extra></extra>"
        )
    ))

    # 2. Trace en Ligne/Points (Axe Y Secondaire: Symétrie %)
    if temp_df[point_col].dropna().any():
        sym_name = point_col.replace('Sym ', 'Asymétrie ') + ' (%)'
        
        fig.add_trace(go.Scatter(
            x=temp_df['Date Label'],
            y=temp_df[point_col],
            name=sym_name,
            mode='lines+markers',
            marker=dict(color='black', size=12),
            line=dict(dash='dash', color='black', width=3),
            yaxis='y2',
            hovertemplate=(
                f"<b>{sym_name}:</b> %{{y:.2f}} %<br>"
                "<b>Date Test:</b> %{x}<extra></extra>"
            )
        ))

    # 3. Configuration de la Mise en Page
    y2_max = max(temp_df[point_col].max() * 1.2 if not temp_df[point_col].empty and temp_df[point_col].max() > 0 else 20, SEUIL_ASYMETRIE * 1.5)
    y1_max = temp_df[bar_cols].max().max() * 1.15 if not temp_df[bar_cols].empty else 100
    
    fig.update_layout(
        title={
            'text': title,
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=18)
        },
        template="plotly_white",
        barmode='group', # Groupement des barres par date
        xaxis=dict(
            title="", # Enlève le titre de l'axe X
            type='category', 
            tickangle=-45,
            showgrid=True
        ),
        # Axe Y Primaire (Barres - Distance)
        yaxis=dict(
            title=f"Nombre",
            showgrid=True,
            zeroline=False,
            range=[0, y1_max]
        ),
        # Axe Y Secondaire (Points - Asymétrie)
        yaxis2=dict(
            title="Asymétrie (%)",
            overlaying='y',
            side='right',
            showgrid=False,
            zeroline=False,
            range=[0, y2_max],
        ),
        
        # Ajout de la ligne de référence pour le seuil de risque d'asymétrie
        shapes=[
            dict(
                type='line',
                yref='y2', y0=SEUIL_ASYMETRIE, y1=SEUIL_ASYMETRIE,
                xref='paper', x0=0, x1=1,
                line=dict(color='red', width=2, dash='dot'),
                name=f'Seuil {SEUIL_ASYMETRIE}%'
            )
        ],

        legend=dict(
            orientation="h", 
            yanchor="top",
            y=-0.2, 
            xanchor="center",
            x=0.5,
            traceorder="normal"
        ),
        height=550,
        margin=dict(t=50, l=80, r=80, b=100),
    )
    
    # Ajout d'annotation pour le seuil
    fig.add_annotation(
        yref='y2', y=SEUIL_ASYMETRIE,
        xref='paper', x=1.05,
        text=f"",
        showarrow=False,
        font=dict(color="red", size=10),
        align="left"
    )

    return fig


def create_combined_hop_chart(df, bar_cols, point_col, test_name, test_name_legende, y1_unit="cm"):
    """
    Crée un graphique combiné pour un type de Hop Test : Barres (distance) + Ligne/Points (asymétrie).
    
    Args:
        df (pd.DataFrame): DataFrame contenant les données.
        bar_cols (list): Liste des colonnes de distance [Droit, Gauche].
        point_col (str): Colonne de l'indicateur de Symétrie (%).
        test_name (str): Nom du test (ex: 'LHT', 'Max SHT').
        y1_unit (str): Unité de mesure pour l'axe des distances (cm, m, etc.).
    """
    
    title = f"{test_name}"
    legende = f"{test_name_legende}"
    
    # Prétraitement : Ne garder que les colonnes nécessaires et trier
    required_cols = ['Date Test'] + bar_cols + [point_col]
    temp_df = df[required_cols].dropna(subset=bar_cols, how='all').sort_values(by='Date Test')
    
    if temp_df.empty:
        st.info(f"Aucune donnée valide trouvée pour le graphique : {title}")
        return None
        
    # Créer la colonne de date formatée comme une chaîne de caractères (pour l'axe X)
    temp_df['Date Label'] = temp_df['Date Test'].dt.strftime('%d %b %Y')

    fig = go.Figure()

    # Définir une ligne de seuil visuelle pour la symétrie à 10%
    SEUIL_ASYMETRIE = 10 
    
    # 1. Traces en Barres (Axe Y Primaire: Distance en cm)
    bar_colors = ['#8B0000', '#FFD700'] # Bleu pour Droit, Orange pour Gauche
    
    # Déterminer les noms de légende
    name_droit = f"{test_name_legende} Droit ({y1_unit})"
    name_gauche = f"{test_name_legende} Gauche ({y1_unit})"
    
    # Fonction pour formater le texte dans la barre
    def format_bar_text(series):
        return series.apply(lambda x: f"{x:.1f}" if pd.notna(x) else "")
    
    # Style de police pour le texte des barres (gras)
    text_font_style = dict(family='Arial', size=12, color='black', weight='bold') # 'bold' n'est pas direct ici, il faut passer par HTML ou CSS si on veut plus de contrôle, mais size et color sont bien pris en charge

    # Trace Droit
    fig.add_trace(go.Bar(
        x=temp_df['Date Label'],
        y=temp_df[bar_cols[0]],
        name=name_droit, 
        marker_color=bar_colors[0],
        yaxis='y1',
        text=format_bar_text(temp_df[bar_cols[0]]),
        textposition='outside',
        textfont=text_font_style, # AJOUT : style de police pour le gras
        hovertemplate=(
            f"<b>{name_droit}:</b> %{{y:.1f}} {y1_unit}<br>"
            "<b>Date Test:</b> %{x}<extra></extra>"
        )
    ))
    
    # Trace Gauche
    fig.add_trace(go.Bar(
        x=temp_df['Date Label'],
        y=temp_df[bar_cols[1]],
        name=name_gauche, 
        marker_color=bar_colors[1],
        yaxis='y1',
        text=format_bar_text(temp_df[bar_cols[1]]),
        textposition='outside',
        textfont=text_font_style, # AJOUT : style de police pour le gras
        hovertemplate=(
            f"<b>{name_gauche}:</b> %{{y:.1f}} {y1_unit}<br>"
            "<b>Date Test:</b> %{x}<extra></extra>"
        )
    ))

    # 2. Trace en Ligne/Points (Axe Y Secondaire: Symétrie %)
    if temp_df[point_col].dropna().any():
        sym_name = point_col.replace('Sym ', 'Asymétrie ') + ' (%)'
        
        fig.add_trace(go.Scatter(
            x=temp_df['Date Label'],
            y=temp_df[point_col],
            name=sym_name,
            mode='lines+markers',
            marker=dict(color='black', size=12),
            line=dict(dash='dash', color='black', width=3),
            yaxis='y2',
            hovertemplate=(
                f"<b>{sym_name}:</b> %{{y:.2f}} %<br>"
                "<b>Date Test:</b> %{x}<extra></extra>"
            )
        ))

    # 3. Configuration de la Mise en Page
    y2_max = max(temp_df[point_col].max() * 1.2 if not temp_df[point_col].empty and temp_df[point_col].max() > 0 else 20, SEUIL_ASYMETRIE * 1.5)
    
    # Pour s'assurer que le texte 'outside' est visible, on augmente légèrement la plage y1
    y1_max = temp_df[bar_cols].max().max() * 1.15 if not temp_df[bar_cols].empty else 100
    
    fig.update_layout(
        title={
            'text': title,
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=18)
        },
        template="plotly_white",
        barmode='group', # Groupement des barres par date
        xaxis=dict(
            title="", # Enlève le titre de l'axe X
            type='category', 
            tickangle=-45,
            showgrid=True
        ),
        # Axe Y Primaire (Barres - Distance)
        yaxis=dict(
            title=f"Distance ({y1_unit})",
            showgrid=True,
            zeroline=False,
            range=[0, y1_max] # Ajustement de la plage Y1 pour le texte 'outside'
        ),
        # Axe Y Secondaire (Points - Asymétrie)
        yaxis2=dict(
            title="Asymétrie (%)",
            overlaying='y',
            side='right',
            showgrid=False,
            zeroline=False,
            range=[0, y2_max],
        ),
        
        # Ajout de la ligne de référence pour le seuil de risque d'asymétrie
        shapes=[
            dict(
                type='line',
                yref='y2', y0=SEUIL_ASYMETRIE, y1=SEUIL_ASYMETRIE,
                xref='paper', x0=0, x1=1,
                line=dict(color='red', width=2, dash='dot'),
                name=f'Seuil {SEUIL_ASYMETRIE}%'
            )
        ],

        legend=dict(
            orientation="h", 
            yanchor="top",
            y=-0.2, 
            xanchor="center",
            x=0.5,
            traceorder="normal"
        ),
        height=500,
        margin=dict(t=50, l=10, r=10, b=100),
    )
    
    # Ajout d'annotation pour le seuil
    fig.add_annotation(
        yref='y2', y=SEUIL_ASYMETRIE,
        xref='paper', x=1.05,
        text=f"",
        showarrow=False,
        font=dict(color="red", size=10),
        align="left"
    )

    return fig

def create_combined_hop_performance(df, bar_cols, test_name, test_name_legende, y1_unit="cm"):
    """
    Crée un graphique de barres groupées (Y1) pour les Hop Tests (SHT, THT, etc.).
    Ceci est la version Performance, sans l'indicateur d'asymétrie ni l'axe Y secondaire.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les données.
        bar_cols (list): Liste des colonnes de distance [Droit, Gauche].
        test_name (str): Nom du test (ex: 'Max SHT', 'THT').
        test_name_legende (str): Nom du test pour la légende (ex: 'SHT', 'THT').
        y1_unit (str): Unité de mesure pour l'axe des distances (cm, m, etc.).
    """
    
    title = f"{test_name} - Évolution des Performances"
    
    # 1. CONVERSION CRUCIALE : S'assurer que 'Date Test' est bien un format datetime
    # Le paramètre errors='coerce' remplace les valeurs invalides par NaT (Not a Time), 
    # ce qui empêche l'erreur.
    df['Date Test'] = pd.to_datetime(df['Date Test'], errors='coerce')
    
    # Prétraitement : Ne garder que les colonnes nécessaires et trier
    # On utilise dropna(subset=['Date Test']) pour retirer les lignes qui auraient 
    # une date invalide (convertie en NaT par 'coerce')
    required_cols = ['Date Test'] + bar_cols
    temp_df = df[required_cols].dropna(subset=['Date Test'] + bar_cols, how='all').sort_values(by='Date Test')
    
    if temp_df.empty:
        st.info(f"Aucune donnée valide trouvée pour le graphique : {title}")
        return None
        
    # Créer la colonne de date formatée comme une chaîne de caractères (pour l'axe X)
    # Cette ligne fonctionne maintenant car 'Date Test' est garanti d'être datetime
    temp_df['Date Label'] = temp_df['Date Test'].dt.strftime('%d %b %Y')

    fig = go.Figure()
    
    # --- COULEURS SANG ET OR ---
    bar_colors = ['#8B0000', '#FFD700'] # Rouge Bordeaux (Sang) pour Droit, Or Vieilli pour Gauche
    
    # Déterminer les noms de légende
    name_droit = f"{test_name_legende} Droit ({y1_unit})"
    name_gauche = f"{test_name_legende} Gauche ({y1_unit})"
    
    # Fonction pour formater le texte dans la barre (1 décimale)
    def format_bar_text(series):
        return series.apply(lambda x: f"{x:.1f}" if pd.notna(x) else "")
    
    text_font_style = dict(family='Arial', size=12, color='black')

    # 1. Trace Droit
    fig.add_trace(go.Bar(
        x=temp_df['Date Label'],
        y=temp_df[bar_cols[0]],
        name=name_droit, 
        marker_color=bar_colors[0],
        text=format_bar_text(temp_df[bar_cols[0]]),
        textposition='outside',
        textfont=text_font_style, 
        hovertemplate=(
            f"<b>{name_droit}:</b> %{{y:.1f}} {y1_unit}<br>"
            "<b>Date Test:</b> %{x}<extra></extra>"
        )
    ))
    
    # 2. Trace Gauche
    fig.add_trace(go.Bar(
        x=temp_df['Date Label'],
        y=temp_df[bar_cols[1]],
        name=name_gauche, 
        marker_color=bar_colors[1],
        text=format_bar_text(temp_df[bar_cols[1]]),
        textposition='outside',
        textfont=text_font_style, 
        hovertemplate=(
            f"<b>{name_gauche}:</b> %{{y:.1f}} {y1_unit}<br>"
            "<b>Date Test:</b> %{x}<extra></extra>"
        )
    ))

    # Mise à jour du style de texte pour les barres (affichage en gras)
    fig.update_traces(
        selector=dict(type='bar'), 
        texttemplate='<b>%{text}</b>', 
        textfont=dict(size=14, color='black') 
    )

    # 3. Configuration de la Mise en Page
    
    y1_max = temp_df[bar_cols].max().max() * 1.15 if not temp_df[bar_cols].empty else 100
    
    fig.update_layout(
        title={
            'text': title,
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=18)
        },
        template="plotly_white",
        barmode='group', 
        xaxis=dict(
            title="", 
            type='category', 
            tickangle=-45,
            showgrid=True
        ),
        # Axe Y Primaire (Barres - Distance)
        yaxis=dict(
            title=f"Distance ({y1_unit})",
            showgrid=True,
            zeroline=False,
            range=[0, y1_max] 
        ),
        
        legend=dict(
            orientation="h", 
            yanchor="top",
            y=-0.2, 
            xanchor="center",
            x=0.5,
            traceorder="normal"
        ),
        height=550, 
        margin=dict(t=50, l=80, r=80, b=100), 
    )
    
    return fig

def create_CMJ_unilat_perf(df, bar_cols, title, y1_label):
    """
    Crée un graphique de barres groupées (Y1) pour les hauteurs de saut unilatérales.
    Ceci est la version Performance, sans l'indicateur d'asymétrie ni l'axe Y secondaire.
    
    Les valeurs numériques sont affichées en gras au-dessus des barres (sans décimale).
    
    Args:
        df (pd.DataFrame): DataFrame contenant les données.
        bar_cols (list): Liste des colonnes de hauteur [Droit, Gauche].
        title (str): Titre du graphique.
        y1_label (str): Label de l'axe Y (ex: 'Hauteur de saut (cm)').
    """
    
    # Prétraitement : Ne garder que les colonnes nécessaires et trier
    required_cols = ['Date Test'] + bar_cols
    temp_df = df[required_cols].dropna(subset=bar_cols, how='all').sort_values(by='Date Test')
    
    if temp_df.empty:
        st.info(f"Aucune donnée valide trouvée pour le graphique : {title}")
        return None
        
    # Créer la colonne de date formatée comme une chaîne de caractères (pour l'axe X)
    temp_df['Date Label'] = temp_df['Date Test'].dt.strftime('%d %b %Y')

    fig = go.Figure()

    # --- COULEURS SANG ET OR ---
    bar_colors = ['#8B0000', '#FFD700'] # Rouge Bordeaux (Sang) pour Droit, Or Vieilli pour Gauche
    
    # 1. Traces en Barres (Axe Y Primaire: Hauteur en cm)
    for i, col in enumerate(bar_cols):
        # Utilisation de .replace pour une meilleure lisibilité dans la légende
        legende_name = col.replace('Max CMJ 1J ', 'CMJ 1J ') 

        fig.add_trace(go.Bar(
            x=temp_df['Date Label'],
            y=temp_df[col],
            name=legende_name, 
            marker_color=bar_colors[i % len(bar_colors)],
            yaxis='y1',
            hovertemplate=(
                f"<b>{legende_name}:</b> %{{y:.1f}} cm<br>" # Note: 1 décimale dans le hover
                "<b>Date Test:</b> %{x}<extra></extra>"
            )
        ))

    # Application du format de texte aux traces de barres (0 décimale, gras, outside)
    fig.update_traces(
        selector=dict(type='bar'), 
        texttemplate='<b>%{y:.1f}</b>', # Affichage en gras sans décimale pour le CMJ 1J
        textposition='outside',
        textfont=dict(size=14, color='black') 
    )
    
    # 2. Configuration de la Mise en Page
    # Calcul des ranges: Augmenter y1_max pour laisser de la place au texte 'outside'
    y1_max = temp_df[bar_cols].max().max() * 1.15 if not temp_df[bar_cols].empty else 30 

    fig.update_layout(
        title={
            'text': title,
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=20)
        },
        template="plotly_white",
        barmode='group', 
        xaxis=dict(
            title="",
            type='category', 
            tickangle=-45,
            showgrid=True
        ),
        # Axe Y Principal (Barres)
        yaxis=dict(
            title=y1_label,
            showgrid=True,
            zeroline=False,
            range=[0, y1_max] 
        ),
        # Suppression de l'Axe Y Secondaire (yaxis2)
        
        legend=dict(
            orientation="h", 
            yanchor="top",
            y=-0.2, 
            xanchor="center",
            x=0.5,
            traceorder="normal"
        ),
        height=550, # Hauteur harmonisée
        margin=dict(t=50, l=80, r=80, b=100), # Marges harmonisées
    )
    
    return fig