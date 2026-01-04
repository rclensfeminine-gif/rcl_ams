import streamlit as st
import pandas as pd
import os
import re
from datetime import date, timedelta
from sections.planning.pipeline import PLANNING_DIR, get_match_color, get_monday, load_week_names, save_week_names, load_planning_data, save_planning_data
from sections.menu.menu import custom_sidebar_menu


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

# Options pour les menus déroulants
JOURS_SEMAINE = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
PERIODES_JOURNEE = ["Matin", "Midi", "Après-midi"]

# Dictionnaire des modèles prédéfinis avec des horaires
TEMPLATES = {
    "Sélectionner": "",
    "Séance M": "9h30 - Vidéo\n10h - Activation\n10h30 - Séance",
    "Muscu AP": "15h - Muscu",
    "Séance + Muscu": "14h30 - Activation\n15h - Séance\n17h - Muscu",
    "S J-4": "10h - Vidéo\n10h15 - Strap\n10h30 - Activation\n11h - Séance",
    "S J-3 M": "9h45 - Vidéo\n10h15 - Strap\n10h30 - Activation\n11h - Séance",
    "M J-3 AP": "13h30 - Atelier tactique\n14h30 - Muscu",
    "S J-3 AP": "13h30 - Atelier tactique\n14h30 - Activation\n15h - Séance\n17h - Muscu",
    "S J-2": "14h - Vidéo\n14h30 - Activation\n15h - Séance\n16h30 - Muscu",
    "S J-1": "14h15 - Vidéo\n15h - Activation\n15h30 - Séance"
}

# --- Fonction d'initialisation de la session ---
def initialize_session(selected_monday):
    planning_filename = f"planning_{selected_monday}.csv"
    planning_filepath = os.path.join(PLANNING_DIR, planning_filename)
    df_planning = load_planning_data(planning_filepath)

    if df_planning.empty:
        df_planning = pd.DataFrame(index=PERIODES_JOURNEE, columns=JOURS_SEMAINE).fillna('')
    else:
        df_planning = df_planning.astype(str).replace('nan', '')

    st.session_state['planning_data'] = df_planning
    st.session_state['last_loaded_date'] = selected_monday

    for periode in PERIODES_JOURNEE:
        for jour in JOURS_SEMAINE:
            events_key = f"events_{periode}_{jour}"
            template_key = f"template_{periode}_{jour}"

            initial_value = st.session_state['planning_data'].loc[periode, jour]
            st.session_state[events_key] = initial_value
            st.session_state[template_key] = list(TEMPLATES.keys())[0]

# --- UI de l'application ---
st.title("Planning")

# Sélecteur de date pour le lundi de la semaine
current_monday = get_monday(date.today())
if 'last_loaded_date' not in st.session_state:
    st.session_state['last_loaded_date'] = current_monday

selected_monday = st.date_input("Sélectionner le lundi de la semaine",
                                 value=st.session_state['last_loaded_date'],
                                 key="date_selector")

# Si la date a changé, relancer la page pour recharger les données
if selected_monday != st.session_state['last_loaded_date']:
    initialize_session(selected_monday)
    st.rerun()

# Initialisation de la session si c'est la première fois ou si l'utilisateur a relancé l'app
if 'planning_data' not in st.session_state:
    initialize_session(selected_monday)

# Chargement du nom de la semaine
week_names = load_week_names()
week_key = selected_monday.isoformat()
default_week_name = week_names.get(week_key, "")

# Champ pour nommer la semaine
new_week_name = st.text_input(f"Nom de la semaine ({selected_monday.strftime('%d/%m/%Y')}):", value=default_week_name)

# --- Fonction de rappel pour mettre à jour le text_area ---
def update_from_template(periode, jour):
    template_key = f"template_{periode}_{jour}"
    events_key = f"events_{periode}_{jour}"
    selected_template = st.session_state[template_key]
    if selected_template != "Sélectionner un modèle...":
        st.session_state[events_key] = TEMPLATES[selected_template]

# --- Affichage du planning avec des widgets ---
st.header("Remplir le planning")

# Création des en-têtes de colonnes
cols_header = st.columns(len(JOURS_SEMAINE) + 1)
cols_header[0].empty()
for i, jour in enumerate(JOURS_SEMAINE):
    with cols_header[i + 1]:
        st.markdown(f"<div style='text-align: center;'><b>{jour}</b></div>", unsafe_allow_html=True)

# Affichage de la grille du planning
for periode in PERIODES_JOURNEE:
    cols = st.columns(len(JOURS_SEMAINE) + 1)
    with cols[0]:
        st.markdown(f"<div style='text-align: center;'><b>{periode}</b></div>", unsafe_allow_html=True)
    
    for i, jour in enumerate(JOURS_SEMAINE):
        with cols[i + 1]:
            events_key = f"events_{periode}_{jour}"
            template_key = f"template_{periode}_{jour}"
            
            st.selectbox(
                label="Sélectionner un modèle",
                options=list(TEMPLATES.keys()),
                key=template_key,
                label_visibility="collapsed",
                on_change=update_from_template,
                args=(periode, jour)
            )

            st.text_area(
                label=f"{periode} - {jour}",
                key=events_key,
                height=150,
                label_visibility="collapsed"
            )

# --- Sauvegarde ---
st.markdown("---")

# Définir planning_filepath avant le bouton
planning_filename = f"planning_{selected_monday}.csv"
planning_filepath = os.path.join(PLANNING_DIR, planning_filename)

if st.button("Sauvegarder le planning"):
    new_df_planning = pd.DataFrame(index=PERIODES_JOURNEE, columns=JOURS_SEMAINE)
    
    for periode in PERIODES_JOURNEE:
        for jour in JOURS_SEMAINE:
            events_text = st.session_state[f"events_{periode}_{jour}"]
            new_df_planning.loc[periode, jour] = events_text

    save_planning_data(new_df_planning, planning_filepath)
    st.session_state['planning_data'] = new_df_planning
    
    week_names[week_key] = new_week_name
    save_week_names(week_names)
    
    st.success(f"Le planning '{new_week_name}' a été sauvegardé avec succès !")
    st.rerun()

# --- Fonctions de style et de couleur ---
def format_cell(text):
    if isinstance(text, str):
        return text.replace('\n', '<br>')
    return text

# --- Styles globaux pour les tableaux ---
table_style_main = """
<style>
    table {
        width: 100%;
        border-collapse: collapse;
        table-layout: fixed;
    }
    th, td {
        border: 1px solid black;
        text-align: center;
        vertical-align: middle;
        font-size: 12px;
    }
    th {
        background-color: #f2f2f2;
    }
    td:first-child, th:first-child {
        width: 75px; /* Largeur pour les colonnes des périodes et des jours */
    }
    .flex-container {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        height: 100%;
    }
</style>
"""
st.markdown(table_style_main, unsafe_allow_html=True)

# --- Styles spécifiques pour le planning hebdomadaire (AJOUTÉ) ---
table_style_week = """
<style>
    /* Styles spécifiques pour le planning hebdomadaire (weekly-table) */
    
    /* BORDURES EXTÉRIEURES ÉPAISSES */
    .weekly-table {
        border: 2px solid #000000 !important;
        border-collapse: collapse; /* Assure que les bordures se fusionnent correctement */
    }
    
    /* Centrage du texte dans l'en-tête (les jours) */
    .weekly-table th {
        text-align: center !important; 
    }
    
    /* BARRE HORIZONTALE (sous les jours) */
    .weekly-table thead tr th {
        /* Bordure sous la ligne des jours */
        border-bottom: 2px solid #000000 !important;
    }

    /* BARRE VERTICALE entre Périodes et Lundi (1ère colonne) */
    .weekly-table th:first-child,
    .weekly-table td:first-child { 
        border-right: 2px solid #000000 !important;
    }

    /* BARRE HORIZONTALE INITIATE (au-dessus du Matin, sur toute la largeur) */
    /* Cible la première ligne du corps du tableau (Matin) */
    .weekly-table tbody tr:first-child td {
        border-top: 2px solid #000000 !important;
    }
    
    /* BORDURES VERTICALES ÉPAISSES entre les jours (TH et TD) */
    /* On cible tous les TH et TD SAUF la première colonne (qui a déjà sa règle) et la dernière colonne */
    .weekly-table th:not(:first-child):not(:last-child),
    .weekly-table tbody tr td:not(:first-child):not(:last-child) {
        border-right: 2px solid #000000 !important;
    }

    /* Bordures restantes pour les TD, rétablit une bordure fine par défaut si besoin */
    .weekly-table td {
        border: 1px solid #ccc; /* Bordure fine pour la séparation horizontale */
    }

</style>
"""
st.markdown(table_style_week, unsafe_allow_html=True)


# --- Affichage du Planning de la semaine (MODIFIÉ POUR CLASSE CSS) ---
st.markdown("---")
st.header("Planning de la semaine : ")

if 'planning_data' in st.session_state and not st.session_state['planning_data'].empty:
    df_display = st.session_state['planning_data'].copy()
    
    if new_week_name:
        st.subheader(new_week_name)
    
    # AJOUT DE LA CLASSE 'weekly-table'
    html_table = "<table class='weekly-table'>" 
    
    # En-tête : Jour
    html_table += "<thead>"
    html_table += "<tr><th rowspan='2'></th>" # Cellule vide en haut à gauche
    for jour in JOURS_SEMAINE:
        html_table += f"<th>{jour}</th>"
    html_table += "</tr>"
    html_table += "<tr>"
    # Ceci était une ligne d'en-tête vide et n'est pas nécessaire pour ce tableau
    html_table += "</tr></thead>" 

    # Corps : Période + Contenu
    html_table += "<tbody>"
    for periode in PERIODES_JOURNEE:
        html_table += "<tr>"
        # Colonne Périodes (Matin, Midi, Après-midi)
        html_table += f"<td style='height: 80px;'><b>{periode}</b></td>" 
        for jour in JOURS_SEMAINE:
            cell_value = df_display.loc[periode, jour]
            cell_content = format_cell(cell_value)
            color_style = get_match_color(cell_value)
            
            # Application de la classe flex-container
            html_table += f"<td style='{color_style}; height: 80px;'><div class='flex-container'><b>{cell_content}</b></div></td>"
        html_table += "</tr>"
    html_table += "</tbody>"
    html_table += "</table>"
    
    st.markdown(html_table, unsafe_allow_html=True)
else:
    st.info("Aucun planning n'a été rempli ou sauvegardé pour cette semaine.")

# --- Légende ---
st.subheader("Légende")

legend_html = """
<style>
    .legend-table {
        border-collapse: collapse;
        width: 100%;
    }
    .legend-table td {
        border: 3px solid #ddd;
        padding: 10px;
        font-size: 12px;
    }
    .color-box {
        width: 20px;
        height: 20px;
        border: 1px solid black;
        display: inline-block;
        vertical-align: middle;
        margin-right: 5px;
    }
    .legend-item {
        display: inline-block;
        margin-right: 20px;
    }
</style>
<div>
    <span class="legend-item"><div class="color-box" style="background-color: #006400;"></div> Championnat (J)</span>
    <span class="legend-item"><div class="color-box" style="background-color: #008000;"></div> Coupe LFFP (CL)</span>
    <span class="legend-item"><div class="color-box" style="background-color: #2E8B57;"></div> Coupe de France (CDF)</span>
    <span class="legend-item"><div class="color-box" style="background-color: #90EE90;"></div> Prépa / Amical</span>
    <span class="legend-item"><div class="color-box" style="background-color: #FFD700;"></div> Séance / Compensation</span>
    <span class="legend-item"><div class="color-box" style="background-color: #99c2ff;"></div> Récupération</span>
    <span class="legend-item"><div class="color-box" style="background-color: #FFC0CB;"></div> Muscu</span>
</div>
"""
st.markdown(legend_html, unsafe_allow_html=True)

# --- Planning du mois ---
st.markdown("---")
st.header("Planning du mois :")

# --- Styles spécifiques pour le planning mensuel (CORRIGÉ V5 : BORDURE VERTICALE INITIALE) ---
table_style_month = """
<style>
    /* Styles généraux de la table mensuelle */
    .monthly-table {
        border-collapse: collapse; 
        border: 2px solid #000000; 
        width: 100%;
    }
    
    .monthly-table th, .monthly-table td {
        border: 1px solid #ccc;
        padding: 5px;
        text-align: center;      /* Centrage horizontal */
        vertical-align: middle;  /* Centrage vertical */
    }
    
    /* En-têtes (Semaine et Période) */
    .monthly-table th {
        background-color: #f0f0f0;
        font-weight: bold;
    }
    
    /* 1. Suppression de la bordure sous les noms de semaines */
    .monthly-table thead tr:first-child th {
        border-bottom: 1px solid #ccc !important; 
    }

    /* 2. Conservation de la bordure sous Matin/Midi/Apres-midi */
    .monthly-table thead tr:nth-child(2) th {
        border-bottom: 2px solid #000000 !important; 
    }
    
    /* 3. AJOUT DES SÉPARATEURS DE SEMAINES (Barres verticales épaisses) */
    
    /* Cible les TH de la PREMIÈRE ligne (Semaine du...) et applique la bordure à droite */
    /* Exclut le TH vide et le dernier TH */
    .monthly-table thead tr:first-child th:not(:last-child) {
        border-right: 2px solid #000000 !important;
    }
    
    /* Cible les TH/TD de la DEUXIÈME ligne et du CORPS pour la séparation INTERNE de semaine */
    .monthly-table thead tr:nth-child(2) th:nth-child(3n + 3),
    .monthly-table tbody tr td:nth-child(3n + 4) {
        border-right: 2px solid #000000 !important;
    }
    
    /* Correction de la dernière colonne: le dernier th doit avoir la bordure du tableau, pas la bordure de séparation interne */
    .monthly-table thead tr:nth-child(2) th:last-child {
         border-right: 2px solid #000000; 
    }
    
    /* --- NOUVEAUX STYLES DE BORDURES INITIALES --- */
    
    /* BARRE VERTICALE INITIATE (avant Matin de la 1ère semaine) */
    /* Cible la première cellule de données dans le corps (Matin du Lundi) */
    .monthly-table tbody tr td:nth-child(2) {
        /* Nous utilisons border-left pour la séparation interne après la colonne Jour */
        border-left: 2px solid #000000 !important; 
    }
    
    /* BARRE HORIZONTALE INITIATE (au-dessus du Lundi, sur toute la largeur) */
    /* Cible la première ligne de données (Lundi) et applique la bordure en haut */
    .monthly-table tbody tr:first-child td {
        border-top: 2px solid #000000 !important;
    }
    
    /* La bordure pour les jours doit rester épaisse (colonne Jours) */
    .monthly-table td:first-child {
        background-color: #e0e0e0;
        font-weight: bold;
        /* La bordure droite de cette colonne est la séparation verticale */
        border-right: 2px solid #000000 !important;
    }
    
    /* Le conteneur Flex pour centrer le contenu multi-ligne */
    .monthly-table .flex-container {
        display: flex;
        flex-direction: column;
        justify-content: center; /* Centrage vertical */
        align-items: center; /* Centrage horizontal */
        height: 100%; 
        min-height: 80px; 
    }
    
</style>
"""
st.markdown(table_style_month, unsafe_allow_html=True)


# Déterminer les 4 prochains lundis à partir de la date sélectionnée
prochains_lundis = []
current_week_monday = selected_monday
for _ in range(4):
    prochains_lundis.append(current_week_monday)
    current_week_monday += timedelta(days=7)

# --- Construire le tableau HTML du mois manuellement ---
html_table_month = "<table class='monthly-table'>" # Ajout de la classe CSS pour le ciblage

# Première ligne d'en-têtes : les semaines
html_table_month += "<thead>"
html_table_month += "<tr><th rowspan='2'></th>"
# On compte le nombre de semaines pour potentiellement ajuster la bordure dans le CSS
num_weeks = len(prochains_lundis) 
for i, monday in enumerate(prochains_lundis):
    week_name = load_week_names().get(monday.isoformat(), f"Semaine du {monday.strftime('%d/%m')}")
    # Le style de bordure est maintenant géré par :nth-child dans le CSS
    html_table_month += f"<th colspan='3'>{week_name}</th>" 
html_table_month += "</tr>"

# Deuxième ligne d'en-têtes : les périodes de la journée
html_table_month += "<tr>"
for _ in prochains_lundis:
    for periode in PERIODES_JOURNEE:
        # Le style de bordure est géré par :nth-child dans le CSS
        html_table_month += f"<th>{periode}</th>"
html_table_month += "</tr>"
html_table_month += "</thead>"

# Corps du tableau : les jours et les données
html_table_month += "<tbody>"
for jour in JOURS_SEMAINE:
    html_table_month += "<tr>"
    # Le style 'height: 80px;' est conservé ici pour la première colonne, mais les autres TD n'en ont plus besoin car le flex-container le gère
    html_table_month += f"<td style='height: 80px;'><b>{jour}</b></td>"
    
    for monday in prochains_lundis:
        planning_filepath_month = os.path.join(PLANNING_DIR, f"planning_{monday}.csv")
        df_week = load_planning_data(planning_filepath_month)
        
        for periode in PERIODES_JOURNEE:
            cell_value = ""
            if not df_week.empty and jour in df_week.columns and periode in df_week.index:
                cell_value = df_week.loc[periode, jour]
            
            cell_content = str(cell_value).replace('\n', '<br>').replace('nan', '')
            color_style = get_match_color(cell_value)
            
            # Suppression du style 'height: 80px;' du TD de contenu, car c'est géré par le CSS et le flex-container
            # Le style de bordure est géré par :nth-child dans le CSS
            html_table_month += f"<td style='{color_style}'><div class='flex-container'><b>{cell_content}</b></div></td>"
            
    html_table_month += "</tr>"

html_table_month += "</tbody>"
html_table_month += "</table>"

# Afficher le tableau HTML
st.markdown(html_table_month, unsafe_allow_html=True)

# --- Légende ---
st.subheader("Légende")
st.markdown(legend_html, unsafe_allow_html=True)

# --- Planning Annuel Dynamique ---
st.markdown("---")
st.header("Planning de la saison :")

# Styles spécifiques pour le planning annuel
# Mise à jour du style pour l'encadrement complet
table_style_annual = """
<style>
    /* Styles généraux de la table */
    .annual-table {
        border-collapse: collapse; /* Supprime l'espace entre les bordures */
        border: 2px solid #000000; /* BORDURE GLOBALE HAUTE ET BASSE */
        width: 100%;
        table-layout: fixed;
    }
    .annual-table th, .annual-table td {
        height: 30px; 
        font-size: 8px; 
        padding: 2px;
        /* Bordures fines par défaut à l'intérieur */
        border-right: 1px solid #ccc; 
        border-bottom: 1px solid #eee; 
    }

    /* Centrage du texte des mois (TH) */
    .annual-table th {
        text-align: center; /* AJOUT : Centre le nom des mois */
        border-bottom: 2px solid #000000 !important; /* Bordure sous les mois en gras (faisant partie de la bordure globale) */
        border-top: none; /* Géré par la bordure globale du tableau */
    }

    /* STYLE POUR LES JOURS INEXISTANTS (ex: 30 Février) */
    .annual-table .invalid-date {
        background-color: #f0f0f0; /* Gris clair */
        color: #aaaaaa; /* Texte gris */
    }
    
    /* 1. Séparateur Horizontal (Jour 1 de chaque mois) */
    .annual-table .monthly-separator td {
        border-top: 2px solid #000000 !important; /* Bordure haute en gras et noire */
        font-weight: bold; 
    }
    
    /* 2. Séparateur Vertical entre les mois */
    .annual-table .monthly-v-separator {
        border-right: 2px solid #000000 !important; /* Bordure droite en gras et noire */
    }
    
    /* 3. Bordure Gauche (Avant les numéros de jour) */
    .annual-table th:first-child, .annual-table td:first-child {
        border-right: 2px solid #000000 !important; /* Bordure droite épaisse pour la colonne des jours */
        border-left: none; /* La bordure du tableau gère l'extrême gauche */
    }
    .annual-table th:first-child {
        font-weight: bold;
    }
    
    /* 4. Bordure Droite (Après Juin) */
    /* La bordure droite est gérée par la classe .monthly-v-separator sur l'index 12 */
    
    /* 5. Bordure Basse (Dernière ligne) */
    .annual-table .last-row td {
        border-bottom: none !important; /* La bordure globale du tableau gère l'extrême bas */
    }

    /* Ajustement des coins pour ne pas avoir de doubles bordures inutiles */
    .annual-table th {
        border-bottom: 2px solid #000000 !important; /* Bordure sous les mois en gras (faisant partie de la bordure globale) */
        border-top: none; /* Géré par la bordure globale du tableau */
    }
    
</style>
"""
st.markdown(table_style_annual, unsafe_allow_html=True)

def load_annual_planning_data(year_start):
    """Charge toutes les données de planning pour l'année de juillet à juin."""
    annual_data = {}
    
    start_date = date(year_start, 7, 1)
    end_date = date(year_start + 1, 6, 30)

    current_date = start_date
    while current_date <= end_date:
        monday_of_week = get_monday(current_date)
        planning_filename = f"planning_{monday_of_week}.csv"
        planning_filepath = os.path.join(PLANNING_DIR, planning_filename)
        
        df_week = load_planning_data(planning_filepath)
        
        if not df_week.empty:
            df_week = df_week.astype(str).replace('nan', '')

            for day_index, jour in enumerate(JOURS_SEMAINE):
                current_day_date = monday_of_week + timedelta(days=day_index)
                if start_date <= current_day_date <= end_date:
                    
                    found_keywords = set()
                    
                    for periode in PERIODES_JOURNEE:
                        if periode in df_week.index and jour in df_week.columns:
                            cell_content = df_week.loc[periode, jour].strip()
                            
                            if not cell_content:
                                continue 

                            lines = [line.strip() for line in cell_content.split('\n') if line.strip()]

                            for line in lines:
                                # Priorité 1 : Recherche de Matchs (J, CL, CDF, Amical, Prépa)
                                match_game_ref = re.match(r'(j|cl|cdf)\.?\s*(\d+|finale)', line, re.IGNORECASE)
                                match_amical_prepa = re.match(r'(amical|prépa)', line, re.IGNORECASE)
                                
                                if match_game_ref or match_amical_prepa:
                                    found_keywords.add(line) 
                                    continue 
                                
                                # Priorité 2 : Recherche de Séances / Muscu / Comp (Extraction de l'abréviation)
                                match_seance = re.search(r'(séance|s)\s*(\d+)', line, re.IGNORECASE)
                                if match_seance:
                                    found_keywords.add(f"S{match_seance.group(2)}")
                                    continue
                                
                                match_comp = re.search(r'(compensation|c)\s*(\d+)', line, re.IGNORECASE)
                                if match_comp:
                                    found_keywords.add(f"C{match_comp.group(2)}")
                                    continue
                                
                                match_muscu = re.search(r'(muscu|m)\s*(\d+)', line, re.IGNORECASE)
                                if match_muscu:
                                    found_keywords.add(f"M{match_muscu.group(2)}")
                                    continue
                                
                                # Ajout de Récupération ou Récup au contenu final
                                if "récup" in line.lower() or "recup" in line.lower():
                                    found_keywords.add("Récup")
                                    continue
                    
                    # --- Aggrégation et Tri ---
                    final_content_parts = []
                    
                    # 1. Matchs, Amicaux, Prépa (texte complet)
                    matches_and_others_kw = sorted([
                        item for item in found_keywords 
                        if not (
                            (item.lower().startswith('s') and re.match(r's\d+', item.lower())) or 
                            (item.lower().startswith('m') and re.match(r'm\d+', item.lower())) or
                            (item.lower().startswith('c') and re.match(r'c\d+', item.lower()))
                        )
                    ])
                    final_content_parts.extend(matches_and_others_kw) 
                    
                    # 2. Séances, Muscu, Comp (abréviations)
                    seances = sorted([item for item in found_keywords if item.lower().startswith('s') and re.match(r's\d+', item.lower())])
                    muscus = sorted([item for item in found_keywords if item.lower().startswith('m') and re.match(r'm\d+', item.lower())])
                    compensations = sorted([item for item in found_keywords if item.lower().startswith('c') and re.match(r'c\d+', item.lower())])

                    final_content_parts.extend(seances)
                    final_content_parts.extend(muscus)
                    final_content_parts.extend(compensations)
                    
                    # Sauvegarde du contenu final pour la date
                    annual_data[current_day_date] = '\n'.join(final_content_parts)
        
        current_date += timedelta(days=7)
    
    return annual_data

# Année de la saison à afficher
current_year = date.today().year
season_options = [f"{current_year-1}-{current_year}", f"{current_year}-{current_year+1}"]
selected_season = st.selectbox("Sélectionner la saison", options=season_options, index=1)
year_start_int = int(selected_season.split('-')[0])

annual_data_dict = load_annual_planning_data(year_start_int)

# Dictionnaire pour mapper les noms des mois aux mois de l'année (1-12)
month_mapping = {
    "Juillet": 7, "Août": 8, "Septembre": 9, "Octobre": 10, "Novembre": 11, "Décembre": 12,
    "Janvier": 1, "Février": 2, "Mars": 3, "Avril": 4, "Mai": 5, "Juin": 6
}
months = ["Juillet", "Août", "Septembre", "Octobre", "Novembre", "Décembre", "Janvier", "Février", "Mars", "Avril", "Mai", "Juin"]

# Création de la table
html_annual = "<table class='annual-table'><thead><tr><th></th>"
for i, month in enumerate(months):
    # Appliquer le séparateur vertical sur l'entête du mois, y compris Juin pour la bordure droite globale
    header_class = "monthly-v-separator" if i <= len(months) - 1 else "" # Appliqué à toutes les colonnes des mois
    html_annual += f"<th class='{header_class}'>{month}</th>"
html_annual += "</tr></thead><tbody>"

# Remplissage de la table par jour et par mois
for day in range(1, 32):
    
    # 1. Séparateur Horizontal (Jour 1)
    row_class = "monthly-separator" if day == 1 else "" 
    
    # 2. Bordure Basse (Jour 31 pour fermer le tableau)
    if day == 31:
        row_class += " last-row"
    
    html_annual += f"<tr class='{row_class}'>"
    html_annual += f"<td><b>{day}</b></td>" # La bordure gauche est gérée par le CSS sur td:first-child

    for i, month_name in enumerate(months):
        
        month_number = month_mapping[month_name]
        year_to_use = year_start_int + 1 if month_number <= 6 else year_start_int
        
        cell_content = ""
        color_style = ""
        
        try:
            # Tenter de créer une date
            current_date = date(year_to_use, month_number, day)
        except ValueError:
            # Si la date n'est pas valide (ex: 30 Février, 31 Septembre)
            current_date = None
            cell_class_extra = "invalid-date"

        if current_date and current_date in annual_data_dict:
            cell_content = annual_data_dict.get(current_date, "")
            color_style = get_match_color(cell_content)
        
        # Séparateur Vertical: appliqué à toutes les colonnes de mois, y compris Juin pour fermer à droite
        cell_class = "monthly-v-separator" if i <= len(months) - 1 else ""
        
        # Le contenu de la cellule est formaté ici
        html_annual += f"<td class='{cell_class}' style='{color_style}'>{format_cell(cell_content)}</td>"
            
    html_annual += "</tr>"
    
html_annual += "</tbody></table>"

st.markdown(html_annual, unsafe_allow_html=True)

# --- Légende ---
st.subheader("Légende")
st.markdown(legend_html, unsafe_allow_html=True)