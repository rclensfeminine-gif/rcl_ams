import streamlit as st
import pandas as pd
import os
import json
import re
import uuid
from datetime import timedelta, datetime

# Chemins de fichiers
PLANNING_DIR = "data/planning"
WEEK_NAMES_FILE = os.path.join(PLANNING_DIR, "week_names.json")

# --- Configuration des chemins ---
FILE_PATH_PLANNING = "data/planning/"
FILE_PATH_PLANNING_INDIV = "data/planning_indiv/"
FILE_PATH_SEMAINE = "data/planning/week_names.json" 
FILE_PATH_IDENTITE = "data/identite.csv"

# --- CONSTANTES ---
JOURS_SEMAINE = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
PERIODES_JOURNEE_INDIV = ["Matin (8h-12h)", "Midi (12h-14h)", "Après-midi (14h-20h)"] 
PERIODES_KEYS = [p.split('(')[0].strip() for p in PERIODES_JOURNEE_INDIV] 
PERIODES_JOURNEE_SHORT = [p.split('(')[0].strip() for p in PERIODES_JOURNEE_INDIV] 
TYPES_RDV = ["Soin", "Récup", "RDV Doc", "Diététicien", "Entretient", "Prépa Mental", "Testing", "Autre"]

# Créer le dossier s'il n'existe pas
if not os.path.exists(PLANNING_DIR):
    os.makedirs(PLANNING_DIR)

# --- Fonctions de gestion ---
def get_monday(selected_date):
    """Calcule le lundi de la semaine pour une date donnée."""
    start_of_week = selected_date - timedelta(days=selected_date.weekday())
    return start_of_week

def load_week_names():
    """Charge les noms de semaines depuis un fichier JSON."""
    if os.path.exists(WEEK_NAMES_FILE):
        with open(WEEK_NAMES_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_week_names(names):
    """Sauvegarde les noms de semaines dans un fichier JSON."""
    with open(WEEK_NAMES_FILE, 'w') as f:
        json.dump(names, f)

def load_planning_data(filepath):
    """Charge les données du planning depuis un fichier CSV."""
    try:
        df = pd.read_csv(filepath, index_col=0)
        return df
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()
    
# --- FONCTIONS DE CHARGEMENT DE BASE ---
def save_planning_data(df, filepath):
    """Sauvegarde le DataFrame du planning dans un fichier CSV."""
    df.to_csv(filepath)

def get_match_color(text):
    """
    Applique un style CSS basé sur le contenu de la cellule.
    Fonctionne pour les contenus complets (semaine/mois) et les contenus nettoyés (annuel).
    
    Recherche : 
    1. Matchs : Mot-clé + Chiffre (ex: J4, CL2, CDF1, Amical 3, Prépa 1)
    2. Entraînement : Mot-clé ou Abréviation + Chiffre (ex: Séance 3, S3, Muscu 1, M1, Compensation 2, C2)
    3. Récupération : Mot-clé "Récup"
    """
    if not text:
        return ""

    text = str(text).lower().strip()
    
    # -------------------------------------------------------------
    # 1. Détection des Matchs et Compétitions (Priorité aux mots-clés + nombre)
    # Recherche : [J/CL/CDF/Amical/Prépa] + nombre
    # -------------------------------------------------------------
    
    # Vert le plus clair (Amical/Prépa)
    # Cherche 'amical' ou 'prépa' suivi d'un espace/un point ou un chiffre (pour couvrir 'Amical 1' ou juste 'Amical')
    if re.search(r'(amical|prépa)[\s\.]?\d*', text):
        return "background-color: #90EE90; color: #000000;"
    
    # Vert le plus foncé pour J (Championnat)
    # Cherche 'j' suivi d'un espace/un point ou un chiffre (pour couvrir 'J4' ou 'J.4')
    if re.search(r'j[\s\.]?\d+', text):
        return "background-color: #006400; color: white;"
    
    # Vert foncé (Coupe de la Ligue)
    # Cherche 'cl' suivi d'un espace/un point ou un chiffre
    if re.search(r'cl[\s\.]?(\d+|finale)', text):
        return "background-color: #008000; color: white;"
    
    # Vert moyen (Coupe de France)
    # Cherche 'cdf' suivi d'un espace/un point ou un chiffre
    if re.search(r'cdf[\s\.]?(\d+|finale)', text):
        return "background-color: #2E8B57; color: white;"

    # -------------------------------------------------------------
    # 2. Détection des Entraînements (Mot-clé + nombre OU Abréviation + nombre)
    # -------------------------------------------------------------

    # Jaune/Or (Séance / S# / Compensation / C#)
    # Cherche 'séance' + nombre OU 's' + nombre OU 'compensation' + nombre OU 'c' + nombre
    is_seance_or_comp = (
        re.search(r'séance[\s\.]?\d+', text) or 
        re.search(r's\s*\d+', text) or
        re.search(r'compensation[\s\.]?\d*', text) or # compensation n'a pas toujours de chiffre
        re.search(r'c\s*\d+', text)
    )
    
    if is_seance_or_comp:
        return "background-color: #FFD700; color: #000000;"
    
    # Rose (Muscu / M#)
    # Cherche 'muscu' + nombre OU 'm' + nombre
    is_muscu = (
        re.search(r'muscu[\s\.]?\d+', text) or
        re.search(r'm\s*\d+', text)
    )
    
    if is_muscu:
        return "background-color: #FFC0CB; color: #000000;"

    # -------------------------------------------------------------
    # 3. Détection de la Récupération
    # -------------------------------------------------------------

    # Bleu clair (Récupération)
    if "récup" in text or "recup" in text:
        return "background-color: #99c2ff; color: #000000;"
    
    # Si aucune correspondance spécifique, retourne une chaîne vide
    return ""