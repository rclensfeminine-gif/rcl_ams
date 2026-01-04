import streamlit as st

def custom_sidebar_menu():
    """
    Crée et affiche le menu de navigation personnalisé (GPS) dans la barre latérale.
    Cette fonction doit être appelée au début de chaque page pour assurer la persistance du menu.
    """
    
    # --- Lien de Retour à la Page d'Accueil (Placé en haut pour la visibilité) ---
    # CORRECTION IMPORTANTE: Le chemin doit être le nom de fichier de la page principale
    # si elle est à la racine (ex: 'Accueil.py' ou 'Home.py').
    st.sidebar.page_link("Accueil.py", label="🏠 Accueil", icon=None, use_container_width=True)

    # --- Le Menu Déroulant GPS ---
    st.sidebar.header("Menu")
    with st.sidebar.expander("🛰️ GPS"):
        
        # Les chemins vers les pages dans le dossier 'pages/' doivent commencer par 'pages/'
        
        # 1. Sous-section GPS Groupe
        st.page_link("pages/1_GPS_groupe.py", label="GPS - Groupe", icon=None, use_container_width=True)
        
        # 2. Sous-section GPS Individuel
        st.page_link("pages/2_GPS_indiv.py", label="GPS - Indiv", icon=None, use_container_width=True)

        # 3. Sous-section GPS Match
        st.page_link("pages/3_GPS_match.py", label="GPS - Match", icon=None, use_container_width=True)
            
        # 4. Sous-section GPS Statistiques
        st.page_link("pages/4_GPS_statistiques.py", label="GPS - Statistiques", icon=None, use_container_width=True)

    # --- Le Menu Déroulant Joueuse ---
    with st.sidebar.expander("🏃 Joueuse"):
        
        # Les chemins vers les pages dans le dossier 'pages/' doivent commencer par 'pages/'
        
        # 1. Sous-section GPS Groupe
        st.page_link("pages/7_Identité.py", label="Identité", icon=None, use_container_width=True)
        
        # 2. Sous-section GPS Individuel
        st.page_link("pages/8_Suivi_joueuse.py", label="Suivi joueuse", icon=None, use_container_width=True)

        # 3. Sous-section GPS Match
        st.page_link("pages/9_Fiche_joueuse.py", label="Fiche joueuse", icon=None, use_container_width=True)

    # --- Le Menu Déroulant Etat de forme ---
    with st.sidebar.expander("⚡ Etat de forme"):
        
        # Les chemins vers les pages dans le dossier 'pages/' doivent commencer par 'pages/'
        
        # 1. Sous-section GPS Groupe
        st.page_link("pages/10_Wellness.py", label="Wellness", icon=None, use_container_width=True)

    # --- Le Menu Déroulant Organisation ---
    with st.sidebar.expander("📅 Organisation"):
        
        # Les chemins vers les pages dans le dossier 'pages/' doivent commencer par 'pages/'
        
        # 1. Sous-section Planning
        st.page_link("pages/5_Planning.py", label="Planning", icon=None, use_container_width=True)

        # 2. Sous-section Planning indiv
        st.page_link("pages/6_Planning_individuel.py", label="Planning individuel", icon=None, use_container_width=True)

    
    # --- Le Menu Déroulant Kiné ---
    with st.sidebar.expander("🩺 Kiné"):
        
        # Les chemins vers les pages dans le dossier 'pages/' doivent commencer par 'pages/'
        
        # 1. Sous-section GPS Groupe
        st.page_link("pages/11_Suivi_médical.py", label="Suivi médical", icon=None, use_container_width=True)
        
        # 2. Sous-section GPS Individuel
        st.page_link("pages/12_Transmission_médical.py", label="Transmission médical", icon=None, use_container_width=True)