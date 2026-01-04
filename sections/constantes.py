##Remplacement des noms de colonne

DICT_REPLACE_COLS = {
    "Acceleration B2-3 Total Efforts (Gen 2)": "Accel >2m.s²",
    "Deceleration B2-3 Total Efforts (Gen 2)": "Decel >2m.s²",
    "Maximum Velocity (km/h)": "V max"
    }

LIST_ORGANISATION_COLS_MATCH = [
    "Player Name",
    "Position Name",
    "Activity Name",
    "Mi-temps",
    "Field Time",
    "Total Distance (m)",
    "VHSR Total Distance (m)",
    "SPR Total Distance (m)",
    "SPR + Total Distance (m)",
    "Sprint (m)",
    "VHSR + SPR effort",
    "Sprint effort",
    "Accel >2m.s²",
    "Decel >2m.s²",
    "V max",
    "Meterage Per Minute"
]

LIST_ORGANISATION_COLS_SEANCE = [
    'Activity Name',
    'Date',
    'Jour semaine',
    'Total Time', 
    'Field Time',
    'Total Distance (m)', 
    'HSR Total distance (m)',
    'VHSR Total Distance (m)', 
    'SPR Total Distance (m)',
    'SPR + Total Distance (m)', 
    'VHSR + SPR effort', 
    'VHSR effort',
    'Sprint effort', 
    'Accel >2m.s²', 
    'Decel >2m.s²', 
    'V max',
    'Meterage Per Minute',
    'Cardio',
    'Muscu',
    's.Cardio',
    's.Muscu'
]

LIST_ORGANISATION_COLS_INDIV = [
    'Name',
    'Activity Name',
    'Date',
    'Jour semaine',
    'Total Time', 
    'Field Time',
    'Total Distance (m)', 
    'HSR Total distance (m)',
    'VHSR Total Distance (m)', 
    'SPR Total Distance (m)',
    'SPR + Total Distance (m)', 
    'VHSR + SPR effort', 
    'VHSR effort',
    'Sprint effort', 
    'Accel >2m.s²', 
    'Decel >2m.s²', 
    'V max',
    'Meterage Per Minute', 
    'Cardio',
    'Muscu',
    's.Cardio',
    's.Muscu',
    'Presence'
]

ONGLET_GPS_SAISON = ["2025-26", "2024-25", "2023-24"]

ONGLET_GPS_TYPE_MATCH = ["championnat", "coupe", "prepa", "amical"]

ONGLET_GPS_TYPE = ["S", "C", "M", "Réa"]

cols_ref_match = [
    "Player Name",
    "Position Name",
    "Activity Name",
    "Type match",
    "Saison",
    "Field Time",
    "Mi-temps",
    "Total Distance (m)",
    "VHSR Total Distance (m)",
    "SPR Total Distance (m)",
    "SPR + Total Distance (m)",
    "Sprint (m)",
    "VHSR + SPR effort",
    "Sprint effort",
    "Accel >2m.s²",
    "Decel >2m.s²",
    "V max",
    "Meterage Per Minute"
]

cols_ref_match_indiv = [
    "Position Name",
    "Activity Name",
    "Type match",
    "Saison",
    "Field Time",
    "Mi-temps",
    "Total Distance (m)",
    "VHSR Total Distance (m)",
    "Sprint (m)",
    "VHSR effort",
    "Sprint effort",
    "Accel >2m.s²",
    "Decel >2m.s²",
    "V max",
    "Meterage Per Minute"
]

cols_ref_seance = [
    "Name",
    "Position Name",
    "Activity Name",
    "Jour semaine",
    'Total Time',
    "Saison",
    "Field Time",
    "Total Distance (m)",
    'HSR Total distance (m)',
    "VHSR Total Distance (m)",
    "SPR Total Distance (m)",
    "SPR + Total Distance (m)",
    "VHSR + SPR effort",
    "VHSR effort",
    "Sprint effort",
    "Accel >2m.s²",
    "Decel >2m.s²",
    "V max",
    "Meterage Per Minute"
]

cols_ref_avec_mi_temps = [
    "Player Name",
    "Position Name",
    "Activity Name",
    "Type match",
    "Saison",
    "Field Time",
    "Mi-temps",
    "Total Distance (m)",
    "VHSR Total Distance (m)",
    "SPR Total Distance (m)",
    "SPR + Total Distance (m)",
    "Sprint (m)",
    "VHSR + SPR effort",
    "Sprint effort",
    "Accel >2m.s²",
    "Decel >2m.s²",
    "V max",
    "Meterage Per Minute"
]

cols_num = [
    "Total Time",
    "Field Time",
    "Total Distance (m)",
    'HSR Total distance (m)',
    "VHSR Total Distance (m)",
    "SPR Total Distance (m)",
    "SPR + Total Distance (m)",
    "Sprint (m)",
    "VHSR effort",
    "Sprint effort",
    "Accel >2m.s²",
    "Decel >2m.s²",
    "V max",
    "Meterage Per Minute", 
    "Cardio",
    "Muscu",
    "s.Cardio",
    "s.Muscu"
]

cols_de_regroupement = [
    'Player Name', 
    'Activity Name', 
    'Position Name', 
    'Type match', 
    'Saison'
]

cols_de_regroupement_indiv = [
    'Activity Name', 
    'Position Name', 
    'Type match', 
    'Saison'
]

cols_de_regroupement_seance = [ 
    'Activity Name', 
    'Position Name', 
    'Jour semaine',
    'Date', 
    'Saison'
]

cols_de_regroupement_mi_temps = [
    'Player Name', 
    'Activity Name', 
    'Position Name', 
    'Type match', 
    'Saison',
    'Mi-temps'
]

cols_de_regroupement_match = [
    'Activity Name', 
    'Type match', 
    'Saison'
]

marqueur_volume = [
    "Total Distance (m)"
]

marqueur_intensite = [
    "VHSR Total Distance (m)",
    "SPR Total Distance (m)",
    "SPR + Total Distance (m)",
    "Sprint (m)",
    "VHSR + SPR effort",
    "Sprint effort"
]

marqueur_musculaire = [
    "Accel >2m.s²",
    "Decel >2m.s²"
]

marqueurs = marqueur_volume + marqueur_intensite + marqueur_musculaire + ['V max'] +["Meterage Per Minute"]
agg_dict = {col: 'max' for col in marqueurs}

cols_joueuses = [
    'Madelynn ANDERSON',
    'Louann ARCHIER',
    'Clara BERTRAND',
    'Ambre BOUCHARD',
    'Tess DAVID',
    'Fatima EL GHAZOUANI',
    'Kaina EL KOUMIR',
    'Julia EVRARD',
    'Aude GBEDJISSI',
    'Sherly JEUDY',
    'Emmy JEZEQUEL',
    'Blandine JOLY',
    'Lara KAZANDJIAN',
    'Romane LEJEUNE',
    'Jennifer LIMAGE',
    'Alizée MEREAU',
    'Jennifer MEUNIER',
    'Lizzy MILLEQUANT',
    'Laureen OILLIC',
    'Dayana PIERRE LOUIS',
    'Laurine PINOT',
    'Carla POLITO',
    'Fany PRONIEZ',
    'Manon REVELLI',
    'Emma SMAALI',
    'Naomie VAGRE',
    'Sofia GUELLATI',
    'Célia REMILI'
]

agg_dict_seance = {
    'Total Time': 'mean',
    "Field Time": 'mean',
    "Total Distance (m)": 'mean',
    'HSR Total distance (m)': 'mean',
    "VHSR Total Distance (m)": 'mean',
    "SPR Total Distance (m)": 'mean',
    "SPR + Total Distance (m)": 'mean',
    "VHSR + SPR effort": 'mean',
    "VHSR effort": 'mean',
    "Sprint effort": 'mean',
    "Accel >2m.s²": 'mean',
    "Decel >2m.s²": 'mean',
    "V max": 'mean',
    "Meterage Per Minute": 'mean',
    'Cardio': 'mean',
    'Muscu': 'mean',
    's.Cardio': 'mean',
    's.Muscu': 'mean'
}

agg_dict_semaine = {
    'Total Time': 'sum',
    "Field Time": 'sum',
    "Total Distance (m)": 'sum',
    'HSR Total distance (m)': 'sum',
    "VHSR Total Distance (m)": 'sum',
    "SPR Total Distance (m)": 'sum',
    "SPR + Total Distance (m)": 'sum',
    "VHSR + SPR effort": 'sum',
    "VHSR effort": 'sum',
    "Sprint effort": 'sum',
    "Accel >2m.s²": 'sum',
    "Decel >2m.s²": 'sum',
    "V max": 'max',
    "Meterage Per Minute": 'mean',
    'Cardio': 'sum',
    'Muscu': 'sum',
    's.Cardio': 'sum',
    's.Muscu': 'sum'
}

agg_dict_indiv = {
    'Total Time': 'sum',
    "Field Time": 'sum',
    "Total Distance (m)": 'sum',
    'HSR Total distance (m)': 'sum',
    "VHSR Total Distance (m)": 'sum',
    "SPR Total Distance (m)": 'sum',
    "SPR + Total Distance (m)": 'sum',
    "VHSR + SPR effort": 'sum',
    "VHSR effort": 'sum',
    "Sprint effort": 'sum',
    "Accel >2m.s²": 'sum',
    "Decel >2m.s²": 'sum',
    "V max": 'max',
    "s.Cardio": 'sum',
    "s.Muscu": 'sum'
}

cols_cumul_sum = [
    "Total Time",
    "Field Time",
    "Total Distance (m)",
    'HSR Total distance (m)',
    "VHSR Total Distance (m)",
    "SPR Total Distance (m)",
    "SPR + Total Distance (m)",
    "Sprint (m)",
    "VHSR effort",
    "Sprint effort",
    "Accel >2m.s²",
    "Decel >2m.s²",
    "s.Cardio",
    "s.Muscu"
]

cols_stat = [
    "Total Time",
    "Field Time",
    "Total Distance (m)",
    'HSR Total distance (m)',
    "VHSR Total Distance (m)",
    "SPR Total Distance (m)",
    "SPR + Total Distance (m)",
    "VHSR effort",
    "Sprint effort",
    "Accel >2m.s²",
    "Decel >2m.s²",
    "s.Cardio",
    "s.Muscu"
]