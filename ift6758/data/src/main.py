import requests
import json
import os
import threading
import time
import pandas as pd

class SaisonHockey:
    BASE_URL = "https://statsapi.web.nhl.com/api/v1/game/{}/feed/live/"

    def __init__(self, annee_debut):
        self.annee_debut = annee_debut
        self.annee_fin = annee_debut + 1
        self.data = []
        self.lock = threading.Lock()  # Ajout du verrou
        
    def game_ids(self):
        """ Retourne les GAME_ID des données play-by-play de la LNH pour la saison régulière et les séries 
            éliminatoires concerant les matchs de la saison 2016-17 jusqu'à la saison 2020-21
            
        Les identifiants sont générés en fonction de:
        - '02' pour la saison régulière
        - '03' pour les séries éliminatoires

        :rtype: list
        :return: Liste des GAME_ID pour la saison spécifiée
        """
        ids = []
        for typ in ['02', '03']:  # 02 = saison régulière, 03 = séries éliminatoires
            # Générer les identifiants pour la saison régulière
            if typ == '02':
                limit = 1271 if self.annee_fin > 2017 else 1230
                for game_num in range(1, limit + 1):
                    ids.append(f"{self.annee_debut}{typ}{game_num:04}")
            # Générer les identifiants pour les séries éliminatoires
            else:
                for round_num in range(1, 5):  # 4 rounds
                    for matchup in range(1, 9):  # 8 matchups maximum
                        for game in range(1, 8):  # 7 games maximum
                            ids.append(f"{self.annee_debut}{typ}{game}{matchup}{round_num}0")
        return ids


    def fetch_data(self):
        MAX_THREADS = 10
        game_ids = self.game_ids()
        threads = []
        for game_id in game_ids:
            if len(threads) >= MAX_THREADS:
                for thread in threads:
                    thread.join()
                threads = []
            thread = threading.Thread(target=self._fetch_single_game_data, args=(game_id,))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

    def _fetch_single_game_data(self, game_id):
        for _ in range(3):  # Essayez 3 fois
            try:
                response = requests.get(self.BASE_URL.format(game_id))
                if response.status_code == 200:
                    game_data = response.json()
                    play_by_play_data = game_data.get('liveData', {}).get('plays', {}).get('allPlays', [])
                    with self.lock:
                        self.data.append({"gameID": game_id, "data": play_by_play_data})
                break
            except requests.exceptions.RequestException:
                time.sleep(5)

    def save_data(self, base_path):
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        dfs = []  # Pour stocker chaque DataFrame temporaire
        for game in self.data:
            game_id = game["gameID"]
            game_data = game["data"]
            game_path = f"{base_path}/{game_id}.json"
            with open(game_path, 'w') as f:
                json.dump(game_data, f, indent=4)


    def __add__(self, autre_saison):
        """ Fusionne les données de deux saisons et renvoie une nouvelle instance de SaisonHockey  
            avec les données fusionnées.

        :param autre_saison: Une autre instance de SaisonHockey dont les données doivent être fusionnées.
        :type autre_saison: SaisonHockey
        :rtype: SaisonHockey
        :return: Une nouvelle instance de SaisonHockey contenant les données des deux saisons.
        """
        merged = SaisonHockey(self.annee_debut)
        merged.data = self.data + autre_saison.data
        return merged

def collect_data(start=2016, end=2021):
    """ Vérifie la validité des années fournies. Si l'année de début est supérieure à l'année de fin, 
        une erreur est signalée.

    :param start: Année de début de la plage de saisons. Par défaut, elle est définie sur 2016.
    :type start: int
    :param end: Année de fin de la plage de saisons. Par défaut, elle est définie sur 2021.
    :type end: int
    :rtype: None
    """
    # Vérification de la validité de la plage d'années fournie.
    if start > end:
        print("Error : Start is bigger than end ...")
        print("Ending Code ...")
        return

    # Création d'une liste d'objets SaisonHockey pour chaque année dans la plage spécifiée.
    saisons = [SaisonHockey(annee) for annee in range(start, end)]

    # Pour chaque objet saison dans la liste des saisons:
    for saison in saisons:
        # Construit le chemin du fichier de données pour la saison en cours.
        file_path = f"C:/Users/lebou/Desktop/Cours_Canada/Data_science_IFT6758/project_nhl_data/ift6758-A08-1/ift6758/data/nhl_data/{saison.annee_debut}_{saison.annee_fin}"
        
        # Vérifie si un fichier pour la saison en cours existe déjà.
        if os.path.isfile(file_path):
            print(f"The file '{file_path}' already exists.")
        else:
            # Si non, récupère les données et sauvegarde dans le fichier.
            print(f"fetching data for {saison.annee_debut}_{saison.annee_fin} ..")
            saison.fetch_data()
            print(f"saving data for {saison.annee_debut}_{saison.annee_fin} ..")
            saison.save_data(file_path)
            
# Exécute la fonction collect_data si ce script est exécuté en tant que programme principal.
if __name__ == "__main__":
    collect_data()
