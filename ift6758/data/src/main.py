import requests
import json
import os

class SaisonHockey:
    BASE_URL = "https://statsapi.web.nhl.com/api/v1/game/{}/feed/live/"

    def __init__(self, annee_debut):
        self.annee_debut = annee_debut
        self.annee_fin = annee_debut + 1
        self.data = []
        
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
                            ids.append(f"{self.annee_debut}{typ}0{round_num}{matchup}{game}")

    def fetch_data(self):
        """ Récupère les données play-by-play de chaque jeu pour la saison spécifiée lors de l'initialisation 
            de l'instance. 
        
        Les données sont récupérées depuis l'API de la LNH en utilisant l'URL de base stockée dans BASE_URL. 
        Pour chaque identifiant de jeu généré par la méthode game_ids, la fonction effectue une requête GET 
        pour obtenir les données. Si la requête est réussie (code de statut 200), les données play-by-play 
        sont extraites du JSON et ajoutées à l'attribut 'data' de l'instance.
        
        :rtype: None
        """
        for game_id in self.game_ids():
            response = requests.get(self.BASE_URL.format(game_id))
            if response.status_code == 200:
                game_data = response.json() 
                play_by_play_data = game_data.get('liveData', {}).get('plays', {}).get('allPlays', [])
                self.data.append(play_by_play_data)

    def save_data(self, path):
        """ Enregistre les données play-by-play stockées dans l'attribut 'data' de l'instance 
            sous forme de fichier JSON.

        :param path: Chemin du fichier dans lequel les données seront enregistrées.
        :type path: str
        :rtype: None
        """
        with open(path, 'w') as f:
            json.dump(self.data, f)

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
        file_path = f"../ift6758/data/data_saison_{saison.annee_debut}_{saison.annee_fin}_play_by_play.json"
        
        # Vérifie si un fichier pour la saison en cours existe déjà.
        if os.path.isfile(file_path):
            print(f"The file '{file_path}' already exists.")
        else:
            # Si non, récupère les données et sauvegarde dans le fichier.
            print(f"fetching data for {saison.annee_debut}_{saison.annee_fin} ..")
            saison.fetch_data()
            print(f"saving data for {saison.annee_debut}_{saison.annee_fin} ..")
            saison.save_data(file_path)

    # Construit le chemin du fichier pour sauvegarder toutes les données combinées.
    total_data_file_path = "../ift6758/data/data_total_play_by_play.json"
    
    # Vérifie si un fichier pour les données combinées existe déjà.
    if os.path.isfile(total_data_file_path):
        print(f"The file '{total_data_file_path}' already exists.")
    else:
        # Si non, combine toutes les données des saisons et sauvegarde dans le fichier.
        total_data = sum(saisons, SaisonHockey(2016))
        total_data.save_data(total_data_file_path)

# Exécute la fonction collect_data si ce script est exécuté en tant que programme principal.
if __name__ == "__main__":
    collect_data()
