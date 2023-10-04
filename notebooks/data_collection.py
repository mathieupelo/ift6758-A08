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
        ids = []
        for typ in ['02', '03']:  # 02 = saison régulière, 03 = séries éliminatoires
            if self.annee_fin == 2017:
                limit = 1230 if typ == '02' else 15
            else:
                limit = 1271 if typ == '02' else 15
            for game_num in range(1, limit + 1):
                ids.append(f"{self.annee_debut}{typ}{game_num:04}")
        return ids

    def fetch_data(self):
        for game_id in self.game_ids():
            response = requests.get(self.BASE_URL.format(game_id))
            if response.status_code == 200:
                game_data = response.json()
                play_by_play_data = game_data.get('liveData', {}).get('plays', {}).get('allPlays', [])
                self.data.append(play_by_play_data)

    def save_data(self, path):
        with open(path, 'w') as f:
            json.dump(self.data, f)

    def __add__(self, autre_saison):
        merged = SaisonHockey(self.annee_debut)
        merged.data = self.data + autre_saison.data
        return merged

def collect_data(start = 2016, end = 2021):
    if start > end :
        print("Error : Start is bigger than end ...")
        print("Ending Code ...")
        return

    saisons = [SaisonHockey(annee) for annee in range(start, end)]

    for saison in saisons:
        file_path = f"../ift6758/data/data_saison_{saison.annee_debut}_{saison.annee_fin}_play_by_play.json"
        if os.path.isfile(file_path):
            print(f"The file '{file_path}' already exists.")
        else :
            print(f"fetching data for {saison.annee_debut}_{saison.annee_fin} ..")
            saison.fetch_data()
            print(f"saving data for {saison.annee_debut}_{saison.annee_fin} ..")
            saison.save_data(file_path)

    total_data_file_path = "../ift6758/data/data_total_play_by_play.json"
    if os.path.isfile(total_data_file_path):
        print(f"The file '{total_data_file_path}' already exists.")
    else :
        total_data = sum(saisons, SaisonHockey(2016))
        total_data.save_data(total_data_file_path)

if __name__ == "__main__":
    collect_data()
