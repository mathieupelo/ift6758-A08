import requests
import json

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

def main():
    saisons = [SaisonHockey(annee) for annee in range(2016, 2021)]
    
    for saison in saisons:
        saison.fetch_data()
        saison.save_data(f"data/data_saison_{saison.annee_debut}_{saison.annee_fin}_play_by_play.json")

    total_data = sum(saisons, SaisonHockey(2016))
    total_data.save_data("data/data_total_play_by_play.json")

if __name__ == "__main__":
    main()
