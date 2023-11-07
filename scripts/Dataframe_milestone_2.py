import os
import json
import pandas as pd
from tqdm import tqdm
import math

def load_all_seasons(base_path):
    all_data = {}
    csv_path = 'all_seasons_data.csv'

    if os.path.exists(csv_path):
        print("Chargement des données depuis le fichier CSV existant.")
        return pd.read_csv(csv_path)
    
    seasons = [s for s in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, s))]

    for season in tqdm(seasons, desc='Seasons', position=0):
        season_path = os.path.join(base_path, season)
        json_files = [f for f in os.listdir(season_path) if f.endswith('.json')]

        for json_file in tqdm(json_files, desc=f'Loading {season}', position=1, leave=False):
            json_path = os.path.join(season_path, json_file)
            with open(json_path, 'r') as f:
                game_data = json.load(f)
                
            game_id = json_file.split('.')[0]
            all_data[game_id] = game_data
    
    df = pd.DataFrame.from_dict(all_data)
    return df

# Fonction pour calculer la distance euclidienne entre deux points
def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def find_previous_event(play_data, current_index, current_period, current_period_time):
    if current_index == 0:
        return None  # Pas d'événement précédent pour le premier événement
    
    prev_event = play_data[current_index - 1]
    prev_event_period = prev_event['about']['period']
    prev_event_period_time = prev_event['about']['periodTime']

    # Sacahnt que period_time est une chaîne de format "MM:SS"
    current_period_time_seconds = int(current_period_time.split(':')[0]) * 60 + int(current_period_time.split(':')[1])
    prev_event_period_time_seconds = int(prev_event_period_time.split(':')[0]) * 60 + int(prev_event_period_time.split(':')[1])

    # Calcul le temps écoulé entre les événements
    if current_period == prev_event_period:
        time_since_last_event = current_period_time_seconds - prev_event_period_time_seconds
    else:
        # Calcul le temps restant dans la période précédente et ajout le temps écoulé dans la période actuelle
        # Ceci suppose que chaque période est de 20 minutes. 
        # Peut-être à ajuster si nécessaire pour les périodes supplémentaires / prolongations.
        time_since_last_event = (20 * 60 - prev_event_period_time_seconds) + current_period_time_seconds

    prev_event_data = {
        'last_event_type': prev_event['result']['eventTypeId'],
        'last_event_x': prev_event['coordinates'].get('x', pd.NA),
        'last_event_y': prev_event['coordinates'].get('y', pd.NA),
        'time_since_last_event': time_since_last_event if time_since_last_event > 0 else pd.NA,
        'distance_from_last_event': pd.NA
    }
    return prev_event_data

# Ajout d'une nouvelle fonction pour identifier la dernière pénalité
def find_previous_penalty(play_data, current_index):
    for i in range(current_index - 1, -1, -1):  # Commence par l'événement précédent et remonter
        if play_data[i]['result']['eventTypeId'] == 'PENALTY':
            return play_data[i]
    return None  # Aucune pénalité trouvée


def transformEventData(df: pd.DataFrame) -> pd.DataFrame:
    temp_data = {
        'gameId': [], 'evt_idx': [], 'prd': [], 'prdTime': [], 'team': [],
        'goalFlag': [], 'shotCategory': [], 'coord_x': [], 'coord_y': [],
        'shotBy': [], 'goalieName': [], 'noGoalie': [], 'teamStrength': [],
        'visitorTeam': [], 'hostTeam': [], 'homeRinkSide': [], 'awayRinkSide': [],
        # Ces caractéristiques sont pour l'ingénierie des caractéristiques 2 du Milestone 2
        'last_event_type': [], 'last_event_x': [], 'last_event_y': [],
        'time_since_last_event': [], 'distance_from_last_event': []
    }
    
    
    temp_data['time_since_last_penalty'] = []

    for idx in range(df.shape[1]):
        play_data = df.iloc[:, idx]["liveData"]["plays"]["allPlays"]
        game_details = df.iloc[:, idx]
        
        period_details = game_details.get('liveData', {}).get('linescore', {}).get('periods', [])
        
        if period_details:
            first_period = period_details[0]
            home_rink_side = first_period.get('home', {}).get('rinkSide', 'N/A')
            away_rink_side = first_period.get('away', {}).get('rinkSide', 'N/A')
        else:
            home_rink_side = 'N/A'
            away_rink_side = 'N/A'
            
        for event_index, single_event in enumerate(play_data):
            evt_type = single_event['result']['eventTypeId']
            if evt_type not in ["SHOT", "GOAL"]:
                continue
            
            temp_data['gameId'].append(game_details.name)
            temp_data['evt_idx'].append(single_event['about']['eventIdx'])
            temp_data['prd'].append(single_event['about']['period'])
            temp_data['prdTime'].append(single_event['about']['periodTime'])
            temp_data['team'].append(single_event['team']['name'])
            temp_data['goalFlag'].append(evt_type == "GOAL")
            temp_data['shotCategory'].append(single_event['result'].get('secondaryType', pd.NA))
            temp_data['coord_x'].append(single_event['coordinates'].get('x', pd.NA))
            temp_data['coord_y'].append(single_event['coordinates'].get('y', pd.NA))
            temp_data['homeRinkSide'].append(home_rink_side)
            temp_data['awayRinkSide'].append(away_rink_side)
            
            str_code = 'NA' if evt_type == "SHOT" else single_event['result']['strength']['code']
            temp_data['teamStrength'].append(str_code)
            
            temp_data['visitorTeam'].append(game_details['gameData']['teams']['away']['name'])
            temp_data['hostTeam'].append(game_details['gameData']['teams']['home']['name'])
            
            player_data = single_event.get('players', [])
            shooter_counter, goalie_counter = 0, 0
            
            for player_entry in player_data:
                player_role = player_entry['playerType']
                
                if player_role in ['Scorer', 'Shooter']:
                    temp_data['shotBy'].append(player_entry['player']['fullName'])
                    shooter_counter += 1
                
                elif player_role == 'Goalie':
                    temp_data['goalieName'].append(player_entry['player']['fullName'])
                    goalie_counter += 1
            
            is_empty_net = single_event['result'].get('emptyNet', False)
            temp_data['noGoalie'].append(is_empty_net)
            
            if is_empty_net:
                temp_data['goalieName'].append("EmptyNet")
                goalie_counter += 1
            elif shooter_counter > 0 and goalie_counter == 0:
                temp_data['goalieName'].append(pd.NA)
                goalie_counter += 1
            
            if shooter_counter != goalie_counter:
                raise ValueError("Shooter count and goalie count do not match")
            
            # Ajout des données de l'événement précédent
            current_period = single_event['about']['period']
            current_period_time = single_event['about']['periodTime']
                        
            prev_event_data = find_previous_event(play_data, event_index, current_period, current_period_time)

            if prev_event_data:
                #coordonnées du tir actuel pour calculer la distance
                current_x = single_event['coordinates'].get('x', pd.NA)
                current_y = single_event['coordinates'].get('y', pd.NA)

                # Calcul des données temporelles et spatiales si les données sont complètes
                if prev_event_data['last_event_x'] is not pd.NA and prev_event_data['last_event_y'] is not pd.NA and current_x is not pd.NA and current_y is not pd.NA:
                    prev_event_data['distance_from_last_event'] = calculate_distance(
                        prev_event_data['last_event_x'], prev_event_data['last_event_y'],
                        current_x, current_y)
            
                for key in ['last_event_type', 'last_event_x', 'last_event_y', 'time_since_last_event', 'distance_from_last_event']:
                    temp_data[key].append(prev_event_data.get(key, pd.NA))
            else:
                # Si prev_event_data est None (c'est le premier événement), on ajoute des valeurs NA.
                for key in ['last_event_type', 'last_event_x', 'last_event_y', 'time_since_last_event', 'distance_from_last_event']:
                    temp_data[key].append(pd.NA)
    
    output_df = pd.DataFrame(temp_data)
    
    output_df.to_csv('../data/derivatives/dataframe_milestone_2.csv', index=False)
    return output_df