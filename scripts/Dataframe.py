import os
import json
import pandas as pd
from tqdm import tqdm

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

def transformEventData(df: pd.DataFrame) -> pd.DataFrame:
    temp_data = {
        'gameId': [], 'evt_idx': [], 'prd': [], 'prdTime': [], 'team': [],
        'goalFlag': [], 'shotCategory': [], 'coord_x': [], 'coord_y': [],
        'shotBy': [], 'goalieName': [], 'noGoalie': [], 'teamStrength': [],
        'visitorTeam': [], 'hostTeam': [], 'homeRinkSide': [], 'awayRinkSide': []
    }

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

        for single_event in play_data:
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
    
    output_df = pd.DataFrame(temp_data)
    output_df.to_csv('../data/derivatives/dataframe.csv', index=False)
    return output_df