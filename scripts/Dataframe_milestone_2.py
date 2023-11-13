import pandas as pd
import math

# Fonction pour calculer la distance euclidienne entre deux points
def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def find_previous_event(play_data, current_index, current_period, current_period_time):
    if current_index == 0:
        return None
    
    prev_event = play_data[current_index - 1]
    prev_event_period = prev_event['about']['period']
    prev_event_period_time = prev_event['about']['periodTime']

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

def handle_goal_event(play_data, current_index, power_play_status, scoring_team_id):
    """
    Gère les événements de but pour vérifier et annuler les pénalités mineures.
    """

    conceding_team = 'home_team' if scoring_team_id != play_data[current_index]['team']['id'] else 'away_team'

    if power_play_status[conceding_team]:
        power_play_status[conceding_team].sort(key=lambda x: x['start_time'])

        for penalty in power_play_status[conceding_team]:
            if penalty['duration'] == 120:  # 2 minutes en secondes
                power_play_status[conceding_team].remove(penalty)
                break


def update_power_play_status(play_data, current_index, power_play_status, home_team_id, away_team_id):
    current_event = play_data[current_index]
    event_type = current_event['result']['eventTypeId']
    
    current_period_time = int(current_event['about']['periodTime'].split(':')[0]) * 60 + int(current_event['about']['periodTime'].split(':')[1])

    if event_type == "GOAL" and 'team' in current_event:
        scoring_team_id = current_event['team']['id']
        handle_goal_event(play_data, current_index, power_play_status, scoring_team_id)

    if event_type == 'PENALTY' and 'team' in current_event:
        team_id = current_event['team']['id']
        penalty_duration = current_event['result']['penaltyMinutes']
        penalized_team = 'home_team' if team_id == home_team_id else 'away_team'

        start_time = int(current_period_time)
        penalty_duration = current_event['result']['penaltyMinutes']

        if power_play_status[penalized_team] is None:
            power_play_status[penalized_team] = [{'start_time': start_time, 'duration': penalty_duration * 60}]
        else:
            power_play_status[penalized_team].append({'start_time': start_time, 'duration': penalty_duration * 60})
            
    for team, penalties in power_play_status.items():
        if penalties:
            power_play_status[team] = [p for p in penalties if current_period_time - p['start_time'] < p['duration']]
    

    

def calculate_skater_count(power_play_status, home_team_name, away_team_name):
    standard_skater_count = 5
    home_team_skaters = standard_skater_count
    away_team_skaters = standard_skater_count

    for team_name, penalties in power_play_status.items():
        if penalties:  # Si l'équipe a des pénalités
            penalty_count = len(penalties)
            if team_name == 'home_team':
                home_team_skaters = max(3, standard_skater_count - penalty_count)
            else:
                away_team_skaters = max(3, standard_skater_count - penalty_count)
                
    print(f"Nombre de joueurs après calcul: Maison - {home_team_skaters}, Visiteur - {away_team_skaters}")


    return {
        home_team_name: home_team_skaters,
        away_team_name: away_team_skaters
    }
    
    
def transformEventData(df: pd.DataFrame) -> pd.DataFrame:
  
    temp_data = {
        'gameId': [], 'evt_idx': [], 'prd': [], 'prdTime': [], 'team': [],
        'goalFlag': [], 'shotCategory': [], 'coord_x': [], 'coord_y': [],
        'shotBy': [], 'goalieName': [], 'noGoalie': [], 'teamStrength': [],
        'visitorTeam': [], 'hostTeam': [], 'homeRinkSide': [], 'awayRinkSide': [],
        # Ces caractéristiques sont pour l'ingénierie des caractéristiques 2 du Milestone 2
        'last_event_type': [], 'last_event_x': [], 'last_event_y': [],
        'time_since_last_event': [], 'distance_from_last_event': [],
        'power_play_time_elapsed': [], 'home_team_skater_count': [],
        'away_team_skater_count': []
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
            
        home_team_name = game_details['gameData']['teams']['home']['name']
        away_team_name = game_details['gameData']['teams']['away']['name']
        power_play_status = {'home_team': None, 'away_team': None}

        for event_index, single_event in enumerate(play_data):
            
            period_time_str = single_event['about']['periodTime']
            minutes, seconds = map(int, period_time_str.split(':'))
            period_time= int(minutes * 60 + seconds)

            update_power_play_status(play_data, event_index, power_play_status, home_team_name, away_team_name)

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
                    
            elapsed_time = 0
            for status in power_play_status.values():
                if status:
                    for penalty in status:
                        elapsed_time = max(elapsed_time, period_time - penalty['start_time'])
            temp_data['power_play_time_elapsed'].append(elapsed_time)

            skater_count = calculate_skater_count(power_play_status, home_team_name, away_team_name)
            temp_data['home_team_skater_count'].append(skater_count[home_team_name])
            temp_data['away_team_skater_count'].append(skater_count[away_team_name])
            
    output_df = pd.DataFrame(temp_data)
    output_df.to_csv('../data/derivatives/dataframe_milestone_2.csv', index=False)
    return output_df