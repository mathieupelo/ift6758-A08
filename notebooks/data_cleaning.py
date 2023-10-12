import json
import os

import pandas as pd
import ast

import requests

def remove_missing_values(df):
	df.dropna(inplace=True)
	return df

def remove_duplicate_values(df):
	df = df.drop_duplicates()
	return df

def create_event_df(data):
    if os.path.isfile("../ift6758/data_clean/eventdf_clean.csv"):
        print(f"The file ../ift6758/data_clean/eventdf_clean.csv already exists.")
        return
    # We create an empty DataFrame
    columns = ['event', 'eventCode', 'eventTypeId', 'description', 'eventIdx', 'eventId', 'period', 'periodType',
               'ordinalNum', 'periodTime', 'periodTimeRemaining', 'dateTime', 'goalsHome', 'goalsAway']
    event_df = pd.DataFrame(columns=columns)

    # We create new entries
    # new_row = {'Column1': 'Value1', 'Column2': 'Value2'}

    new_rows = []
    # for every entry in df
    for entry in data:
        for line in data[entry]:
            try:
                event = line['result']['event']
                eventCode = line['result']['eventCode']
                eventTypeId = line['result']['eventTypeId']
                description = line['result']['description']
                eventIdx = line['about']['eventIdx']
                eventId = line['about']['eventId']
                period = line['about']['period']
                periodType = line['about']['periodType']
                ordinalNum = line['about']['ordinalNum']
                periodTime = line['about']['periodTime']
                periodTimeRemaining = line['about']['periodTimeRemaining']
                dateTime = line['about']['dateTime']
                goalsHome = line['about']['goals']['home']
                goalsAway = line['about']['goals']['away']

                new_row = {'event': event, 'eventCode': eventCode, 'eventTypeId': eventTypeId,
                           'description': description, 'eventIdx': eventIdx, 'eventId': eventId,
                           'period': period, 'periodType': periodType, 'ordinalNum': ordinalNum,
                           'periodTime': periodTime, 'periodTimeRemaining': periodTimeRemaining,
                           'dateTime': dateTime, 'goalsHome': goalsHome, 'goalsAway': goalsAway
                           }
                new_rows.append(new_row)
            except:
                pass

    event_df = pd.concat([event_df, pd.DataFrame(new_rows)], ignore_index=True)
    event_df = remove_missing_values(event_df)
    event_df = remove_duplicate_values(event_df)
    event_df.to_csv("../ift6758/data_clean/eventdf_clean.csv", index=False)

def create_faceoff_df(data):
    if os.path.isfile("../ift6758/data_clean/faceoffdf_clean.csv"):
        print(f"The file ../ift6758/data_clean/faceoffdf_clean.csv already exists.")
        return

    # We create an empty DataFrame
    columns = ['event']
    faceoff_df = pd.DataFrame(columns=columns)

    # We create new entries
    # new_row = {'Column1': 'Value1', 'Column2': 'Value2'}

    new_rows = []
    # for every entry in df
    for entry in data:
        for line in data[entry]:
            try:
                if line["result"]["eventTypeId"] == "FACEOFF":
                    event = line['result']['event']
                    eventCode = line['result']['eventCode']
                    eventTypeId = line['result']['eventTypeId']
                    description = line['result']['description']
                    dateTime = line['about']['dateTime']
                    period = line['about']['period']
                    periodType = line['about']['periodType']
                    goalsHome = line['about']['goals']['home']
                    goalsAway = line['about']['goals']['away']
                    winner = line['players'][0]["player"]["fullName"]
                    loser = line['players'][1]["player"]["fullName"]
                    new_row = {'event': event, 'eventCode':eventCode, 'eventTypeId': eventTypeId,
                               'description':description, 'dateTime': dateTime, 'period': period,
                               'periodType': periodType, 'goalsHome': goalsHome, 'goalsAway': goalsAway,
                               'winner': winner, 'loser': loser}

                    new_rows.append(new_row)
            except:
                pass

    faceoff_df = pd.concat([faceoff_df, pd.DataFrame(new_rows)], ignore_index=True)
    faceoff_df = remove_missing_values(faceoff_df)
    faceoff_df = remove_duplicate_values(faceoff_df)
    faceoff_df.to_csv("../ift6758/data_clean/faceoffdf_clean.csv", index=False)

def fetch_player_info(link):
    BASE_URL = 'https://statsapi.web.nhl.com'
    url = BASE_URL + link
    response = requests.get(url)
    if response.status_code == 200:
        print("Response Received...")
        print(response)

        # Or access the response content as JSON (if it's JSON)
        json_data = response.json()

        birthDate = json_data['people'][0]['id']
        height = json_data['people'][0]['height']
        weight = json_data['people'][0]['weight']
        primaryPositionCode = json_data['people'][0]['primaryPosition']['code']
        primaryPositionName = json_data['people'][0]['primaryPosition']['name']

        return birthDate, height, weight, primaryPositionCode, primaryPositionName

def create_player_df(data, fetch_info=0):
    if os.path.isfile("../ift6758/data_clean/playerdf_clean.csv"):
        print(f"The file ../ift6758/data_clean/playerdf_clean.csv already exists.")
        return
    # We create an empty DataFrame
    columns = ['playerid', 'fullName', 'link']
    player_df = pd.DataFrame(columns=columns)

    # We create new entries
    # new_row = {'Column1': 'Value1', 'Column2': 'Value2'}

    new_rows = []
    # for every entry in df
    for entry in data:
        for line in data[entry]:
            try:
                for player in line['players']:
                    playerid = player['player']['id']
                    fullName = player['player']['fullName']
                    link = player['player']['link']
                    if fetch_info == 1:
                        birthDate, height, weight, primaryPositionCode, primaryPositionName = fetch_player_info(link)

                        new_row = {'playerid': playerid, 'fullName' : fullName, 'link': link,
                                   'birthDate': birthDate, 'height': height, 'weight': weight,
                                   'primaryPositionCode': primaryPositionCode, 'primaryPositionName': primaryPositionName}
                    else:
                        new_row = {'playerid': playerid, 'fullName': fullName, 'link': link}
                    new_rows.append(new_row)
            except:
                pass

    player_df = pd.concat([player_df, pd.DataFrame(new_rows)], ignore_index=True)
    player_df = remove_missing_values(player_df)
    player_df = remove_duplicate_values(player_df)
    player_df.to_csv("../ift6758/data_clean/playerdf_clean.csv", index=False)


def create_shoot_df(data, fetch_info=0):
    if os.path.isfile("../ift6758/data_clean/shootdf_clean.csv"):
        print(f"The file ../ift6758/data_clean/shootdf_clean.csv already exists.")
        return
    # We create an empty DataFrame
    columns = []
    shoot_df = pd.DataFrame(columns=columns)

    # We create new entries
    # new_row = {'Column1': 'Value1', 'Column2': 'Value2'}

    new_rows = []
    # for every entry in df
    for entry in data:
        for line in data[entry]:
            try:
                if line["result"]["eventTypeId"] == "SHOT":
                    event = line['result']['event']
                    eventCode = line['result']['eventCode']
                    eventTypeId = line['result']['eventTypeId']
                    description = line['result']['description']
                    secondaryType = line['result']['secondaryType']

                    dateTime = line['about']['dateTime']
                    period = line['about']['period']
                    periodType = line['about']['periodType']
                    periodTime = line['about']['periodTime']
                    periodTimeRemaining = line['about']['periodTimeRemaining']
                    goalsHome = line['about']['goals']['home']
                    goalsAway = line['about']['goals']['away']

                    new_row = {'event': event, 'eventCode': eventCode, 'eventTypeId': eventTypeId,
                               'description': description, 'secondaryType': secondaryType,
                               'dateTime': dateTime, 'period': period, 'periodType': periodType,
                               'periodTime': periodTime, 'periodTimeRemaining': periodTimeRemaining,
                               'goalsHome': goalsHome, 'goalsAway': goalsAway}

                    new_rows.append(new_row)
            except:
                pass

    shoot_df = pd.concat([shoot_df, pd.DataFrame(new_rows)], ignore_index=True)
    shoot_df = remove_missing_values(shoot_df)
    shoot_df = remove_duplicate_values(shoot_df)
    shoot_df.to_csv("../ift6758/data_clean/shootdf_clean.csv", index=False)


def main():

    print("READING JSON")
    json_path = "../ift6758/data/data_total_play_by_play.json"

    data = pd.read_json(json_path)

    create_event_df(data)
    create_faceoff_df(data)
    create_player_df(data)
    create_shoot_df(data)

if __name__ == "__main__":
    main()