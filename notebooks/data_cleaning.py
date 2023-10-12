import json
import pandas as pd
import ast

def create_event_df(data):
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

    print(event_df)
    event_df.to_csv("../ift6758/data_clean/eventdf_clean.csv", index=False)


def main():

    print("READING JSON")
    json_path = "../ift6758/data/data_total_play_by_play.json"

    data = pd.read_json(json_path)
    create_event_df(data)

    """

    def create_faceoff_table(data):
        print("FACEOFF TABLE")
        faceoff_data = []
        for event in data[0]:
            if event["result"]["eventTypeId"] == "FACEOFF":
                print(event["result"])

    create_faceoff_table(df)
    #df.tocsv("/data_clean/")
    """

if __name__ == "__main__":
    main()