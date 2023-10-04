import pandas as pd
import ast



def main():
    print("READING JSON")
    csv_path = "../ift6758/data/data_saison_2016_2017_play_by_play.json"
    df = pd.read_json(csv_path)
    print(df)


    df['event'] = df[0].apply(lambda x: x['result']['event'])
    df['eventCode'] = df[0].apply(lambda x: x['result']['eventCode'])
    df['eventTypeId'] = df[0].apply(lambda x: x['result']['eventTypeId'])
    df['description'] = df[0].apply(lambda x: x['result']['description'])

    df["eventIdx"] = df[0].apply(lambda x: x["about"]["eventIdx"])
    df["eventId"] = df[0].apply(lambda x: x["about"]["eventId"])

    df['period'] = df[0].apply(lambda x: x['about']['period'])
    df['periodType'] = df[0].apply(lambda x: x['about']['periodType'])

    df['ordinalNum'] = df[0].apply(lambda x: x['about']['ordinalNum'])

    df['periodTime'] = df[0].apply(lambda x: x['about']['periodTime'])
    df['periodTimeRemaining'] = df[0].apply(lambda x: x['about']['periodTimeRemaining'])
    df['dateTime'] = df[0].apply(lambda x: x['about']['dateTime'])
    df['goals_away'] = df[0].apply(lambda x: x['about']['goals']['away'])
    df['goals_home'] = df[0].apply(lambda x: x['about']['goals']['home'])

    #df['coordinatesX'] = df[0].apply(lambda x: x['coordinates']['x'])
    #df['coordinatesY'] = df[0].apply(lambda x: x['coordinates']['y'])


    df = df[['event', 'eventCode', 'eventTypeId', 'description', 'eventIdx', 'eventId','period', 'periodType', 'periodTime',
             'periodTimeRemaining', 'dateTime', 'goals_away', 'goals_home']]
    print(df.columns)
    print(df)



if __name__ == "__main__":
    main()