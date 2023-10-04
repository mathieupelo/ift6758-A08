import pandas as pd




def main():
    print("READING JSON")
    csv_path = "../ift6758/data/data_saison_2016_2017_play_by_play.json"
    df = pd.read_json(csv_path)
    df.head()

    print(df)




if __name__ == "__main__":
    main()