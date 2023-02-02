import pandas as pd
import pickle


def dfGen(file):
    """
    :param file: string
    :return: dataframe
    """
    # lista dataframe di tutti i file
    df = pd.DataFrame(columns=["userStory"])

    lines = open("Data/" + file, "r").readlines()
    for line in lines:
        line = line.strip()
        df = df.append({"userStory": line}, ignore_index=True)
    indexNames = df[(df['userStory'] == "")].index
    df.drop(indexNames, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # salvataggio
    with open('out/' + file + '.pkl', 'wb') as dfl:
        pickle.dump(df, dfl)

    return df
