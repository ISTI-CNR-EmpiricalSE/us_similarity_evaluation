import random

from pyproject.groupSimilarities import group_find_file
from pyproject.misure import jaccard_similarity, cosine_distance_countvectorizer_method, bert, wordMover_word2vec, \
    euclidean, lsi, universal_sentence_encoder
from sklearn.metrics.pairwise import cosine_similarity
from pyproject.utili import backup, sortTriple, preprocessing, transform, loadModelUSE
import pickle
import seaborn as sns

from nltk import download

import pandas as pd
import nltk
from nltk.corpus import stopwords
from gensim import models, corpora, similarities

import matplotlib.pyplot as plt
import os

if not os.path.exists('fileUtili/GoogleNews-vectors-negative300.bin.gz'):
    raise ValueError("SKIP: You need to download the google news model")

model_word2vec = models.keyedvectors.KeyedVectors.load_word2vec_format(
    'fileUtili/GoogleNews-vectors-negative300.bin.gz', binary=True)
modelUSE = loadModelUSE()


# calcolo misure
def confronto(file, misura, flag_pre):
    """
    calculates the given distance
    :param file: string
    :param misura: string
    :param flag_pre: boolean, default: False
    :return: dataframe[userStory, misura]
    """
    backup(file)
    with open('out/' + file + '.pkl', 'rb') as dfl:
        df = pickle.load(dfl)

    userStories = df["userStory"].tolist()
    if flag_pre:
        userStories = preprocessing(userStories)
        misura = misura + '_preProcessed'

    if misura == "cosine_vectorizer" or misura == "cosine_vectorizer_preProcessed":
        if misura in df.columns:
            return df[["userStory", misura]]
        complete_list = []
        for n in range(0, len(userStories)):
            sentence_sim_list = []
            for m in range(0, len(userStories)):
                sentence_sim_list.append((n, m,
                                          cosine_distance_countvectorizer_method(userStories[n], userStories[m]) / 100))
            sentence_sim_list = sortTriple(sentence_sim_list)
            complete_list.append(sentence_sim_list)
        df[misura] = complete_list

    elif misura == "jaccard" or misura == "jaccard_preProcessed":
        if misura in df.columns:
            return df[["userStory", misura]]

        complete_list = []
        for n in range(0, len(userStories)):
            sentence_sim_list = []
            for m in range(0, len(userStories)):
                sentence_sim_list.append((n, m,
                                          jaccard_similarity(userStories[n], userStories[m])))
            sentence_sim_list = sortTriple(sentence_sim_list)
            complete_list.append(sentence_sim_list)
        df[misura] = complete_list

    elif misura == "wordMover_word2vec" or misura == "wordMover_word2vec_preProcessed":
        if misura in df.columns:
            return df[["userStory", misura]]

        stop_words = stopwords.words('english')

        complete_list = []
        for n in range(0, len(userStories)):
            sentence_sim_list = []
            for m in range(0, len(userStories)):
                sentence_sim_list.append((n, m,
                                          wordMover_word2vec(userStories[n], userStories[m], model_word2vec,
                                                             stop_words)))
            sentence_sim_list = sortTriple(sentence_sim_list)
            sentence_sim_list = sentence_sim_list[::-1]
            complete_list.append(sentence_sim_list)
        df[misura] = complete_list

    elif misura == "bert_cosine" or misura == "bert_cosine_preProcessed":
        if misura in df.columns:
            return df[["userStory", misura]]

        bert_vects = bert(userStories)
        cosSim_list = []
        complete_list = []

        for first_sentence in bert_vects:
            cosSim = cosine_similarity([first_sentence], bert_vects[0:])
            cosSim_list.append(cosSim[0])

        for n in range(0, len(userStories)):
            lista = []
            for m in range(0, len(userStories)):
                tripla = (n, m, cosSim_list[n][m])
                lista.append(tripla)
            complete_list.append(sortTriple(lista))

        df[misura] = complete_list

    elif misura == "euclidean" or misura == "euclidean_preProcessed":
        if misura in df.columns:
            return df[["userStory", misura]]

        tok_sentences = []
        for sentence in userStories:
            tok_sentences.append(nltk.word_tokenize(sentence))

        transformed_us = transform(tok_sentences)

        complete_list = []
        for i in range(len(userStories)):
            score_list = euclidean(i, transformed_us)
            score_list = sortTriple(score_list)
            score_list = score_list[::-1]
            complete_list.append(score_list)

        df[misura] = complete_list

    elif misura == "lsi_cosine" or misura == "lsi_cosine_preProcessed":
        if misura in df.columns:
            return df[["userStory", misura]]

        complete_list = lsi(userStories)[0]

        # creo le triple
        score_list = []
        for n in range(0, len(complete_list)):
            triple_list = []
            for m in range(0, len(complete_list)):
                tripla = (n, m, complete_list[n][m])
                triple_list.append(tripla)
            triple_list = sortTriple(triple_list)
            score_list.append(triple_list)

        df[misura] = score_list

    elif misura == "universal_sentence_encoder" or misura == "universal_sentence_encoder_preProcessed":
        if misura in df.columns:
            return df[["userStory", misura]]

        score_list = universal_sentence_encoder(userStories, modelUSE())
        sorted_list = []
        for list in score_list:
            sorted_list.append(sortTriple(list))

        df[misura] = sorted_list

    # salvataggio
    with open('out/' + file + '.pkl', 'wb') as dfl:
        pickle.dump(df, dfl)

    return df


# us più simili tra loro (5 valori più alti)
def most_similar(file, misura, flag_pre):
    """
    returns the most similar user stories in the file, using 'misura'
    :param file: string
    :param misura: string
    :param flag_pre: default: False
    :return: list of triples (us1, us2, val)
    """

    df = confronto(file, misura, flag_pre)

    if flag_pre:
        misura = misura + '_preProcessed'

    max_list = []

    val_list = df[misura].tolist()

    # lista unica
    merged_list = []
    for list in val_list:
        merged_list = merged_list + list

    sorted_list = sortTriple(merged_list)

    if misura == "wordMover_word2vec":
        sorted_list = sorted_list[::-1]
    if misura == "euclidean":
        sorted_list = sorted_list[::-1]

    if len(sorted_list) < 6:
        max_list = sorted_list
        rem = max_list[0]
        max_list.remove(rem)

    else:
        max_val_list = []  # lista 5 valori massimi
        i = 0
        while len(max_val_list) < 5:
            if sorted_list[i][0] != sorted_list[i][1]:
                if sorted_list[i][2] not in max_val_list:
                    max_val_list.append(sorted_list[i][2])
                    max_list.append(sorted_list[i])
                else:
                    if (sorted_list[i][1], sorted_list[i][0], sorted_list[i][2]) not in max_list:
                        if sorted_list[i] not in max_list:
                            max_list.append(sorted_list[i])
            i = i + 1

        while sorted_list[i][2] in max_val_list:
            if sorted_list[i][0] != sorted_list[i][1]:
                if (sorted_list[i][1], sorted_list[i][0], sorted_list[i][2]) not in max_list:
                    if sorted_list[i] not in max_list:
                        max_list.append(sorted_list[i])
            i = i + 1

    userStories = df["userStory"].tolist()
    for tripla in max_list:
        prima = userStories[tripla[0]]
        seconda = userStories[tripla[1]]
        print(tripla)
        print(prima)
        print(seconda)

    return max_list


# dato file scelgo 2 requisiti random e ret le 2 us più simili
def find_most_similar(file):
    """
    returns the most similar user stories in the file, using 'misura'
    :param file: string
    :param misura: string
    :param flag_pre: default: False
    :return: 2 coppie di user story più simili
    """
    # 2 indici random
    df = confronta_tutti(file)
    ind1 = random.randint(0, len(df) - 1)
    ind2 = random.randint(0, len(df) - 1)
    for misura in ['jaccard', 'cosine_vectorizer', 'bert_cosine',
                   'wordMover_word2vec', 'euclidean', 'lsi_cosine', 'universal_sentence_encoder']:

        # uso sempre preprocessing
        misura = misura + '_preProcessed'

        ind1_list = df[misura][ind1]
        ind2_list = df[misura][ind2]

        print(misura)
        # caso con la us stessa come prima della lista
        if ind1_list[0][0] == ind1_list[0][1]:
            print(df['userStory'][ind1])
            ind = ind1_list[1][1]
            print(df['userStory'][ind])
        else:
            print(df['userStory'][ind1])
            ind = ind1_list[0][1]
            print(df['userStory'][ind])
        # caso con la us stessa come prima della lista
        if ind2_list[0][0] == ind2_list[0][1]:
            print(df['userStory'][ind2])
            ind = ind2_list[1][1]
            print(df['userStory'][ind])
        else:
            print(df['userStory'][ind2])
            ind = ind2_list[0][1]
            print(df['userStory'][ind])


# heatmap misura
def heatmap(file, misura, flag_pre):
    """
    generate heatmap for the given similarity
    :param file: string
    :param misura: string
    :param flag_pre: default: False
    """
    complete_df = confronto(file, misura, flag_pre)

    if flag_pre:
        misura = misura + '_preProcessed'

    # riordino i valori
    val_list = complete_df[misura].tolist()
    heat_list = []
    for list in val_list:
        new_list = []
        list.sort(key=lambda x: x[1])
        for tripla in list:
            new_list.append(tripla[2])
        heat_list.append(new_list)

    heat_df = pd.DataFrame(heat_list, columns=range(0, len(val_list)))

    heat = sns.heatmap(heat_df)
    plt.savefig('out/heatmap/' + file + misura + '.png')
    plt.clf()
    return


# applica tutte le misure sul file
def confronta_tutti(file):
    """
    caluclates all the similarities
    :param file: string
    :return: dataframe
    """
    backup(file)
    with open('out/' + file + '.pkl', 'rb') as dfl:
        df = pickle.load(dfl)

    colonne = ["jaccard", "cosine_vectorizer", "bert_cosine", "wordMover_word2vec",
               "euclidean", "lsi_cosine", "universal_sentence_encoder"]

    colonne_df = df.columns

    for misura in colonne:
        if misura not in colonne_df:
            confronto(file, misura, False)
            df = confronto(file, misura, True)
        if misura + '_preProcessed' not in colonne_df:
            confronto(file, misura, False)
            df = confronto(file, misura, True)

        heatmap(file, misura, True)
        heatmap(file, misura, False)

    return df


def get_line_byText(file, us):
    """
    returns the us's raw in the dataframe dataframe
    :param file: string
    :param us: string
    :return: list of pairs(similarity, ranked list)
        ranked list is a list of triples (us1, us2, value)
    """
    df = confronta_tutti(file)

    ranked_lists = []
    userStories = df["userStory"].tolist()
    ind = userStories.index(us)

    for col in df.columns:
        ranked = (col, df[col][ind])
        ranked_lists.append(ranked)

    return ranked_lists


def get_line_byId(file, us_id):
    """
    returns the us's raw in the dataframe dataframe
    :param file: string
    :param us_id: ind
    :return: list of pairs(similarity, ranked list)
        ranked list is a list of triples (us1, us2, value)
    """
    df = confronta_tutti(file)

    ranked_lists = []
    for col in df.columns:
        ranked = (col, df[col][us_id])
        ranked_lists.append(ranked)

    return ranked_lists


# creo dataframe unico con tutte le misure
def concat_all_dataframes():
    """
    creates a dataframe for every file in Data
    calculates all the similarities for the user stories in the filee
    and concatenates all the dataframes
    :return: dataframe
    """
    # backup
    if not "all_dataframes.pkl" in os.listdir("out"):
        print("creazione Data frame")
        all_df = pd.DataFrame(columns=["userStory"])
    else:
        with open('out/all_dataframes.pkl', 'rb') as dfl:
            all_df = pickle.load(dfl)
            return all_df

    for file in os.listdir("Data"):
        df = confronta_tutti(file)
        all_df = pd.concat([all_df, df])

    # salvataggio
    with open('out/all_dataframes.pkl', 'wb') as dfl:
        pickle.dump(all_df, dfl)

    return all_df


def find_file(n, file_us, k, group_fun, misura, flagPre):
    """
    :param n: int
    :param file_us: int
    :param k: int
    :param group_fun: string
    :param misura: string
    :param flagPre: boolean
    :return: dataframe
    """

    n_succ = 0
    n_fail = 0
    for i in range(0, n):
        result = group_find_file(k, file_us, group_fun, misura, flagPre)
        if result == "success":
            n_succ = n_succ + 1
        else:
            n_fail = n_fail + 1

    if flagPre:
        misura = misura + 'preProcessed'
    test_result = file_us + ' ' + misura + ' ' + group_fun + ' ' + ', k = ' + str(k) + ', successi = ' + str(
        n_succ) + ', tentativi =' + str(n) + '\n'
    test_result_file = open("out/group_test_result.txt", "a")
    test_result_file.write(str(test_result))
    test_result_file.close()
    print(test_result)


def find_file_test(file_us, group_fun):
    """
    calls find_file with k = 1,2,3,4,5
    :param file_us: int
    :param group_fun: string
    :param misura: string
    :param flagPre: boolean
    """
    for misura in ["jaccard", "cosine_vectorizer", "bert_cosine", "wordMover_word2vec",
                   "euclidean", "lsi_cosine"]:
        for i in range(1, 6):
            # n = numero di test
            n = 1
            find_file(n, file_us, i, group_fun, misura, False)
            find_file(n, file_us, i, group_fun, misura, True)
    return
