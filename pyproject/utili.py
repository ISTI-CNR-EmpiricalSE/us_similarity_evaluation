from pyproject.misure import jaccard_similarity, cosine_distance_countvectorizer_method, bert, wordMover_word2vec, \
    euclidean, lsi, universal_sentence_encoder
from sklearn.metrics.pairwise import cosine_similarity
from pyproject.generazioneDf import dfGen
import os
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

import logging

from nltk import download
import numpy as np
import sklearn

import pandas as pd
import nltk
from nltk.corpus import stopwords
from gensim import models, corpora, similarities

import matplotlib.pyplot as plt
import numpy as np
import os



def preprocessing(userStories):
    """
        removes from repetitive sequences user stories
    :param userStories: string list
    :return: string list: modified user story list
    """
    words_to_remove = ["as a ", "as an ", "as "
                                          "i want to be able to ", "i want to ", "i want ", "i only want ",
                       "i would like to ", "i would like a ", "i would be able to ", "i'm able to ",
                       "so that i can ", "so that i ", "so that ", "so "]
    new_userStories = []
    for sentence in userStories:
        sentence = sentence.lower()
        for w in words_to_remove:
            sentence = sentence.replace(w, '')
        new_userStories.append(sentence)

    return new_userStories


def sortTriple(list):
    """
    sorts the list by value in descending order
    :param list: list of triples (us1, us2, value)
    :return:
    """

    list.sort(key=lambda x: x[2], reverse=True)
    return list


def backup(fileName):
    """
    checks if the file (fileName) already has a dataframe
    :param fileName: string
    """

    if not fileName + ".pkl" in os.listdir("out"):
        print("creazione Data frame")
        dfGen(fileName)
    else:
        print("file esistente")


# transform for euclidean
def transform(sentences):
    """
        encodes and tokenizes sentences
    :param sentences: string list
    """

    tokens = [w for s in sentences for w in s]

    results = []
    label_enc = sklearn.preprocessing.LabelEncoder()
    onehot_enc = sklearn.preprocessing.OneHotEncoder()

    encoded_all_tokens = label_enc.fit_transform(list(set(tokens)))
    encoded_all_tokens = encoded_all_tokens.reshape(len(encoded_all_tokens), 1)

    onehot_enc.fit(encoded_all_tokens)

    for sentence in sentences:
        encoded_words = label_enc.transform(sentence)
        encoded_words = onehot_enc.transform(
            encoded_words.reshape(len(encoded_words), 1))

        results.append(np.sum(encoded_words.toarray(), axis=0))
    return results


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

    if misura == "cosine_vectorizer":
        if flag_pre:
            userStories = preprocessing(userStories)
            misura = misura + '_preProcessed'
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

    elif misura == "jaccard":
        if flag_pre:
            userStories = preprocessing(userStories)
            misura = misura + '_preProcessed'
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

    elif misura == "wordMover_word2vec":
        if flag_pre:
            userStories = preprocessing(userStories)
            misura = misura + '_preProcessed'
        if misura in df.columns:
            return df[["userStory", misura]]

        # Import and download stopwords from NLTK.

        download('stopwords')  # Download stopwords list.
        stop_words = stopwords.words('english')

        if not os.path.exists('Data/GoogleNews-vectors-negative300.bin.gz'):
            raise ValueError("SKIP: You need to download the google news model")

        model_word2vec = models.keyedvectors.KeyedVectors.load_word2vec_format(
            'Data/GoogleNews-vectors-negative300.bin.gz', binary=True)
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

    elif misura == "bert_cosine":
        if flag_pre:
            userStories = preprocessing(userStories)
            misura = misura + '_preProcessed'
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

    elif misura == "euclidean":
        if flag_pre:
            userStories = preprocessing(userStories)
            misura = misura + '_preProcessed'
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

    elif misura == "lsi_cosine":
        if flag_pre:
            userStories = preprocessing(userStories)
            misura = misura + '_preProcessed'
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

    elif misura == "universal_sentence_encoder":
        if flag_pre:
            userStories = preprocessing(userStories)
            misura = misura + '_preProcessed'
        if misura in df.columns:
            return df[["userStory", misura]]

        score_list = universal_sentence_encoder(userStories)
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

    df = confronto(file, misura)

    if flag_pre:
        misura = misura + '_preProcessed'

    max_list = []

    val_list = df[misura].tolist()
    sorted_list = []

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


# heatmap misura
def heatmap(file, misura, flag_pre):
    """
    generate heatmap for the given similarity
    :param file: string
    :param misura: string
    :param flag_pre: default: False
    """
    complete_df = confronto(file, misura)

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
    plt.show()
    return "done"


# apllica tutte le misure sul file
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
               "euclidean", "lsi_cosine", "universal_sentence_encoder", "jaccard_preProcessed",
               "cosine_vectorizer_preProcessed", "bert_cosine_preProcessed", "wordMover_word2vec_preProcessed",
               "euclidean_preProcessed", "lsi_cosine_preProcessed", "universal_sentence_encoder_preProcessed"]

    colonne_df = df.columns

    for misura in colonne:
        if misura not in colonne_df:
            df = confronto(file, misura, False)
            df = confronto(file, misura, True)

    return df


def get_line(file, us):
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


def concat_all_dataframes():
    """
    creates a dataframe for every file in Data
    calculates all the similarities for the user stories in the filee
    and concatenates all the dataframes
    :return: dataframe
    """
    if not "all_dataframes.pkl" in os.listdir("out"):
        print("creazione Data frame")
        all = pd.DataFrame(columns=["userStory"])
    else:
        with open('out/all_dataframes.pkl', 'rb') as dfl:
            all = pickle.load(dfl)
            return all

    for file in os.listdir("Data"):
        if file != "GoogleNews-vectors-negative300.bin.gz":
            df = confronta_tutti(file)
            all = pd.concat([all, df])

    # salvataggio
    with open('out/all_dataframes.pkl', 'wb') as dfl:
        pickle.dump(all, dfl)

    return all
