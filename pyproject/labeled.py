import os
import pickle

from sklearn.metrics.pairwise import cosine_similarity

from pyproject.misure import cosine_distance_countvectorizer_method, jaccard_similarity, wordMover_word2vec, euclidean, \
    universal_sentence_encoder_2param, bert
from pyproject.utili import loadModelUSE
import pandas as pd
import nltk
from nltk.corpus import stopwords
from gensim import models

if not os.path.exists('fileUtili/GoogleNews-vectors-negative300.bin.gz'):
    raise ValueError("SKIP: You need to download the google news model")

from pyproject.utili import preprocessing, transform

modelUSE = loadModelUSE()


def score(first, second, misura, flag_pre):
    """
    :param first: string
    :param second: string list
    :param misura: string
    :param flag_pre: boolean
    :return: lista di confronti
    """
    sentence = first
    second_set = second

    if flag_pre:
        sentence = str(preprocessing([sentence]))
        set = preprocessing(second_set)
        second_set = []
        for sent in set:
            second_set.append(str(sent))

    complete_list = []
    if misura == "cosine_vectorizer" or misura == "cosine_vectorizer_preProcessed":
        for m in range(0, len(second_set)):
            complete_list.append((m, cosine_distance_countvectorizer_method(sentence, second_set[m]) / 100))

    elif misura == "jaccard" or misura == "jaccard_preProcessed":
        for m in range(0, len(second_set)):
            complete_list.append((m, jaccard_similarity(sentence, second_set[m])))

    elif misura == "wordMover_word2vec" or misura == "wordMover_word2vec_preProcessed":
        model_word2vec = models.keyedvectors.KeyedVectors.load_word2vec_format(
            'fileUtili/GoogleNews-vectors-negative300.bin.gz', binary=True)
        stop_words = stopwords.words('english')

        for m in range(0, len(second_set)):
            complete_list.append((m, wordMover_word2vec(sentence, second_set[m], model_word2vec,
                                                        stop_words)))

    elif misura == "euclidean" or misura == "euclidean_preProcessed":

        all_set = [sentence]

        for sent in second_set:
            all_set.append(sent)

        tok_sentences = []
        for sent in all_set:
            tok_sentences.append(nltk.word_tokenize(sent))

        transformed_us = transform(tok_sentences)

        complete_list = []
        score_list = [euclidean(0, transformed_us)]

        for lista in score_list:
            for tripla in lista:
                if tripla[1] != 0:
                    complete_list.append([tripla[1], tripla[2]])
        return complete_list

    elif misura == "universal_sentence_encoder" or \
            misura == "universal_sentence_encoder_preProcessed":

        triple_list = universal_sentence_encoder_2param([sentence], second_set, modelUSE)
        for tripla in triple_list:
            complete_list.append([tripla[1], tripla[2]])

    elif misura == "bert_cosine" or misura == "bert_cosine_preProcessed":

        bert_vects_first_set = bert([sentence])
        bert_vects_second_set = bert(second_set)
        cosSim_list = []

        for sent in bert_vects_first_set:
            cosSim = cosine_similarity([sent], bert_vects_second_set[0:])
            cosSim_list.append(cosSim[0])

        for m in range(0, len(second_set)):
            coppia = (m, cosSim_list[0][m])
            complete_list.append(coppia)

    return complete_list


def ret_label(label_list):
    """
    returns the only element in lable_list != NaN
    :param label_list: list
    :return: string
    """
    for label in label_list:
        if str(label) != "nan":
            return label


def excel_to_dataframe(fileName):
    """
    creates a dataframe from excel file
    :param fileName: string
    :return: dataframe [USER STORY][LABEL]
    """

    if fileName + '.pkl' in os.listdir("out"):
        with open('out/' + fileName + '.pkl', 'rb') as dfl:
            df = pickle.load(dfl)
            return df

    tempDf = pd.read_excel("Data/Archive/" + fileName, sheet_name="Labeled User Stories")
    label_list = []
    for n in range(0, len(tempDf)):
        label = ret_label([tempDf["Administrative and management activities related to camps' company (customer)"][n],
                           tempDf["Administrative and management activities related to camps' company  (facilities)"][
                               n],
                           tempDf["Administrative and management activities related to camps' company (personel)"][n],
                           tempDf["Managing individual camps (everyone perspective)"][n],
                           tempDf["Communication (including social media, feedback and emergency)"][n],
                           tempDf["New Feature"][n]])
        label_list.append(label)

    df = pd.DataFrame(columns=["USER STORY"])
    df["USER STORY"] = tempDf["USER STORY"]
    df["LABEL"] = label_list

    # salvataggio
    with open('out/' + fileName + '.pkl', 'wb') as dfl:
        pickle.dump(df, dfl)

    return df


def original_us_dataframe():
    """
    the dataframe contains the original userStories
    :param: fileName: string
    :return: dataframe [0][1][2][complete]
    """
    df = pd.read_excel("Data/Archive/user-story-original.xlsx", index_col=None, header=None)

    for n in range(0, len(df)):
        if str(df[2][n]) == "nan":
            df[2][n] = ""

    df["complete"] = df[0] + df[1] + df[2]

    # salvataggio
    with open('out/user-story-original.xlsx.pkl', 'wb') as dfl:
        pickle.dump(df, dfl)
    return df


