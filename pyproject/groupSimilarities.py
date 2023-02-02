from pyproject.misure import universal_sentence_encoder_2param, jaccard_similarity, \
    cosine_distance_countvectorizer_method, bert, wordMover_word2vec, \
    euclidean
from pyproject.utili import preprocessing, transform, loadModelUSE


from random import randint

from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from gensim import models, corpora, similarities

import os

if not os.path.exists('fileUtili/GoogleNews-vectors-negative300.bin.gz'):
    raise ValueError("SKIP: You need to download the google news model")
modelUSE = loadModelUSE()


def group_score(first, second, misura, flag_pre):
    """
    effettua il confronto tra le user story in first e quelle in second
    restituisce la lista ordinata per valore
    :param first: string list
    :param second: string list
    :param misura: string
    :param flag_pre: boolean
    :return: lista di triple (us1,us2,valore similarità tra us1 e us2)
    """
    first_set = first
    second_set = second

    if flag_pre:
        first_set = preprocessing(first_set)
        second_set = preprocessing(second_set)
        misura = misura + "_preProcessed"

    complete_list = []
    if misura == "cosine_vectorizer" or misura == "cosine_vectorizer_preProcessed":
        for n in range(0, len(first_set)):
            for m in range(0, len(second_set)):
                complete_list.append((n, m,
                                      cosine_distance_countvectorizer_method(first_set[n], second_set[m]) / 100))

    elif misura == "jaccard" or misura == "jaccard_preProcessed":
        for n in range(0, len(first_set)):
            for m in range(0, len(second_set)):
                complete_list.append((n, m,
                                      jaccard_similarity(first_set[n], second_set[m])))

    elif misura == "wordMover_word2vec" or misura == "wordMover_word2vec_preProcessed":
        model_word2vec = models.keyedvectors.KeyedVectors.load_word2vec_format(
            'fileUtili/GoogleNews-vectors-negative300.bin.gz', binary=True)
        stop_words = stopwords.words('english')

        for n in range(0, len(first_set)):
            for m in range(0, len(second_set)):
                complete_list.append((n, m,
                                      wordMover_word2vec(first_set[n], second_set[m], model_word2vec,
                                                         stop_words)))

    elif misura == "euclidean" or misura == "euclidean_preProcessed":

        k = len(first_set)

        all_set = []

        for sentence in first_set:
            all_set.append(sentence)
        for sentence in second_set:
            all_set.append(sentence)

        tok_sentences = []
        for sentence in all_set:
            tok_sentences.append(nltk.word_tokenize(sentence))

        transformed_us = transform(tok_sentences)

        complete_list = []
        score_list = []

        for i in range(0, k):
            score_list.append(euclidean(i, transformed_us))

        for list in score_list:
            for tripla in list:
                if tripla[1] not in range(0, k):
                    complete_list.append(tripla)
        return complete_list

    elif misura == "universal_sentence_encoder" or \
            misura == "universal_sentence_encoder_preProcessed":

        complete_list = universal_sentence_encoder_2param(first_set, second_set, modelUSE())

    elif misura == "bert_cosine" or misura == "bert_cosine_preProcessed":

        bert_vects_first_set = bert(first_set)
        bert_vects_second_set = bert(second_set)
        cosSim_list = []

        for sentence in bert_vects_first_set:
            cosSim = cosine_similarity([sentence], bert_vects_second_set[0:])
            cosSim_list.append(cosSim[0])

        complete_list = []
        for n in range(0, len(first_set)):
            for m in range(0, len(second_set)):
                complete_list.append([n, m, cosSim_list[n][m]])

    return complete_list


def aggregate_score(first, second, misura, flag_pre):
    """
    effettua il confronto tra  first e second (stringhe contenenti user story)
    restituisce il valore di similarità
    :param first: string
    :param second: string
    :param misura: string
    :param flag_pre: boolean
    :return: float
    """
    if flag_pre:
        first = str(preprocessing(first))
        second = str(preprocessing(second))
        misura = misura + "_preProcessed"

    if misura == "cosine_vectorizer" or misura == "cosine_vectorizer_preProcessed":
        return cosine_distance_countvectorizer_method(first, second) / 100

    elif misura == "jaccard" or misura == "jaccard_preProcessed":
        return jaccard_similarity(first, second)

    elif misura == "wordMover_word2vec" or misura == "wordMover_word2vec_preProcessed":
        model_word2vec = models.keyedvectors.KeyedVectors.load_word2vec_format(
            'fileUtili/GoogleNews-vectors-negative300.bin.gz', binary=True)
        stop_words = stopwords.words('english')

        return wordMover_word2vec(first, second, model_word2vec, stop_words)

    elif misura == "euclidean" or misura == "euclidean_preProcessed":

        all_set = [first, second]

        tok_sentences = []
        for sentence in all_set:
            tok_sentences.append(nltk.word_tokenize(sentence))

        transformed_us = transform(tok_sentences)

        res = euclidean(0, transformed_us)

        return res[1]

    elif misura == "universal_sentence_encoder" or misura == "universal_sentence_encoder_preProcessed":
        score = universal_sentence_encoder_2param([first], [second], modelUSE())
        return score[0]

    elif misura == "bert_cosine" or misura == "bert_cosine_preProcessed":

        bert_vects_first_set = bert(first)
        bert_vects_second_set = bert(second)

        cosSim = cosine_similarity([bert_vects_first_set], [bert_vects_second_set])

        return cosSim[0][0]


def misuraMax(first, second, misura, flag_pre):
    """
    restituisce il massimo valore di similarità dopo il confronto tra
    le user story in first e quelle in second
    :param first: string list
    :param second: string list
    :param misura: string
    :param flag_pre: boolean
    :return: float
    """
    score_list = group_score(first, second, misura, flag_pre)

    if misura == "wordMover_word2vec" or misura == "euclidean" \
            or misura == "wordMover_word2vec" or misura == "euclidean":
        max_val = 1000
        for tripla in score_list:
            if tripla[2] < max_val:
                max_val = tripla[2]
    else:
        max_val = 0
        for tripla in score_list:
            if tripla[2] > max_val:
                max_val = tripla[2]
    return max_val


def misuraAverage(first, second, misura, flag_pre):
    """
    restituisce il valore medio di similarità dopo il confronto tra
    le user story in first e quelle in second
    :param first: string list
    :param second: string list
    :param misura: string
    :param flag_pre: boolean
    :return: float
    """
    score_list = group_score(first, second, misura, flag_pre)

    somma = 0
    for tripla in score_list:
        somma = somma + tripla[2]

    avg = somma / len(score_list)
    return avg


def misuraAggregate(first, second, misura, flag_pre):
    """
    unifica le user story presenti in first in una unica string,
    fa lo stesso con second, restitusice aggregate_score
    :param first: string list
    :param second: string list
    :param misura:
    :param flag_pre:
    :return: float
    """
    first_set = ""
    second_set = ""
    for sentence in first:
        first_set = first_set + sentence
    for sentence in second:
        second_set = second_set + sentence

    return aggregate_score(first_set, second_set, misura, flag_pre)


def group_find_file(k, file_name, group_fun, misura, flagPre):
    """
    pops k user stories from a random file in Data
    applies misura using group_fun
    stores the result in a dataframe
    :param k: int
    :param file_name: string
    :param group_fun: string
    :param misura: string
    :param flagPre: boolean
    :return: dataframe
    """

    userStories = []
    lines = open("Data/txt_test/" + file_name, "r").readlines()
    for line in lines:
        if line != '\n':
            userStories.append(line)

    us_test = []
    for i in range(0, k):
        us_ind = randint(0, len(userStories) - 1)
        us_temp = userStories.pop(us_ind)
        us_test.append(us_temp)

    print("us test:")
    print(us_test)

    files = os.listdir("Data/txt_test")
    val_list = []
    # rimuovo dal file le US che sto testando
    for file in files:
        us = []
        lines = open("Data/txt_test/" + file, "r").readlines()
        for line in lines:
            if line != '\n':
                if file != file_name:
                    us.append(line)
                else:
                    if line not in us_test:
                        us.append(line)

        if group_fun == "max":
            val_list.append(misuraMax(us_test, us, misura, flagPre))
        if group_fun == "avg":
            val_list.append(misuraAverage(us_test, us, misura, flagPre))
        if group_fun == "aggr":
            val_list.append(misuraAggregate(us_test, us, misura, flagPre))

    print(val_list)
    if flagPre:
        misura = misura + "_preProcessed"
    if misura == "wordMover_word2vec" or misura == "euclidean" \
            or misura == "wordMover_word2vec_preProcessed" or misura == "euclidean_preProcessed":
        result = min(val_list)
    else:
        result = max(val_list)

    result_ind = val_list.index(result)
    result_file = files[result_ind]

    if result_file == file_name:
        result = "success"
    else:
        result = "fail"

    return result
