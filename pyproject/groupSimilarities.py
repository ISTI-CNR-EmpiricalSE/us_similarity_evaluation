from anaconda_project.project_ops import download
import tensorflow as tf
import tensorflow_hub as hub

from pyproject.misure import jaccard_similarity, cosine_distance_countvectorizer_method, bert, wordMover_word2vec, \
    euclidean, lsi, universal_sentence_encoder, universal_sentence_encoder_2param
from sklearn.metrics.pairwise import cosine_similarity
from pyproject.utili import sortTriple, transform, preprocessing

from nltk import download

import nltk
from nltk.corpus import stopwords
from gensim import models, corpora, similarities

import os


def group_score(first, second, misura, flag_pre):
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

        stop_words = stopwords.words('english')

        if not os.path.exists('fileUtili/GoogleNews-vectors-negative300.bin.gz'):
            raise ValueError("SKIP: You need to download the google news model")

        model_word2vec = models.keyedvectors.KeyedVectors.load_word2vec_format(
            'fileUtili/GoogleNews-vectors-negative300.bin.gz', binary=True)
        for n in range(0, len(first_set)):
            for m in range(0, len(second_set)):
                complete_list.append((n, m,
                                      wordMover_word2vec(first_set[n], second_set[m], model_word2vec,
                                                         stop_words)))

    elif misura == "euclidean" or misura == "euclidean_preProcessed":

        tok_sentences = []
        for sentence in second_set:
            tok_sentences.append(nltk.word_tokenize(sentence))
        second_transformed_us = transform(tok_sentences)

        tok_sentences = []
        for sentence in first_set:
            tok_sentences.append(nltk.word_tokenize(sentence))
        first_transformed_us = transform(tok_sentences)

        for i in range(len(first_transformed_us)):
            score_list = euclidean(i, second_transformed_us)
            for tripla in score_list:
                complete_list.append(tripla)

    elif misura == "universal_sentence_encoder" or misura == "universal_sentence_encoder_preProcessed":
        module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
        model = hub.load(module_url)
        complete_list = universal_sentence_encoder_2param(first_set, second_set, model)

    return complete_list


def aggregate_score(first, second, misura, flag_pre):
    if flag_pre:
        first = preprocessing(first)
        second = preprocessing(second)
        misura = misura + "_preProcessed"

    if misura == "cosine_vectorizer" or misura == "cosine_vectorizer_preProcessed":
        return cosine_distance_countvectorizer_method(first, second) / 100

    elif misura == "jaccard" or misura == "jaccard_preProcessed":
        return jaccard_similarity(first, second)

    elif misura == "wordMover_word2vec" or misura == "wordMover_word2vec_preProcessed":

        stop_words = stopwords.words('english')

        if not os.path.exists('fileUtili/GoogleNews-vectors-negative300.bin.gz'):
            raise ValueError("SKIP: You need to download the google news model")

        model_word2vec = models.keyedvectors.KeyedVectors.load_word2vec_format(
            'fileUtili/GoogleNews-vectors-negative300.bin.gz', binary=True)

        return wordMover_word2vec(first, second, model_word2vec, stop_words)

    elif misura == "euclidean" or misura == "euclidean_preProcessed":

        tok_sentences = []
        tok_sentences.append(nltk.word_tokenize(second))
        second_transformed_us = transform(tok_sentences)

        tok_sentences = []
        tok_sentences.append(nltk.word_tokenize(first))
        first_transformed_us = transform(tok_sentences)

        return euclidean(first, second)

    elif misura == "universal_sentence_encoder" or misura == "universal_sentence_encoder_preProcessed":
        module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
        model = hub.load(module_url)
        score = universal_sentence_encoder_2param([first], [second], model)
        return score[0]


def misuraMax(first, second, misura, flag_pre):

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
    score_list = group_score(first, second, misura, flag_pre)

    somma = 0
    for tripla in score_list:
        somma = somma + tripla[2]

    avg = somma / len(score_list)
    return avg


def misuraAggregate(first, second, misura, flag_pre):

    first_set = ""
    second_set = ""
    for sentence in first:
        first_set = first_set + sentence
    for sentence in second:
        second_set = second_set + sentence

    return aggregate_score(first_set, second_set, misura, flag_pre)
