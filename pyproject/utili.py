from pyproject.generazioneDf import dfGen
import sklearn
import numpy as np
import os

import tensorflow as tf
import tensorflow_hub as hub


def preprocessing(userStories):
    """
    removes from repetitive sequences user stories
    :param userStories: string list
    :return: string list: modified user story list
    """
    words_to_remove = ["as a ", "as an ", "as "
                                          "i want to be able to ", "i want to ", "i want ", "i only want ",
                       "i would like to ", "i would like a ", "i would be able to ", "i'm able to ",
                       "i am able to ", "so that i can ", "so that i ", "so that ", "so "]
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
        dfGen(fileName)

    return


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


def sort_list(val_list):
    sorted_by_second = sorted(val_list, key=lambda tup: tup[1], reverse=True)
    return sorted_by_second


def loadModelUSE():
    module_url_USE = "https://tfhub.dev/google/universal-sentence-encoder/4"
    modelUSE = hub.load(module_url_USE)
    return modelUSE