from pyproject.misure import jaccard_similarity, cosine_distance_countvectorizer_method, bert, wordMover_word2vec, \
    euclidean, lsi
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
import pandas as pd
import re
import seaborn as sns
from absl import logging
import tensorflow as tf

import tensorflow_hub as hub


# ordina coppie (us, valore)
def sortTriple(list):
    list.sort(key=lambda x: x[2], reverse=True)
    return list


# controllo esistenza df del file
def backup(fileName):
    if not fileName + ".pkl" in os.listdir("out"):
        print("creazione Data frame")
        dfGen(fileName)
    else:
        print("file esistente")


# transform for euclidean
def transform(sentences):
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
def confronto(args):
    file = args.usFile
    misura = args.misura
    backup(file)
    with open('out/' + file + '.pkl', 'rb') as dfl:
        df = pickle.load(dfl)

    userStories = df["userStory"].tolist()

    if misura == "cosine_vectorizer":
        if "cosine_vectorizer" in df.columns:
            return df[["userStory", "cosine_vectorizer"]]
        complete_list = []
        for n in range(0, len(userStories)):
            sentence_sim_list = []
            for m in range(0, len(userStories)):
                sentence_sim_list.append((n, m,
                                          cosine_distance_countvectorizer_method(userStories[n], userStories[m]) / 100))
            sentence_sim_list = sortTriple(sentence_sim_list)
            complete_list.append(sentence_sim_list)
        df["cosine_vectorizer"] = complete_list

    elif misura == "jaccard":
        if "jaccard" in df.columns:
            return df[["userStory", "jaccard"]]
        complete_list = []
        for n in range(0, len(userStories)):
            sentence_sim_list = []
            for m in range(0, len(userStories)):
                sentence_sim_list.append((n, m,
                                          jaccard_similarity(userStories[n], userStories[m])))
            sentence_sim_list = sortTriple(sentence_sim_list)
            complete_list.append(sentence_sim_list)
        df["jaccard"] = complete_list

    elif misura == "wordMover_word2vec":
        if "wordMover_word2vec" in df.columns:
            return df[["userStory", "wordMover_word2vec"]]

        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
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
        df["wordMover_word2vec"] = complete_list

    elif misura == "bert_cosine":
        if "bert_cosine" in df.columns:
            return df[["userStory", "bert_cosine"]]
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

        df["bert_cosine"] = complete_list

    elif misura == "euclidean":
        if "euclidean" in df.columns:
            return df[["userStory", "euclidean"]]

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

        df["euclidean"] = complete_list

    elif misura == "lsi_cosine":
        if "lsi_cosine" in df.columns:
            return df[["userStory", "lsi_cosine"]]

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

        df["lsi_cosine"] = score_list

    elif misura == "universal_sentence_encoder":
        if "universal_sentence_encoder" in df.columns:
            return df[["userStory", "universal_sentence_encoder"]]

        module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
        model = hub.load(module_url)

        # embeddings:
        def embed(sentence):
            return model(sentence)

        embedded_sentences = embed(userStories)

        total_list = []
        for first in range(0, len(embedded_sentences)):
            sim_list = []
            for other in range(0, len(embedded_sentences)):
                tripla = (first, other, np.inner(embedded_sentences[first],
                                                 embedded_sentences[other]))
                sim_list.append(tripla)
            sim_list = sortTriple(sim_list)
            total_list.append(sim_list)

        df["universal_sentence_encoder"] = total_list

    # salvataggio
    with open('out/' + file + '.pkl', 'wb') as dfl:
        pickle.dump(df, dfl)

    return df[["userStory", misura]]


# us più simili tra loro (5 valori più alti)
def most_similar(args):
    df = confronto(args)
    misura = args.misura

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

    return "done"


# heatmap misura
def heatmap(args):
    complete_df = confronto(args)
    misura = args.misura

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
