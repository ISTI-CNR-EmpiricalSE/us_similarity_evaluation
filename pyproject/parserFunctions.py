import random

from matplotlib import pyplot as plt
from sklearn.metrics._classification import precision_recall_fscore_support

from pyproject.groupSimilarities import group_find_file
from pyproject.labeled import excel_to_dataframe, score
from pyproject.misure import jaccard_similarity, cosine_distance_countvectorizer_method, bert, wordMover_word2vec, \
    euclidean, lsi, universal_sentence_encoder
from sklearn.metrics.pairwise import cosine_similarity
from pyproject.utili import backup, sortTriple, preprocessing, transform, loadModelUSE, sort_list
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
    :param file_us: string
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
    :param file_us: string
    :param group_fun: string

    """
    for misura in ["jaccard", "cosine_vectorizer", "bert_cosine", "wordMover_word2vec",
                   "euclidean", "lsi_cosine"]:
        for i in range(1, 6):
            # n = numero di test
            n = 1
            find_file(n, file_us, i, group_fun, misura, False)
            find_file(n, file_us, i, group_fun, misura, True)
    return


def excel_confronto(fileName, misura, flagPre):
    if not fileName + ".pkl" in os.listdir("out"):
        df = excel_to_dataframe(fileName)
    else:
        with open('out/' + fileName + '.pkl', 'rb') as dfl:
            df = pickle.load(dfl)

    if flagPre:
        misura = misura + '_preProcessed'

    if misura in df.columns:
        return df

    with open('out/user-story-original.xlsx.pkl', 'rb') as dfl:
        original_df = pickle.load(dfl)
    original_us_list = original_df['complete']

    score_list = []

    for sentence in df["USER STORY"]:
        temp_list = score(sentence, original_us_list, misura, flagPre)
        temp_list = sort_list(temp_list)
        if misura == "wordMover_word2vec" or misura == "euclidean" or \
                misura == "wordMover_word2vec_preProcessed" or misura == "euclidean_preProcessed":
            temp_list.reverse()
        score_list.append(temp_list)

    df[misura] = score_list

    # salvataggio
    with open('out/' + fileName + '.pkl', 'wb') as dfl:
        pickle.dump(df, dfl)

    print(df)
    return df


def success_fail_onebyone(misura, falgPre):
    for file in os.listdir("Data/Archive"):
        if file != 'user-story-original.xlsx':
            success_fail_one_file(file, misura, falgPre)


def success_fail_all_files(misura, flagPre):
    df_list = []
    for file in os.listdir("Data/Archive"):
        if file != 'user-story-original.xlsx':
            df_list.append(excel_confronto(file, misura, flagPre))

    (sumN, countN) = (0, 0)
    (sumE, countE) = (0, 0)
    (sumOther, countOther) = (0, 0)

    if flagPre:
        misura = misura + '_preProcessed'
    for df in df_list:
        n = 0
        for lista in df[misura]:
            val = float(lista[0][1])
            if df['LABEL'][n] == 'E':
                sumE = sumE + val
                countE = countE + 1
            elif df['LABEL'][n] == 'N':
                sumN = sumN + val
                countN = countN + 1
            else:
                sumOther = sumOther + val
                countOther = countOther + 1
            n = n + 1

    avgE = sumE / countE
    avgN = sumN / countN
    avgOther = sumOther / countOther

    (succ, fail) = (0, 0)
    for df in df_list:
        n = 0
        for lista in df[misura]:
            val = float(lista[0][1])
            diffE = abs(avgE - val)
            diffN = abs(avgN - val)
            diffOther = abs(avgOther - val)
            minimo = min(diffE, diffN, diffOther)
            if df['LABEL'][n] == 'E':
                if minimo == diffE:
                    succ = succ + 1
                else:
                    fail = fail + 1
            elif df['LABEL'][n] == 'N':
                if minimo == diffN:
                    succ = succ + 1
                else:
                    fail = fail + 1
            else:
                if minimo == diffOther:
                    succ = succ + 1
                else:
                    fail = fail + 1
            n = n + 1

    #print(misura)
    print(succ/(fail+succ))
    #print('FAIL: ', fail)
    #print('SUCCESS: ', succ)


def rank_all(misura, flagPre):
    df_list = []
    for file in os.listdir("Data/Archive"):
        if file != 'user-story-original.xlsx':
            df_list.append(excel_confronto(file, misura, flagPre))

    if flagPre:
        misura = misura + '_preProcessed'

    i = 0
    for df in df_list:
        n = 0
        for lista in df[misura]:
            val = float(lista[0][1])
            if df['LABEL'][n] == 'E':
                col = 'coral'
            elif df['LABEL'][n] == 'N':
                col = 'lightblue'
            else:
                col = 'black'
            plt.scatter(i, val, c=col)
            n = n + 1
            i = i + 1

    plt.savefig('out/plots/all_' + misura + '.png')
    plt.show()


def rank_us(fileName, misura, flagPre):
    if not fileName + ".pkl" in os.listdir("out"):
        df = excel_to_dataframe(fileName)
    else:
        with open('out/' + fileName + '.pkl', 'rb') as dfl:
            df = pickle.load(dfl)

    if flagPre:
        if misura + '_preProcessed' in df.columns:
            df = df[["USER STORY", "LABEL", misura + '_preProcessed']]
        else:
            df = excel_confronto(fileName, misura, flagPre)
            df = df[["USER STORY", "LABEL", misura + '_preProcessed']]

    else:
        if misura in df.columns:
            df = df[["USER STORY", "LABEL", misura]]
        else:
            df = excel_confronto(fileName, misura, flagPre)
            df = df[["USER STORY", "LABEL", misura]]

    if flagPre:
        misura = misura + '_preProcessed'
    n = 0
    for lista in df[misura]:
        val = float(lista[0][1])
        print(df['LABEL'][n])
        if df['LABEL'][n] == 'E':
            col = 'coral'
        elif df['LABEL'][n] == 'N':
            col = 'lightblue'
        else:
            col = 'black'
        plt.scatter(n, val, c=col)
        n = n + 1

    plt.savefig('out/plots/' + fileName + misura + '.png')
    plt.show()
    return


def prec_rec_all_files_with_avg(misura, flagPre):
    df_list = []
    for file in os.listdir("Data/Archive"):
        if file != 'user-story-original.xlsx':
            df_list.append(excel_confronto(file, misura, flagPre))

    (sumN, countN) = (0, 0)
    (sumE, countE) = (0, 0)
    (sumOther, countOther) = (0, 0)

    if flagPre:
        misura = misura + '_preProcessed'
    for df in df_list:
        n = 0
        for lista in df[misura]:
            val = float(lista[0][1])
            if df['LABEL'][n] == 'E':
                sumE = sumE + val
                countE = countE + 1
            elif df['LABEL'][n] == 'N':
                sumN = sumN + val
                countN = countN + 1
            else:
                sumOther = sumOther + val
                countOther = countOther + 1
            n = n + 1

    avgE = sumE / countE
    avgN = sumN / countN
    avgOther = sumOther / countOther

    exp_labels_list = []
    calc_labels_list = []

    for df in df_list:
        n = 0
        for lista in df[misura]:
            val = float(lista[0][1])
            diffE = abs(avgE - val)
            diffN = abs(avgN - val)
            diffOther = abs(avgOther - val)
            minimo = min(diffE, diffN, diffOther)
            if df['LABEL'][n] != 'E' and df['LABEL'][n] != 'N':
                exp_labels_list.append('other')
            else:
                exp_labels_list.append('N')
            if minimo == diffE:
                calc_labels_list.append('E')
            if minimo == diffN:
                calc_labels_list.append('N')
            if minimo == diffOther:
                calc_labels_list.append('other')
            n = n + 1

    res = precision_recall_fscore_support(exp_labels_list, calc_labels_list, average='macro')
    print(misura)
    print('precision:', res[0])  # ability of the classifier not to label as positive a sample that is negative. precisio
    print('recall: ', res[1])  # ability of the classifier to find all the positive samples. recall
    print(2*(res[0]*res[1])/(res[0]+res[1]))  # fscore


def prec_rec_onebyone(misura, falgPre):
    for file in os.listdir("Data/Archive"):
        if file != 'user-story-original.xlsx':
            prec_rec_one_file_2_labels(file, misura, falgPre)


def success_fail_one_file(fileName, misura, flagPre):
    if not fileName + ".pkl" in os.listdir("out"):
        df = excel_to_dataframe(fileName)
    else:
        with open('out/' + fileName + '.pkl', 'rb') as dfl:
            df = pickle.load(dfl)

    (sumN, countN) = (0, 0)
    (sumE, countE) = (0, 0)
    (sumOther, countOther) = (0, 0)

    if flagPre:
        new_misura = misura + '_preProcessed'
        if new_misura not in df.columns:
            df = excel_confronto(fileName, misura, True)
        misura = new_misura

    if misura not in df.columns:
        df = excel_confronto(fileName, misura, False)

    n = 0

    for lista in df[misura]:
        val = float(lista[0][1])
        if df['LABEL'][n] == 'E':
            sumE = sumE + val
            countE = countE + 1
            col = 'coral'
        elif df['LABEL'][n] == 'N':
            sumN = sumN + val
            countN = countN + 1
            col = 'lightblue'
        else:
            sumOther = sumOther + val
            countOther = countOther + 1
            col = 'black'
        n = n + 1
        plt.scatter(n, val, c=col)

    if countE != 0:
        avgE = sumE / countE
    else:
        avgE = 100
    if countN != 0:
        avgN = sumN / countN
    else:
        avgN = 100
    if countOther != 0:
        avgOther = sumOther / countOther
    else:
        avgOther = 100

    (succ, fail) = (0, 0)
    n = 0
    for lista in df[misura]:
        val = float(lista[0][1])
        diffE = abs(avgE - val)
        diffN = abs(avgN - val)
        diffOther = abs(avgOther - val)
        minimo = min(diffE, diffN, diffOther)
        if df['LABEL'][n] == 'E':
            if minimo == diffE:
                succ = succ + 1
            else:
                fail = fail + 1
        elif df['LABEL'][n] == 'N':
            if minimo == diffN:
                succ = succ + 1
            else:
                fail = fail + 1
        else:
            if minimo == diffOther:
                succ = succ + 1
            else:
                fail = fail + 1
        n = n + 1

    print(misura)
    print(fileName)
    print('FAIL: ', fail)
    print('SUCCESS: ', succ)

    plt.savefig('out/plots/single_file_plots/' + fileName + misura + '.png')
    plt.clf()


def prec_rec_all_files_2_labels(misura, flagPre):
    df_list = []
    for file in os.listdir("Data/Archive"):
        if file != 'user-story-original.xlsx':
            df_list.append(excel_confronto(file, misura, flagPre))

    if flagPre:
        misura = misura + '_preProcessed'

    (minimo, massimo) = (100, 0)
    for df in df_list:
        for lista in df[misura]:
            val = float(lista[0][1])
            if val > massimo:
                massimo = val
            if val < minimo:
                minimo = val

    k = ((massimo + minimo)/2) - 0.3

    exp_labels_list = []
    calc_labels_list = []

    for df in df_list:
        n = 0
        for lista in df[misura]:
            if df['LABEL'][n] != 'E' and df['LABEL'][n] != 'N':
                exp_labels_list.append('other')
            else:
                exp_labels_list.append('N')
            val = float(lista[0][1])
            if misura == "wordMover_word2vec" or misura == "euclidean" or \
                    misura == "wordMover_word2vec_preProcessed" or misura == "euclidean_preProcessed":
                if val > k:
                    calc_labels_list.append('other')
                else:
                    calc_labels_list.append('N')
            else:
                if val > k:
                    calc_labels_list.append('N')
                else:
                    calc_labels_list.append('other')

            n = n + 1

    res = precision_recall_fscore_support(exp_labels_list, calc_labels_list, average='macro', zero_division=1)
    print('precision:', res[0])  # ability of the classifier not to label as positive a sample that is negative. precision
    print('recall: ', res[1])  # ability of the classifier to find all the positive samples. recall
    print(2*(res[0]*res[1])/(res[0]+res[1]))  # fscore


def prec_rec_one_file(fileName, misura, flagPre):
    if not fileName + ".pkl" in os.listdir("out"):
        df = excel_to_dataframe(fileName)
    else:
        with open('out/' + fileName + '.pkl', 'rb') as dfl:
            df = pickle.load(dfl)

    (sumN, countN) = (0, 0)
    (sumE, countE) = (0, 0)
    (sumOther, countOther) = (0, 0)

    if flagPre:
        misura = misura + '_preProcessed'

    n = 0
    for lista in df[misura]:
        val = float(lista[0][1])
        if df['LABEL'][n] == 'E':
            sumE = sumE + val
            countE = countE + 1
        elif df['LABEL'][n] == 'N':
            sumN = sumN + val
            countN = countN + 1
        else:
            sumOther = sumOther + val
            countOther = countOther + 1
        n = n + 1

    if countE != 0:
        avgE = sumE / countE
    else:
        avgE = 100
    if countN != 0:
        avgN = sumN / countN
    else:
        avgN = 100
    if countOther != 0:
        avgOther = sumOther / countOther
    else:
        avgOther = 100

    exp_labels_list = []
    calc_labels_list = []

    n = 0
    for lista in df[misura]:
        val = float(lista[0][1])
        diffE = abs(avgE - val)
        diffN = abs(avgN - val)
        diffOther = abs(avgOther - val)
        minimo = min(diffE, diffN, diffOther)
        if df['LABEL'][n] != 'E' and df['LABEL'][n] != 'N':
            exp_labels_list.append('other')
        else:
            exp_labels_list.append('N')
        if minimo == diffE:
            calc_labels_list.append('E')
        if minimo == diffN:
            calc_labels_list.append('N')
        if minimo == diffOther:
            calc_labels_list.append('other')
        n = n + 1

    res = precision_recall_fscore_support(exp_labels_list, calc_labels_list, average='macro')
    print('precision:', res[0])  # ability of the classifier not to label as positive a sample that is negative. precisio
    print('recall: ', res[1])  # ability of the classifier to find all the positive samples. recall
    print(2*(res[0]*res[1])/(res[0]+res[1]))  # fscore


def prec_rec_one_file_2_labels(fileName, misura, flagPre):
    if not fileName + ".pkl" in os.listdir("out"):
        df = excel_to_dataframe(fileName)
    else:
        with open('out/' + fileName + '.pkl', 'rb') as dfl:
            df = pickle.load(dfl)

    if flagPre:
        misura = misura + '_preProcessed'

    (minimo, massimo) = (100, 0)
    for lista in df[misura]:
        val = float(lista[0][1])
        if val > massimo:
            massimo = val
        if val < minimo:
            minimo = val

    k = ((massimo + minimo)/2) - 0.3

    exp_labels_list = []
    calc_labels_list = []

    n = 0
    for lista in df[misura]:
        if df['LABEL'][n] != 'E' and df['LABEL'][n] != 'N':
            exp_labels_list.append('other')
        else:
            exp_labels_list.append('N')
        val = float(lista[0][1])
        if misura == "wordMover_word2vec" or misura == "euclidean" or \
                misura == "wordMover_word2vec_preProcessed" or misura == "euclidean_preProcessed":
            if val > k:
                calc_labels_list.append('other')
            else:
                calc_labels_list.append('N')
        else:
            if val > k:
                calc_labels_list.append('N')
            else:
                calc_labels_list.append('other')

        n = n + 1

    res = precision_recall_fscore_support(exp_labels_list, calc_labels_list, average='macro', zero_division=1)
    print('precision:', res[0])  # ability of the classifier not to label as positive a sample that is negative. precision
    print('recall: ', res[1])  # ability of the classifier to find all the positive samples. recall
    print(2*(res[0]*res[1])/(res[0]+res[1]))  # fscore