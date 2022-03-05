import os
import pickle

from sklearn.metrics._classification import precision_recall_fscore_support
from sklearn.metrics.pairwise import cosine_similarity

from pyproject.misure import cosine_distance_countvectorizer_method, jaccard_similarity, wordMover_word2vec, euclidean, \
    universal_sentence_encoder_2param, bert
from pyproject.utili import sort_list, loadModelUSE, preprocessing
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from gensim import models, corpora, similarities

if not os.path.exists('fileUtili/GoogleNews-vectors-negative300.bin.gz'):
    raise ValueError("SKIP: You need to download the google news model")

from pyproject.utili import preprocessing, transform

modelUSE = loadModelUSE()


def score(first, second, misura, flag_pre):
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


def success_fail_onebyone(misura, falgPre):
    for file in os.listdir("Data/Archive"):
        if file != 'user-story-original.xlsx':
            success_fail_one_file(file, misura, falgPre)


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


def prec_rec_onebyone(misura, falgPre):
    for file in os.listdir("Data/Archive"):
        if file != 'user-story-original.xlsx':
            prec_rec_one_file_2_labels(file, misura, falgPre)
