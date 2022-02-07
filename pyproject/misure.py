import sklearn.metrics.pairwise
from gensim.models.lsimodel import LsiModel
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial import distance
from scipy.stats import entropy

import gensim.corpora as corpora

import numpy as np
from gensim import corpora, similarities
from gensim.models import LsiModel
from nltk.corpus import stopwords

import nltk
import sklearn

import re


# cosine distance count vectorizer
def cosine_distance_countvectorizer_method(s1, s2):
    # sentences to list
    allsentences = [s1, s2]

    # text to vector
    vectorizer = CountVectorizer()
    all_sentences_to_vector = vectorizer.fit_transform(allsentences)
    text_to_vector_v1 = all_sentences_to_vector.toarray()[0].tolist()
    text_to_vector_v2 = all_sentences_to_vector.toarray()[1].tolist()

    # distance of similarity
    cosine = distance.cosine(text_to_vector_v1, text_to_vector_v2)
    return round((1 - cosine) * 100, 2)


# jaccard similarity
def jaccard_similarity(query, document):
    sentence = query.split()
    other = document.split()
    intersection = set(sentence).intersection(set(other))
    union = set(sentence).union(set(other))
    return round(len(intersection) / len(union), 2)


# BERT
def bert(sentences):
    model_name = 'bert-base-nli-mean-tokens'

    model = SentenceTransformer(model_name)

    # ogni frase rappresentata da un vec di dim 768
    sentence_vecs = model.encode(sentences)
    return sentence_vecs


# word Mover Distance + word2vec
def wordMover_word2vec(s1, s2, model, stop_words):
    s1 = s1.lower().split()
    s2 = s2.lower().split()

    # remove stop words
    s1 = [w for w in s1 if w not in stop_words]
    s2 = [w for w in s2 if w not in stop_words]

    distance = model.wmdistance(s1, s2)  # Compute WMD as normal.

    return distance


# euclidean distance
# ret: lista triple per us "first"
def euclidean(first, sentences):
    score_list = []
    for i, sentence in enumerate(sentences):
        score = sklearn.metrics.pairwise.euclidean_distances([sentences[i]], [sentences[first]])[0][0]
        tripla = (first, i, score)
        score_list.append(tripla)
    return score_list


# function to filter out stopwords and apply word stemming for LSI
def filter_words_and_get_word_stems(document, word_tokenizer, word_stemmer,
                                    stopword_set, pattern_to_match_words=r"[^\w]",
                                    word_length_minimum_n_chars=2):
    """Remove multiple white spaces and all non word content from text and
    extract words. Then filter out stopwords and words with a length smaller
    than word_length_minimum and apply word stemmer to get wordstems. Finally
    return word stems.
    """
    document = re.sub(pattern_to_match_words, r" ", document)
    document = re.sub(r"\s+", r" ", document)
    words = word_tokenizer.tokenize(document)
    words_filtered = [word.lower()
                      for word in words
                      if word.lower() not in stopword_set and len(word) >= word_length_minimum_n_chars]
    word_stems = [word_stemmer.lemmatize(word) for word in words_filtered]
    return word_stems


def lsi(sentences):
    nltk.download("stopwords")
    nltk.download("wordnet")
    # set stopword set, word stemmer and word tokenizer
    stopword_set = set(stopwords.words("english"))
    word_tokenizer = nltk.tokenize.WordPunctTokenizer()
    word_stemmer = nltk.WordNetLemmatizer()

    # apply cleaning, filtering and word stemming to training documents
    word_stem_arrays_train = [
        filter_words_and_get_word_stems(
            str(sentence),
            word_tokenizer,
            word_stemmer,
            stopword_set
        ) for sentence in sentences]

    # PROCESS

    # create dictionary containing unique word stems of training documents
    # TF (term frequencies) or "global weights"
    dictionary = corpora.Dictionary(
        word_stem_array_train
        for word_stem_array_train in word_stem_arrays_train)

    # create corpus containing word stem id from dictionary and word stem count
    # for each word in each document
    # DF (document frequencies, for all terms in each document) or "local weights"
    corpus = [
        dictionary.doc2bow(word_stem_array_train)
        for word_stem_array_train in word_stem_arrays_train]

    # create LSI model (Latent Semantic Indexing) from corpus and dictionary
    # LSI model consists of Singular Value Decomposition (SVD) of
    # Term Document Matrix M: M = T x S x D'
    # and dimensionality reductions of T, S and D ("Derivation")
    lsi_model = LsiModel(
        corpus=corpus,
        id2word=dictionary  # , num_topics = 2 #(opt. setting for explicit dim. change)
    )

    # calculate cosine similarity matrix for all training document LSI vectors
    cosine_similarity_matrix = similarities.MatrixSimilarity(lsi_model[corpus])

    sim_list = [[row for row in cosine_similarity_matrix]]

    return sim_list


def universal_sentence_encoder(userStories, modelUSE):
    # embeddings:
    def embed(sentence):
        return modelUSE(sentence)

    embedded_sentences = embed(userStories)

    total_list = []
    for first in range(0, len(embedded_sentences)):
        sim_list = []
        for other in range(0, len(embedded_sentences)):
            tripla = (first, other, np.inner(embedded_sentences[first],
                                             embedded_sentences[other]))
            sim_list.append(tripla)
        total_list.append(sim_list)
    return total_list


def universal_sentence_encoder_2param(first_set, second_set, modelUSE):
    # embeddings:
    def embed(sentence):
        return modelUSE(sentence)

    embedded_first = embed(first_set)
    embedded_second = embed(second_set)

    sim_list = []
    for first in range(0, len(embedded_first)):
        for other in range(0, len(embedded_second)):
            tripla = (first, other, np.inner(embedded_first[first],
                                             embedded_second[other]))
            sim_list.append(tripla)
    return sim_list
