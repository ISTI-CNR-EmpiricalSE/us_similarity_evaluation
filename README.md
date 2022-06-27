# us_similarity_evaluation

## Overview
us_similarity_evaluation is a software that compares different similarity measures and text embeddings. The software is meant to be used on user stories using the following format: 
"As a *role* I want *f unction/f eature*, so that *benefit*"

The software allows to analyze data in three different contexts:
1. change impact analysis: given a document containing user stories, the goal is to create a dataframe containing the similarity value of each pair of user story in the file.
2. requirements retrieval for reuse: given different documents containing user stories and a group of requiremnts taken form one of those files, the goal is to find which file they were taken from.
3. bid management for reuse: given a file containing the original user stories for a product and other files of user story for the same products but from a different source, tha goal is to identify which user stories are present in both documents, which are similar between the documents and which are completely unrelated.

The analyses use the following embedding and measures: jaccard, cosine + count vectorizer, cosine + BERT, word mover distance + word2vec, euclidean, cosine + LSI, universal sentence encoder.

