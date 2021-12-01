import argparse
from pyproject.utili import confronto, most_similar, heatmap


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    # parser comando confronto
    parser_confronto = subparsers.add_parser('confronto')
    parser_confronto.add_argument('usFile', type=str, help='file di user stories')
    parser_confronto.add_argument('misura', type=str,
                                  help="misure calcolabili: jaccard | cosine_vectorizer | bert_cosine | "
                                       "wordMover_word2vec | euclidean | lsi_cosine | universal_sentence_encoder")
    parser_confronto.set_defaults(func=confronto)

    # parser comando most_similar (us più simili in un file)
    parser_most_similar = subparsers.add_parser('most_similar')
    parser_most_similar.add_argument('usFile', type=str, help='file di user stories')
    parser_most_similar.add_argument('misura', type=str,
                                     help="misure consentite: jaccard | cosine_vectorizer | bert_cosine | "
                                          "wordMover_word2vec | euclidean | lsi_cosine | universal_sentence_encoder")
    parser_most_similar.set_defaults(func=most_similar)

    # parser comando heatmap di un file
    parser_heatmap = subparsers.add_parser('heatmap')
    parser_heatmap.add_argument('usFile', type=str, help='file di user stories')
    parser_heatmap.add_argument('misura', type=str,
                                help="misure consentite: jaccard | cosine_vectorizer | bert_cosine | "
                                     "wordMover_word2vec | euclidean | lsi_cosine | universal_sentence_encoder")
    parser_heatmap.set_defaults(func=heatmap)

    args = parser.parse_args()
    print(args.func(args))


if __name__ == "__main__":
    main()