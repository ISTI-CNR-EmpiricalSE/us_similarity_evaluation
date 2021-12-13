import argparse
from pyproject.utili import confronto, most_similar, heatmap, confronta_tutti, get_line, concat_all_dataframes


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='parser')

    # parser comando confronto
    parser_confronto = subparsers.add_parser('confronto')
    parser_confronto.add_argument('usFile', type=str, help='file di user stories')
    parser_confronto.add_argument('misura', type=str,
                                  help="misure calcolabili: jaccard | cosine_vectorizer | bert_cosine | "
                                       "wordMover_word2vec | euclidean | lsi_cosine | universal_sentence_encoder | all")
    parser_confronto.add_argument('-p', action='store_true', help='flag per usare il preprocessing')
    parser_confronto.set_defaults(func=confronto)

    # parser comando most_similar (us più simili in un file)
    parser_most_similar = subparsers.add_parser('most_similar')
    parser_most_similar.add_argument('usFile', type=str, help='file di user stories')
    parser_most_similar.add_argument('misura', type=str,
                                     help="misure consentite: jaccard | cosine_vectorizer | bert_cosine | "
                                          "wordMover_word2vec | euclidean | lsi_cosine | universal_sentence_encoder")
    parser_most_similar.add_argument('-p', action='store_true', help='flag per usare il preprocessing')
    parser_most_similar.set_defaults(func=most_similar)

    # parser comando heatmap di un file
    parser_heatmap = subparsers.add_parser('heatmap')
    parser_heatmap.add_argument('usFile', type=str, help='file di user stories')
    parser_heatmap.add_argument('misura', type=str,
                                help="misure consentite: jaccard | cosine_vectorizer | bert_cosine | "
                                     "wordMover_word2vec | euclidean | lsi_cosine | universal_sentence_encoder")
    parser_heatmap.add_argument('-p', action='store_true', help='flag per usare il preprocessing')
    parser_heatmap.set_defaults(func=heatmap)

    # parser comando calcola tutte le misure di un file
    parser_confronta_tutti = subparsers.add_parser('confronta_tutti')
    parser_confronta_tutti.add_argument('usFile', type=str, help='file di user stories')
    parser_confronta_tutti.set_defaults(func=confronta_tutti)

    # parser comando restituisci le ranked list della user story
    parser_get_line = subparsers.add_parser('get_line')
    parser_get_line .add_argument('usFile', type=str, help='file di user stories')
    parser_get_line .add_argument('us', type=str, help=' testo user story')
    parser_get_line .set_defaults(func=get_line)

    # parser comando concatena dataframe dei file in Data
    parser_concat = subparsers.add_parser('concat')
    parser_concat .set_defaults(func=concat_all_dataframes)

    args = parser.parse_args()
    if args.parser == 'confronto':
        print(args.func(args.usFile, args.misura, args.p))
    if args.parser == 'most_similar':
        print(args.func(args.usFile, args.misura, args.p))
    if args.parser == 'heatmap':
        print(args.func(args.usFile, args.misura, args.p))
    if args.parser == 'confronta_tutti':
        print(args.func(args.usFile))
    if args.parser == 'get_line':
        print(args.func(args.usFile, args.us))
    if args.parser == 'concat':
        print(args.func())




if __name__ == "__main__":
    main()
