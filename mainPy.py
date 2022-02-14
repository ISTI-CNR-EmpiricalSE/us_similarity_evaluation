import argparse

from anaconda_project.project_ops import download

from pyproject.labeled import excel_to_dataframe, original_us_dataframe, excel_confronto, rank_us, rank_all, avg_value, \
    precision_recall
from pyproject.parserFunctions import confronto, most_similar, \
    heatmap, confronta_tutti, get_line_byText, concat_all_dataframes, \
    find_file, find_file_test


def main():
    #download stopwords from NLTK
    download('stopwords')  # Download stopwords list.

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
                                     "wordMover_word2vec | euclidean | lsi_cosine | universal_sentence_encoder ")
    parser_heatmap.add_argument('-p', action='store_true', help='flag per usare il preprocessing')
    parser_heatmap.set_defaults(func=heatmap)

    # parser comando calcola tutte le misure di un file
    parser_confronta_tutti = subparsers.add_parser('confronta_tutti')
    parser_confronta_tutti.add_argument('usFile', type=str, help='file di user stories')
    parser_confronta_tutti.set_defaults(func=confronta_tutti)

    # parser comando restituisci le ranked list della user story
    parser_get_line = subparsers.add_parser('get_line')
    parser_get_line.add_argument('usFile', type=str, help='file di user stories')
    parser_get_line.add_argument('us', type=str, help=' testo user story')
    parser_get_line.set_defaults(func=get_line_byText)

    # parser comando concatena dataframe dei file in Data
    parser_concat = subparsers.add_parser('concat')
    parser_concat.set_defaults(func=concat_all_dataframes)

    # parser comando test di trova file di appartenenza
    parser_find_file = subparsers.add_parser('find_file')
    parser_find_file.add_argument('n', type=int, help='numero di test da fare')
    parser_find_file.add_argument('usFile', type=str, help='file di user stories')
    parser_find_file.add_argument('k', type=int, help='numero user Stories da estrarre')
    parser_find_file.add_argument('group_fun', type=str, help='misure tra gruppi: avg, max, aggr')
    parser_find_file.add_argument('misura', type=str,
                                  help="misure consentite: jaccard | cosine_vectorizer | bert_cosine | "
                                       "wordMover_word2vec | euclidean | universal_sentence_encoder ")
    parser_find_file.add_argument('-p', action='store_true', help='flag per usare il preprocessing')
    parser_find_file.set_defaults(func=find_file)

    # parser comando test di trova file di appartenenza
    parser_find_file = subparsers.add_parser('find_file')
    parser_find_file.add_argument('n', type=int, help='numero di test da fare')
    parser_find_file.add_argument('usFile', type=str, help='file di user stories')
    parser_find_file.add_argument('k', type=int, help='numero user Stories da estrarre')
    parser_find_file.add_argument('group_fun', type=str, help='misure tra gruppi: avg, max, aggr')
    parser_find_file.add_argument('misura', type=str,
                                  help="misure consentite: jaccard | cosine_vectorizer | bert_cosine | "
                                       "wordMover_word2vec | euclidean | universal_sentence_encoder ")
    parser_find_file.add_argument('-p', action='store_true', help='flag per usare il preprocessing')
    parser_find_file.set_defaults(func=find_file)

    # parser comando
    parser_find_file_test = subparsers.add_parser('find_file_test')
    parser_find_file_test.add_argument('usFile', type=str, help='file di user stories')
    parser_find_file_test.add_argument('group_fun', type=str, help='misure tra gruppi: avg, max, aggr')
    parser_find_file_test.add_argument('misura', type=str,
                                  help="misure consentite: jaccard | cosine_vectorizer | bert_cosine | "
                                       "wordMover_word2vec | euclidean | universal_sentence_encoder ")
    parser_find_file_test.add_argument('-p', action='store_true', help='flag per usare il preprocessing')
    parser_find_file_test.set_defaults(func=find_file_test)



    # parser crea dataframe del file
    parser_excel = subparsers.add_parser('excel')
    parser_excel.add_argument('fileName', type=str)
    parser_excel.set_defaults(func=excel_to_dataframe)

    # parser crea dataframe per file con user stories originali
    parser_excel_original = subparsers.add_parser('excel_original')
    parser_excel_original.set_defaults(func=original_us_dataframe)

    # parser confronto tra us di un file e quelle originali
    parser_excel_confronto = subparsers.add_parser('excel_confronto')
    parser_excel_confronto.add_argument('fileName', type=str, help='file di user stories')
    parser_excel_confronto.add_argument('misura', type=str,
                                  help="misure consentite: jaccard | cosine_vectorizer | bert_cosine | "
                                       "wordMover_word2vec | euclidean | universal_sentence_encoder ")
    parser_excel_confronto.add_argument('-p', action='store_true', help='flag per usare il preprocessing')
    parser_excel_confronto.set_defaults(func=excel_confronto)

    # parser plot di similarità con us originali
    parser_excel_rank = subparsers.add_parser('excel_rank')
    parser_excel_rank.add_argument('fileName', type=str, help='file di user stories')
    parser_excel_rank.add_argument('misura', type=str,
                                  help="misure consentite: jaccard | cosine_vectorizer | bert_cosine | "
                                       "wordMover_word2vec | euclidean | universal_sentence_encoder ")
    parser_excel_rank.add_argument('-p', action='store_true', help='flag per usare il preprocessing')
    parser_excel_rank.set_defaults(func=rank_us)

    # parser per plot unico di tutti i file
    parser_excel_rank_all = subparsers.add_parser('excel_rank_all')
    parser_excel_rank_all.add_argument('misura', type=str,
                                  help="misure consentite: jaccard | cosine_vectorizer | bert_cosine | "
                                       "wordMover_word2vec | euclidean | universal_sentence_encoder ")
    parser_excel_rank_all.add_argument('-p', action='store_true', help='flag per usare il preprocessing')
    parser_excel_rank_all.set_defaults(func=rank_all)

    # parser avg
    parser_excel_avg = subparsers.add_parser('excel_avg')
    parser_excel_avg.add_argument('misura', type=str,
                                  help="misure consentite: jaccard | cosine_vectorizer | bert_cosine | "
                                       "wordMover_word2vec | euclidean | universal_sentence_encoder ")
    parser_excel_avg.add_argument('-p', action='store_true', help='flag per usare il preprocessing')
    parser_excel_avg.set_defaults(func=avg_value)

    # parser precision/recall
    parser_excel_pr = subparsers.add_parser('prec_rec')
    parser_excel_pr.add_argument('misura', type=str,
                                  help="misure consentite: jaccard | cosine_vectorizer | bert_cosine | "
                                       "wordMover_word2vec | euclidean | universal_sentence_encoder ")
    parser_excel_pr.add_argument('-p', action='store_true', help='flag per usare il preprocessing')
    parser_excel_pr.set_defaults(func=precision_recall)

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
    if args.parser == 'find_file':
        print(args.func(args.n, args.usFile, args.k,
                        args.group_fun, args.misura, args.p))
    if args.parser == 'find_file_test':
        print(args.func(args.usFile,
                        args.group_fun, args.misura, args.p))
    if args.parser == "excel":
        args.func(args.fileName)
    if args.parser == "excel_original":
        args.func()
    if args.parser == 'excel_confronto':
        print(args.func(args.fileName, args.misura, args.p))
    if args.parser == 'excel_rank':
        print(args.func(args.fileName, args.misura, args.p))
    if args.parser == 'excel_rank_all':
        args.func(args.misura, args.p)
    if args.parser == 'excel_avg':
        args.func(args.misura, args.p)
    if args.parser == 'prec_rec':
        args.func(args.misura, args.p)


if __name__ == "__main__":
    main()
