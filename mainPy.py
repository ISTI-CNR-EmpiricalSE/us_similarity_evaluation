import argparse

from anaconda_project.project_ops import download

from pyproject.labeled import excel_to_dataframe, original_us_dataframe
from pyproject.parserFunctions import confronto, most_similar, \
    heatmap, confronta_tutti, get_line_byText, concat_all_dataframes, \
    find_file, find_file_test, find_most_similar, excel_confronto, success_fail_onebyone, success_fail_all_files, \
    rank_all, rank_us, prec_rec_all_files_with_avg, prec_rec_onebyone, prec_rec_all_files_2_labels
from pyproject.utili import to_csv


def main():
    # download stopwords from NLTK
    download('stopwords')  # Download stopwords list.

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='parser')

    # parser comando confronto
    parser_confronto = subparsers.add_parser('confronto', description='funzione che confronta le user story '
                                                                      'presenti nel file')
    parser_confronto.add_argument('usFile', type=str, help='file di user stories')
    parser_confronto.add_argument('misura', type=str,
                                  help="misure calcolabili: jaccard | cosine_vectorizer | bert_cosine | "
                                       "wordMover_word2vec | euclidean | lsi_cosine | universal_sentence_encoder")
    parser_confronto.add_argument('-p', action='store_true', help='flag per usare il preprocessing')
    parser_confronto.set_defaults(func=confronto)

    # parser comando most_similar (us più simili in un file)
    parser_most_similar = subparsers.add_parser('most_similar', description='retsituisce le coppie di user story '
                                                                            'associate ai 5 valori di similarità '
                                                                            'migliori per la misura')
    parser_most_similar.add_argument('usFile', type=str, help='file di user stories')
    parser_most_similar.add_argument('misura', type=str,
                                     help="misure consentite: jaccard | cosine_vectorizer | bert_cosine | "
                                          "wordMover_word2vec | euclidean | lsi_cosine | universal_sentence_encoder")
    parser_most_similar.add_argument('-p', action='store_true', help='flag per usare il preprocessing')
    parser_most_similar.set_defaults(func=most_similar)

    # parser comando most_similar (us più simili in un file)
    parser_find_most_similar = subparsers.add_parser('find_most_similar', descriprion='dato un file scelgle 2 '
                                                                                      'requisiti random e per ognuno '
                                                                                      'restituisce la user story più '
                                                                                      'simile')
    parser_find_most_similar.add_argument('usFile', type=str, help='file di user stories')
    parser_find_most_similar.set_defaults(func=find_most_similar)

    # parser comando heatmap di un file
    parser_heatmap = subparsers.add_parser('heatmap', desctiprion='restituisce la heatmap del file e della misura')
    parser_heatmap.add_argument('usFile', type=str, help='file di user stories')
    parser_heatmap.add_argument('misura', type=str,
                                help="misure consentite: jaccard | cosine_vectorizer | bert_cosine | "
                                     "wordMover_word2vec | euclidean | lsi_cosine | universal_sentence_encoder ")
    parser_heatmap.add_argument('-p', action='store_true', help='flag per usare il preprocessing')
    parser_heatmap.set_defaults(func=heatmap)

    # parser comando calcola tutte le misure di un file
    parser_confronta_tutti = subparsers.add_parser('confronta_tutti', description='applica tutte li misure alle user '
                                                                                  'story del file e completa il '
                                                                                  'dataframe associato al file')
    parser_confronta_tutti.add_argument('usFile', type=str, help='file di user stories')
    parser_confronta_tutti.set_defaults(func=confronta_tutti)

    # parser comando restituisce le ranked list della user story
    parser_get_line = subparsers.add_parser('get_line', description='restituisce la riga del dataframe associata alla '
                                                                    'user story us')
    parser_get_line.add_argument('usFile', type=str, help='file di user stories')
    parser_get_line.add_argument('us', type=str, help=' testo user story')
    parser_get_line.set_defaults(func=get_line_byText)

    # parser comando concatena dataframe dei file in Data
    parser_concat = subparsers.add_parser('concat', description='concatena tutti i dataframe creati in uno unico')
    parser_concat.set_defaults(func=concat_all_dataframes)

    # parser comando test di trova file di appartenenza
    parser_find_file = subparsers.add_parser('find_file', description='funzione che trova il file di appartenenza '
                                                                      'delle user story di test estratte da usFile')
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
    parser_find_file_test = subparsers.add_parser('find_file_test', description='funzione che effettua il test fi '
                                                                                'find_file')
    parser_find_file_test.add_argument('usFile', type=str, help='file di user stories')
    parser_find_file_test.add_argument('group_fun', type=str, help='misure tra gruppi: avg, max, aggr')
    parser_find_file_test.set_defaults(func=find_file_test)

    # parser crea dataframe del file
    parser_excel = subparsers.add_parser('excel', description='trasforma i file excel in data/archive in dataframe '
                                                              'con le colonne [user story][label]')
    parser_excel.add_argument('fileName', type=str)
    parser_excel.set_defaults(func=excel_to_dataframe)

    # parser crea dataframe per file con user stories originali
    parser_excel_original = subparsers.add_parser('excel_original', description='trasforma il file di user story '
                                                                                'originali in un dataframe')
    parser_excel_original.set_defaults(func=original_us_dataframe)

    # parser confronto tra us di un file e quelle originali
    parser_excel_confronto = subparsers.add_parser('excel_confronto', description='confronta le user story nel file '
                                                                                  'fileName con quelle originali, '
                                                                                  'salva i riusltati nel dataframe di '
                                                                                  'fileName')
    parser_excel_confronto.add_argument('fileName', type=str, help='file di user stories')
    parser_excel_confronto.add_argument('misura', type=str,
                                        help="misure consentite: jaccard | cosine_vectorizer | bert_cosine | "
                                             "wordMover_word2vec | euclidean | universal_sentence_encoder ")
    parser_excel_confronto.add_argument('-p', action='store_true', help='flag per usare il preprocessing')
    parser_excel_confronto.set_defaults(func=excel_confronto)

    # parser plot di similarità con us originali
    parser_excel_rank = subparsers.add_parser('excel_rank', description='crea il grafico con i valori di similarità '
                                                                        'delle user story di fileName')
    parser_excel_rank.add_argument('fileName', type=str, help='file di user stories')
    parser_excel_rank.add_argument('misura', type=str,
                                   help="misure consentite: jaccard | cosine_vectorizer | bert_cosine | "
                                        "wordMover_word2vec | euclidean | universal_sentence_encoder ")
    parser_excel_rank.add_argument('-p', action='store_true', help='flag per usare il preprocessing')
    parser_excel_rank.set_defaults(func=rank_us)

    # parser per plot unico di tutti i file
    parser_excel_rank_all = subparsers.add_parser('excel_rank_all', description='crea il grafico con i valori di '
                                                                                'similarità delle user story di tutti'
                                                                                ' i file')
    parser_excel_rank_all.add_argument('misura', type=str,
                                       help="misure consentite: jaccard | cosine_vectorizer | bert_cosine | "
                                            "wordMover_word2vec | euclidean | universal_sentence_encoder ")
    parser_excel_rank_all.add_argument('-p', action='store_true', help='flag per usare il preprocessing')
    parser_excel_rank_all.set_defaults(func=rank_all)

    # parser avg
    parser_excel_avg = subparsers.add_parser('excel_avg', description='confronta le 3 etichette date della misure con '
                                                                      'quelle originali considerando tutti i file di '
                                                                      'test come uno unico')
    parser_excel_avg.add_argument('misura', type=str,
                                  help="misure consentite: jaccard | cosine_vectorizer | bert_cosine | "
                                       "wordMover_word2vec | euclidean | universal_sentence_encoder ")
    parser_excel_avg.add_argument('-p', action='store_true', help='flag per usare il preprocessing')
    parser_excel_avg.set_defaults(func=success_fail_all_files)

    # parser avg single file
    parser_excel_avgs = subparsers.add_parser('excel_avg_s',
                                              description='confronta le 3 etichette date della misure con '
                                                          'quelle originali trattando i file singolarmente')
    parser_excel_avgs.add_argument('misura', type=str,
                                   help="misure consentite: jaccard | cosine_vectorizer | bert_cosine | "
                                        "wordMover_word2vec | euclidean | universal_sentence_encoder ")
    parser_excel_avgs.add_argument('-p', action='store_true', help='flag per usare il preprocessing')
    parser_excel_avgs.set_defaults(func=success_fail_onebyone)

    # parser precision/recall
    parser_excel_pr = subparsers.add_parser('prec_rec', description='confronta le 3 etichette date della misure con '
                                                                    'quelle originali considerando tutti i file di '
                                                                    'test come uno unico, riporta i risultati con '
                                                                    'precision e recall')
    parser_excel_pr.add_argument('misura', type=str,
                                 help="misure consentite: jaccard | cosine_vectorizer | bert_cosine | "
                                      "wordMover_word2vec | euclidean | universal_sentence_encoder ")
    parser_excel_pr.add_argument('-p', action='store_true', help='flag per usare il preprocessing')
    parser_excel_pr.set_defaults(func=prec_rec_all_files_with_avg)

    # parser precision/recall single file
    parser_excel_prs = subparsers.add_parser('prec_rec_s', description='confronta le 3 etichette date della misure con '
                                                                       'quelle originali trattando i file '
                                                                       'signolarmente, report i risultati con '
                                                                       'precision e recall')
    parser_excel_prs.add_argument('misura', type=str,
                                  help="misure consentite: jaccard | cosine_vectorizer | bert_cosine | "
                                       "wordMover_word2vec | euclidean | universal_sentence_encoder ")
    parser_excel_prs.add_argument('-p', action='store_true', help='flag per usare il preprocessing')
    parser_excel_prs.set_defaults(func=prec_rec_onebyone)

    # parser trasformare in csv
    parser_to_csv = subparsers.add_parser('to_csv', description='trasforma il datafram di fileName in un file csv')
    parser_to_csv.add_argument('fileName', type=str)
    parser_to_csv.set_defaults(func=to_csv)

    args = parser.parse_args()
    if args.parser == 'confronto':
        print(args.func(args.usFile, args.misura, args.p))
    if args.parser == 'most_similar':
        print(args.func(args.usFile, args.misura, args.p))
    if args.parser == 'find_most_similar':
        print(args.func(args.usFile))
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
        print(args.func(args.usFile, args.group_fun))
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
    if args.parser == 'excel_avg_s':
        args.func(args.misura, args.p)
    if args.parser == 'prec_rec_s':
        args.func(args.misura, args.p)
    if args.parser == 'to_csv':
        args.func(args.fileName)


if __name__ == "__main__":
    main()
