import sqlite3


def insert_into_db(filename, experiments, context):

    table_name = 'performance'

    conn = sqlite3.connect(filename)

    with conn:

        cur =  conn.cursor()


        for entry in experiments:

            cur.execute(\
                '''insert or ignore into ''' + table_name + ''' (
                context, num_errors, nbest, lazy, morphology, lexicon, word,
                composition_time, search_time, total_time, num_states,
                timeout, ram_usage)
                values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                [entry.get('Context'), entry.get('Number of errors'),
                entry.get('n-best'), entry.get('Lazy'),
                entry.get('Morphology'), entry.get('Lexicon'),
                entry.get('Word'), entry.get('Composition'),
                entry.get('Search'), entry.get('Total time'),
                entry.get('States'), entry.get('Timeout'),
                entry.get('RAM usage')])

            conn.commit

    conn.close()

    return


def main():

    #filename = 'performance_großes_lexicon_lazy.txt'
    #filename = 'performance_kleines_lexicon_lazy.txt'
    #filename = 'performance_großes_lexicon_eager.txt'
    #filename = 'performance_kleines_lexicon_eager.txt'
    #filename = 'performance_extended_small_eager.txt'
    #filename = 'performance_big_lexicon_morphology_eager.txt'
    #filename = 'performance_big_morphology_context_23.txt'
    #filename = 'performance_big_morphology_context_2.txt'
    #filename = 'performance_big_morphology_context_3.txt'
    #filename = 'performance_big_morphology_error_num_4_5.txt'
    filename = 'example_input.txt'

    context = True

    with open(filename) as f:

        data = f.read()
        data = data.strip('\n')

        splitted = data.split('\n\n')

        experiment_list = []

        for experiment in splitted:
            experiment_results = {}
            splitted_entry = experiment.split('\n')

            for value in splitted_entry:
                a = value.split(': ')
                experiment_results[a[0]] = a[1]

            experiment_list.append(experiment_results)

    insert_into_db('performance.db', experiment_list, context)

    return


if __name__ == '__main__':
    main()

