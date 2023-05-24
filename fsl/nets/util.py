#!/usr/bin/env python



def printTitle(title):
    """Convenience function which prints a title. """
    print()
    print('-' * len(title))
    print(title)
    print('-' * len(title))
    print()


def printColumns(titles, columns):
    """Convenience function which pretty-prints a collection of columns in a
    tabular format.

    :arg titles:  A list of titles, one for each column.

    :arg columns: A list of columns, where each column is a list of strings.
    """

    cols  = []

    for t, c in zip(titles, columns):
        cols.append([t] + list(map(str, c)))

    columns = cols
    colLens = []

    for col in columns:
        maxLen = max([len(r) for r in col])
        colLens.append(maxLen)

    fmtStr = ' | '.join(['{{:<{}s}}'.format(l) for l in colLens])

    titles  = [col[0]  for col in columns]
    columns = [col[1:] for col in columns]

    separator = ['-' * l for l in colLens]

    print(fmtStr.format(*titles))
    print(fmtStr.format(*separator))

    nrows = len(columns[0])
    for i in range(nrows):

        row = [col[i] for col in columns]
        print(fmtStr.format(*row))
