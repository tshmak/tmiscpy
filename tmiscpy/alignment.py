def get_alignment_from_editops(target_seq, _editops): 
    # _editops should be the results from Levenshtein.editops(from_seq, target_seq)
    # pip install Levenshtein # https://pypi.org/project/Levenshtein/
    res = list(range(len(target_seq)))

    # First delete the ones that are inserted
    to_delete = []
    for act, fr, to in _editops: 
        if act == 'insert': 
            to_delete.append(to)
    res = [x for x in res if x not in to_delete]

    # Then add [None] to those that are deleted
    delete_pos = sorted([fr for act, fr, _ in _editops if act == 'delete'])
    for pos in reversed(delete_pos): 
        res = res[:pos] + [None] + res[pos:]

    # Label sequence as 'replace', 'equal', or 'delete'
    labels = ['equal' if x is not None else 'delete' for x in res]
    for act, fr, _ in _editops: 
        if act == 'replace': 
            labels[fr] = 'replace'

    return res, labels

if __name__ == '__main__': 
    from Levenshtein import editops
    a = 'abcdefghi'
    b = 'acdEfgGhi'
    moves = editops(a,b)
    alignment, labels = get_alignment_from_editops(b, moves)
    print(a)
    print(b)
    print(alignment)
    print(labels)

