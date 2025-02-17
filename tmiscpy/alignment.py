def get_alignment_from_editops(target_seq, _editops): 
    '''
    _editops should be the results from Levenshtein.editops(from_seq, target_seq)
    (if needed) pip install Levenshtein # https://pypi.org/project/Levenshtein/

    This should produce a list of 'alignment' and 'labels'that are relative to (same length as) 'from_seq'
    Note that insertions are not in the outputs. You need to obtain these from _editops.
    '''
    to_keep = list(range(len(target_seq)))

    # First delete the ones that are inserted
    to_delete = []
    for act, fr, to in _editops: 
        if act == 'insert': 
            to_delete.append(to)
    to_keep = [x for x in to_keep if x not in to_delete]

    # Then put [None] in the 'delete' positions
    delete_pos = sorted([fr for act, fr, _ in _editops if act == 'delete'])
    res = [None] * (len(to_keep) + len(delete_pos))
    j = 0
    for pos in range(len(res)): 
        if pos not in delete_pos:
            res[pos] = to_keep[j]
            j += 1
    assert len(to_keep) == j

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

