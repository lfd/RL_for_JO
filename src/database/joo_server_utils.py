from typing import List

# query is a list of tables
def generateJoinOrders(query: List):
    '''
    Create a list with all possible join trees

    The list is created using dynamic programming
    Uses bit strings instead of sets to check for overlap between subtrees
    This should be significantly faster
    '''
    # dynamic programming
    n = len(query)
    level = [[] for _ in range(n + 1)]
    # level 1
    level[1].extend([(e, toBitString(query, e)) for e in query])
    # level 2 - n
    for new in range(2, n + 1):
        for i in range(1, (new // 2) + 1):
            for idL in range(len(level[i])):
                for idR in range(len(level[new - i])):
                    if (i == new - i) and (idR <= idL):
                        continue
                    left = level[i][idL]
                    right = level[new - i][idR]
                    if (left[1] & right[1]) == 0:
                        level[new].append(([left[0], right[0]], left[1] | right[1]))

    return [e[0] for e in level[n]]

def _toIntString(order, fs, nextId):
    res = []
    if isinstance(order[0], list):
        lstLeft, idLeft, nextId = _toIntString(order[0], fs, nextId)
        res.extend(lstLeft)
    else:
        idLeft = fs.index(order[0])
    if isinstance(order[1], list):
        lstRight, idRight, nextId = _toIntString(order[1], fs, nextId)
        res.extend(lstRight)
    else:
        idRight = fs.index(order[1])
    res.append(idLeft)
    res.append(idRight)
    print(idLeft, idRight, len(fs) - 1)
    return res, nextId, nextId + 1


def toIntString(order, fs):
    if order == []: return []
    result, _, _ = _toIntString(order, fs, len(fs))
    return result

