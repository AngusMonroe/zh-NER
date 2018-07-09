import sys
# sys.path.append('../elastic/')
from search import lookup

def sort(dic):
    for en in dic:
        value = dic[en]
        guess = list(sorted(value.items(), key = lambda x: x[1], reverse=True))
        #special case for date
        if guess[0][0] == 'DATE':
            dic[en] = [(guess[0][0], 1.0)]
        else:
            tonorm = guess[:3]
            names, values = zip(*tonorm)
            names = list(names)
            values = list(values)
            for i in range(len(values)):
                values[i] /= sum(values)
            dic[en] = list(filter(lambda x: x[1] > 0, zip(names, values)))

def merge(dic):
    keys = dic.keys()
# reverse index
    index_words = list(map(lambda x: x.split(), keys))
    index_words = set([i for l in index_words for i in l])
    reverse = dict(zip(index_words, [[] for i in range(len(index_words))]))
    map(lambda x: map(lambda y: reverse[y].append(x), x.split()), keys)
    filtered = list(filter(lambda x: len(x) > 1, reverse.values()))
    dulplications = {}
    all_dups = set()
    for dulps in filtered:
        dulps = sorted(dulps, key=lambda x: len(x))
        end = dulps[-1]
        if end not in dulplications:
            dulplications[end] = set() 
        for i, dulp in enumerate(dulps):
            if end.startswith(dulp) or end.endswith(dulp):
                dulplications[end].add(dulp)
                all_dups.add(dulp)
    if len(dulplications) == 0:
        return dic 
    else:
        merged = {}
        for key, item in dic.items():
            if key not in all_dups:
                merged[key] = item
        """
        for items in dulplications.values():
            guesses = [(k, dic[k]) for k in items] 
            guesses = sorted(guesses, key = lambda x: max(x[1].values()), reverse=True)
            print guesses
            merged[guesses[0][0]] = guesses[0][1]
        """
        for key in dulplications.keys():
            merged[key] = dic[key] 

    return merged
        
def unique(dic):
    name2name = {"LOC":"location", "KEY": "term", "PER": "name", "ORG": "org", "CON": 'con', "DATE": 'date'}
    print ('dic', dic)
    res = {}
    # for item in name2name.values():
    #     res[item] = None
    for key, item in dic.items():
        res[key] = name2name[item[0][0]]
    # print res
    return res

def look_up(results):
    entity_dict = {}
    for res, conf in results:
        for string, label in res:
            if string not in entity_dict:
                entity_dict[string] = {}
            if label not in entity_dict[string]:
                entity_dict[string][label] = 0.
            entity_dict[string][label] += conf
        # only select the first entity
        break
    # entity_dict = merge(entity_dict)

    for en in entity_dict:
        if 'DATE' in entity_dict[en]:
            continue
        lscores = lookup(en)
        # remain Date
         
        # note that the value is normalized (try not to normalize)
        for label, value in lscores.items():
            if label not in entity_dict[en]:
                entity_dict[en][label] = value
            else:
                entity_dict[en][label] *= value
    sort(entity_dict)
    return unique(entity_dict), entity_dict 

def test_dulp():
    a = {'tsinghua university': {'ORG': 0.6}, 'tsinghua': {'ORG': 0.3}, 'university': {'ORG': 0.3}, "data mining": {"KEY": 0.9}, "data": {"KEY": 0.7}}
#    a = {'a': 1, 'b': 2}
    print look_up(a)

if __name__ == '__main__':
    test_dulp()
