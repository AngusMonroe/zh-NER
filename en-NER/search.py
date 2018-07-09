from datetime import datetime
import elasticsearch
from elasticsearch import Elasticsearch
import sys
es_server = "166.111.7.173"
if len(sys.argv) > 1:
    es_server = sys.argv[1]	
es = Elasticsearch(es_server)

# res = es.get(index="author_names", doc_type='line', id=1)
# print(res['_source'])
# why search for Jie

dbs = ['lab_tagging_authors', 'lab_tagging_insts', 'lab_tagging_keys', 'lab_tagging_venues', 'lab_tagging_locs']
dic = {'lab_tagging_authors': 'PER', 'lab_tagging_insts': 'ORG', 'lab_tagging_keys': 'KEY', 'lab_tagging_venues': 'CON', 'lab_tagging_locs': 'LOC', 'other': 'O'}

def hit(index, string):
    body = {
        "query": {
            "query_string": {
                "query": string 
            }
        }
    }
    res = es.search(index=index, body=body)
    return(res['hits']['total'], res['hits']['hits'])

# 35 Hits:
# {u'_score': 11.1586895, u'_type': u'inst', u'_id': u'1347729', u'_source': {u'name': u'University of Unity between the Nations', u'weight': 1}, u'_index': u'insts'}
def score(res):
    hitnum, res = res[0], res[1]
    if hitnum > 0:
        return hitnum * (res[0]['_source']['weight'] + 1)
    return 0
    
def isOther(results, query, scores):
    def norm(string):
        return string.lower()
    max_guess = zip(*list(sorted(scores.items(), key=lambda x: x[1], reverse = True)))[0][0] 
    query = norm(query)
    # predict = norm(results[max_guess][1][0]['_source']['name'])
    predict = results[max_guess][1][:3]
    print (predict)
    if query != predict:
        for key in scores:
            scores[key] = 0  
        scores['other'] = 1

def norm_and_sort(scores):
    # print (scores)
    if scores['lab_tagging_venues'] > 0 or scores['lab_tagging_locs'] > 0:
        for key in scores:
            if key not in ['lab_tagging_venues', 'lab_tagging_locs']:
                scores[key] = 0
        return scores
    total = sum(list(scores.values()))
    if total == 0:
        scores['other'] = 1
        return
    for key in scores:
        scores[key] = float(scores[key]) / total

    
def clean(query):
    return query.split('\'')[0]

def lookup(query):
    query = clean(query)    
    results = {}

    for db in dbs:
        results[db] = hit(db, query)

    scores = {}
    for key, res in results.items():
        scores[key] = score(res)
    
    norm_and_sort(scores)
    cor_scores = {} 
    for key, value in scores.items():
        cor_scores[dic[key]] = value
    # print (cor_scores)
    return cor_scores 

if __name__ == '__main__':
    while True:
        in_ = raw_input('input >>> ')
        lookup(in_)
    
