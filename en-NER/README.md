# [Tool] Entity Recognition in Short Query
This tool helps recognize the entities in short queries posted to Aminer domain.
It could recognize the entity of the following six types,
<pre>
location (country or city, e.g., Beijing)
term (academic term, e.g., data mining)
name (expert name, e.g., Jiawei Han)
org (academic orgnaization e.g., Tsinghua)
con (scientific conference, e.g., kdd)
date (year, e.g., 2012)
</pre>
The query form looks like:
<pre>
Jiawei Han
kdd 2012
papers in tsinghua 2012
</pre>
## Requirements:
1. python 2.7
2. Theano = 1.0.1
3. numpy = 1.14.2
4. elasticsearch
5. flask

## Set up
To set up the service, 
1. run the elasticsearch service

```
./bin/elasticsearch
```
2. run the server

```
python server.py
```

## Query
GET Method:
```
http://166.111.7.173:5010/query/<your query>
```
Return in json, Example:
```
http://166.111.7.173:5010/query/tsinghua 2012
{"tsinghua": "org", "2012": "date"}
```

## File orgnization
<pre>
|- server.py 
|- [dir] dataset (word library)
|- [dir] evaluation (help tools when training)
|- [dir] models (well-trained models)
|- [dir] tagger_top_lookup (main logics of named entity recognition)
</pre>
