---
layout: default
title: "Boolean Retrieval"
---

# Overview

This project demonstrates how to build a simple Boolean Retrieval System that allows users to perform AND, OR, and NOT queries on a collection of documents. We use NLTK for stopword removal and regular expressions for query parsing.

## Installation
Make sure you have Python installed, then install the required dependencies:
```bash
pip install datasets nltk
python -c "import nltk; nltk.download('stopwords')"
```

Now, we will develop a Boolean Retriever on BNS sections. Let's import the BooleanRetriever class


```python
import sys
import os

sys.path.append(os.path.abspath(".."))
from src.BooleanRetriever import BooleanRetrieval
```

#### We will load the retriever and the dataset


```python
bool_retriever = BooleanRetrieval("BNS")
```

Let us print one section


```python
print(bool_retriever.dataset[0]["_id"])
print(bool_retriever.dataset[0]["text"])
```

#### We will build the index


```python
bool_retriever.build_index()
```

We will print the documents


```python
first_key = ""
for key in bool_retriever.documents:
    first_key = key
    break

print(f"Key is {first_key}")
print(bool_retriever.documents[first_key])
```

We will build the index


```python
bool_retriever.build_index()

# Let's see how Inverted Index looks like
for each_doc in bool_retriever.inverted_index["robbery"]:
    print(bool_retriever.documents[each_doc])
```

#### We will search using a simple query


```python
search_results = bool_retriever.search("robbery AND chain-snatching")

for doc_id, content in search_results.items():
    print(f"DocID: {doc_id}\nContent: {content}\n")
```
