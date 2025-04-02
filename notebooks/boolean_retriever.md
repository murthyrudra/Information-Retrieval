## Boolean Retriever

In this notebook, we will develop a Boolean Retriever on a small dataset. Let's import the BooleanRetriever class


```python
import sys
import os

sys.path.append(os.path.abspath(".."))
from src.BooleanRetriever import BooleanRetrieval
```

    /Users/rudramurthy/miniforge3/envs/py310/lib/python3.10/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm


#### We will load the retriever and the dataset


```python
bool_retriever = BooleanRetrieval()
```

We will add some documents


```python
bool_retriever.add_document(1, "Python is a popular programming language.")
bool_retriever.add_document(2, "Java is also a popular programming language.")
bool_retriever.add_document(3, "Python and Java are both object-oriented.")
bool_retriever.add_document(4, "Python is known for its simplicity and readability.")
bool_retriever.add_document(5, "Data science often uses Python for analysis.")
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

We will print the inverted index


```python
# Let's see how Inverted Index looks like
for each_doc in bool_retriever.inverted_index["python"]:
    print(bool_retriever.documents[each_doc])
```

#### We will search using a simple query


```python
query = "python AND java"
results = bool_retriever.search(query)
for doc_id, content in results.items():
    print(f"DocID: {doc_id}\nContent: {content}\n")
```

## Real-World Example


```python
# let us load the Bharatiya Nyay Sanhita dataset
bool_retriever = BooleanRetrieval("BNS")
```

    Loading dataset...
    Dataset loaded.



```python
# Let us print some documents
print(bool_retriever.dataset[0]["_id"])
print(bool_retriever.dataset[0]["text"])
```

    Chapter_I/Section_01.md
    CHAPTER I: PRELIMINARY
    
    Section 1: Short title, commencement and application
    (1) This Act may be called the Bharatiya Nyaya Sanhita, 2023.
    (2) It shall come into force on such date as the Central Government may, by notification in the Official Gazette, appoint, and different dates may be appointed for different provisions of this Sanhita.
    (3) Every person shall be liable to punishment under this Sanhita and not otherwise for every act or omission contrary to the provisions thereof, of which he shall be guilty within India.
    (4) Any person liable, by any law for the time being in force in India, to be tried for an offence committed beyond India shall be dealt with according to the provisions of this Sanhita for any act committed beyond India in the same manner as if such act had been committed within India.
    (5) The provisions of this Sanhita shall also apply to any offence committed by (a) any citizen of India in any place without and beyond India; (b) any person on any ship or aircraft registered in India wherever it may be; (c) any person in any place without and beyond India committing offence targeting a computer resource located in India.
    Explanation: In this section, the word offence includes every act committed outside India which, if committed in India, would be punishable under this Sanhita.
    Illustration.
    A, who is a citizen of India, commits a murder in any place without and beyond India. He can be tried and convicted of murder in any place in India in which he may be found.
    (6) Nothing in this Sanhita shall affect the provisions of any Act for punishing mutiny and desertion of officers, soldiers, sailors or airmen in the service of the Government of India or the provisions of any special or local law.
    



```python
# Time to build index
bool_retriever.build_index()
```

    Building Index: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 358/358 [00:00<00:00, 21782.90it/s]

    Index built.


    



```python
search_results = bool_retriever.search("robbery AND chain-snatching")
```


```python
for doc_id in search_results:
    print(doc_id)
    print(search_results[doc_id])
    print("\n")
```


```python

```
