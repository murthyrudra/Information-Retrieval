---
layout: default
title: "Neural Retrieval"
---

# Overview

This project demonstrates how to build a Dense Passage Retrieval (DPR) System that allows users to perform queries on a collection of documents.

## Installation
Make sure you have Python installed, then install the required dependencies:
```bash
pip install sentence-transformers faiss
```

Now, we will develop a Neural Retriever on BNS sections. Let's import the NeuralRetriever class


```python
import sys
import os

sys.path.append(os.path.abspath(".."))
from src.NeuralRetriever import NeuralRetriever
```

#### We will load the retriever and the dataset


```python
retriever = NeuralRetriever("intfloat/e5-small-v2")
```

Let us load some documents and them to the retriever class for indexing


```python
# Let us load the BNS sections
def load_md_files(base_folder):
    md_files_dict = []

    # Iterate through all folders in the base directory
    for folder in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder)

        # Check if it's a directory
        if os.path.isdir(folder_path):
            # Iterate through all .md files in the folder
            for filename in os.listdir(folder_path):
                if filename.endswith(".md"):
                    file_path = os.path.join(folder_path, filename)

                    # Open the file and read its contents
                    with open(file_path, "r", encoding="utf-8") as file:
                        file_contents = file.read()

                    # Store the contents in the dictionary with the key being "folder/filename"
                    temp_doc = {}
                    temp_doc["_id"] = f"{folder}/{filename}"
                    temp_doc["text"] = file_contents
                    md_files_dict.append(temp_doc)

    return md_files_dict

bns_data = load_md_files("ilab_sdg/")

for each_section in bns_data:
    retriever.add_document(each_section["_id"], each_section["text"])

retriever.build_index()
```

#### We will search using a simple query


```python
# Search for a query
print("Search for 'robbery':")
results = retriever.search("robbery")
print(results)

# Get the matching documents
print("\nTop matching documents:")
for doc_id, score in results:
    print(f"Document {doc_id} (Score: {score:.4f}): {retriever.documents[doc_id]}")
```
