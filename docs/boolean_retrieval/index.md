---
layout: default
title: "Boolean Retrieval"
---

# Overview

This project demonstrates how to build a simple Boolean Retrieval System that allows users to perform AND, OR, and NOT queries on a collection of documents. We use NLTK for stopword removal and regular expressions for query parsing.

## Dataset
We use the BeIR/FiQA dataset, a financial question-answering dataset, to construct the index and perform queries.

## Installation
Make sure you have Python installed, then install the required dependencies:
```bash
pip install datasets nltk
python -c "import nltk; nltk.download('stopwords')"
```

## Usage
The notebook [Boolean Retrieval](../../notebooks/boolean_retriever.ipynb) on demo of a Boolean Retriever.