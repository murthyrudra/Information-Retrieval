import re
import math
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class NeuralRetriever:
    """
    An incremental Neural-IDF retrieval system.

    """

    def __init__(self, model_name):
        # Store documents
        self.documents = {}
        self.doc_count = 0
        self.index = None
        self.model = SentenceTransformer(model_name)

        # Flag to indicate if the index needs updating
        self.index_dirty = False

    def add_document(self, doc_id, content):
        """
        Add a document to the retrieval system incrementally.
        Updates relevant data structures but defers full index update.
        """
        # Store the document
        self.documents[doc_id] = content
        self.doc_count += 1

        self.index_dirty = True

    def build_index(self):

        dimension = self.model.get_sentence_embedding_dimension()

        index = faiss.IndexFlatL2(dimension)  # L2 distance index
        self.doc_ids = list(self.documents.keys())
        doc_embeddings = [self.model.encode(text) for text in self.documents.values()]
        index.add(np.array(doc_embeddings))

        self.index = index

        self.index_dirty = False

    def search(self, query, top_k=10):
        """
        Search for documents matching the query.
        Returns a list of (doc_id, score) tuples sorted by score in descending order.
        """
        # Make sure the index is up to date
        if self.index_dirty:
            self.build_index()

        # Calculate query vector
        query_embedding = np.array([self.model.encode(query)])
        distances, indices = self.index.search(query_embedding, top_k)

        retrieved_docs = [
            self.documents[self.doc_ids[idx]]
            for idx in indices[0]
            if idx < len(self.doc_ids)
        ]

        # Return top-k results
        return retrieved_docs
