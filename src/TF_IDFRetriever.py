import re
import math
from collections import defaultdict, Counter
import numpy as np


class TFIDFRetriever:
    """
    An incremental TF-IDF retrieval system.

    This class allows documents to be added incrementally and performs
    TF-IDF based retrieval without requiring all documents to be loaded at once.
    """

    def __init__(self):
        # Store documents
        self.documents = {}
        # Store term frequency (TF) for each term in each document
        self.term_freq = defaultdict(dict)
        # Store document frequency (DF) for each term
        self.doc_freq = defaultdict(int)
        # Store document vectors (pre-calculated for search efficiency)
        self.doc_vectors = {}
        # Total number of documents
        self.doc_count = 0
        # All terms in the collection
        self.vocabulary = set()
        # Flag to indicate if the index needs updating
        self.index_dirty = False

    def tokenize(self, text):
        """Convert text to lowercase and extract words."""
        # Simple tokenization (lowercase and split on non-alphanumeric chars)
        return re.findall(r"\w+", text.lower())

    def add_document(self, doc_id, content):
        """
        Add a document to the retrieval system incrementally.
        Updates relevant data structures but defers full index update.
        """
        # Store the document
        self.documents[doc_id] = content
        self.doc_count += 1

        # Update term frequencies and document frequencies
        tokens = self.tokenize(content)

        # Calculate term frequency (TF) for this document
        token_counts = Counter(tokens)

        # Update the vocabulary
        self.vocabulary.update(token_counts.keys())

        # Record term frequencies for this document
        for term, count in token_counts.items():
            # Store raw term count for this document
            self.term_freq[term][doc_id] = count

            # Update document frequency (number of documents containing this term)
            if count > 0 and doc_id:
                self.doc_freq[term] += 1

        # Mark the index as needing an update
        self.index_dirty = True

    def update_index(self):
        """
        Update the TF-IDF index for all documents.
        This is called automatically during search if the index is dirty.
        """
        # Only update if there's something to update
        if not self.index_dirty:
            return

        # Calculate TF-IDF vectors for all documents
        for doc_id in self.documents:
            vector = {}
            doc_length = 0

            # Calculate TF-IDF for each term
            for term in self.vocabulary:
                # Calculate term frequency component
                tf = self.term_freq.get(term, {}).get(doc_id, 0)
                if tf > 0:
                    # Log normalization of term frequency
                    tf = 1 + math.log10(tf)

                    # Calculate inverse document frequency component
                    df = self.doc_freq.get(term, 0)
                    idf = math.log10(self.doc_count / (df if df > 0 else 1))

                    # Calculate TF-IDF
                    tfidf = tf * idf
                    vector[term] = tfidf
                    doc_length += tfidf**2

            # Normalize the document vector to unit length (for cosine similarity)
            if doc_length > 0:
                doc_length = math.sqrt(doc_length)
                for term in vector:
                    vector[term] /= doc_length

            # Store the normalized document vector
            self.doc_vectors[doc_id] = vector

        # Reset the dirty flag
        self.index_dirty = False

    def search(self, query, top_k=10):
        """
        Search for documents matching the query.
        Returns a list of (doc_id, score) tuples sorted by score in descending order.
        """
        # Make sure the index is up to date
        if self.index_dirty:
            self.update_index()

        # Tokenize the query
        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        # Calculate query vector
        query_vector = {}
        query_length = 0

        # Use TF-IDF weighting for the query terms
        query_term_counts = Counter(query_tokens)
        for term, count in query_term_counts.items():
            # Query term frequency (with log normalization)
            tf = 1 + math.log10(count)

            # Inverse document frequency
            df = self.doc_freq.get(term, 0)
            idf = math.log10(self.doc_count / (df if df > 0 else 1)) if df > 0 else 0

            # TF-IDF for this query term
            tfidf = tf * idf
            query_vector[term] = tfidf
            query_length += tfidf**2

        # Normalize the query vector
        if query_length > 0:
            query_length = math.sqrt(query_length)
            for term in query_vector:
                query_vector[term] /= query_length

        # Calculate cosine similarity between query and each document
        scores = []
        for doc_id, doc_vector in self.doc_vectors.items():
            # Compute dot product (cosine similarity since vectors are normalized)
            similarity = 0
            for term, weight in query_vector.items():
                if term in doc_vector:
                    similarity += weight * doc_vector[term]

            if similarity > 0:
                scores.append((doc_id, similarity))

        # Sort by score in descending order
        scores.sort(key=lambda x: x[1], reverse=True)

        # Return top-k results
        return scores[:top_k]

    def get_documents(self, doc_ids):
        """Get the content of documents by their IDs."""
        return {
            doc_id: self.documents[doc_id]
            for doc_id in doc_ids
            if doc_id in self.documents
        }

    def get_term_stats(self, term):
        """Get statistics for a specific term."""
        term = term.lower()
        return {
            "document_frequency": self.doc_freq.get(term, 0),
            "documents": self.term_freq.get(term, {}),
            "idf": math.log10(
                self.doc_count
                / (self.doc_freq.get(term, 0) if self.doc_freq.get(term, 0) > 0 else 1)
            ),
        }
