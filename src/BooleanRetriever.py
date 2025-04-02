import re
import os
import datasets
from collections import defaultdict
from nltk.corpus import stopwords
from tqdm import tqdm


class BooleanRetrieval:
    def __init__(self, dataset_name="BNS"):

        if dataset_name == "BNS":
            print("Loading dataset...")
            self.dataset = self.load_md_files("ilab_sdg/")
            print("Dataset loaded.")
        else:
            self.dataset = []

        self.inverted_index = defaultdict(set)
        self.documents = {}
        self.stop_words = set(stopwords.words("english"))

    def load_md_files(self, base_folder):
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

    def preprocess(self, text):
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", "", text)  # Remove special characters
        words = text.split()
        words = [
            word for word in words if word not in self.stop_words
        ]  # Remove stopwords
        return words

    def build_index(self):
        for doc in tqdm(self.dataset, desc="Building Index"):
            doc_id = doc["_id"]
            content = doc["text"]
            self.documents[doc_id] = content
            words = self.preprocess(content)
            for word in set(words):  # Use set to avoid duplicate entries per document
                self.inverted_index[word].add(doc_id)
        print("Index built.")

    def add_document(self, doc_id, content):
        """Add a document to the retrieval system."""
        self.dataset.append({"_id": doc_id, "text": content})

    def _evaluate_query(self, query):
        """
        Recursively evaluate a boolean query to find matching documents.

        The function handles four types of operations:
        1. Parentheses: Evaluates nested expressions first
        2. NOT: Finds documents that don't contain specific terms
        3. AND: Finds the intersection of document sets
        4. OR: Finds the union of document sets

        Args:
            query (str): The boolean query to evaluate

        Returns:
            set: A set of document IDs that match the query
        """
        # STEP 1: Handle parentheses expressions first (highest precedence)
        if "(" in query:
            # Find the matching closing parenthesis by counting nested levels
            start = query.find("(")
            count = 1  # Start with 1 open parenthesis
            for i in range(start + 1, len(query)):
                if query[i] == "(":
                    count += 1  # Increment for nested open parenthesis
                elif query[i] == ")":
                    count -= 1  # Decrement for each closing parenthesis
                if (
                    count == 0
                ):  # When we reach 0, we've found the matching closing parenthesis
                    end = i
                    break

            # Recursively evaluate the expression inside parentheses
            inner_result = self._evaluate_query(query[start + 1 : end])

            # Create a placeholder for the result of the parenthetical expression
            # This allows us to treat the result as a single term in the remaining query
            result_placeholder = f"RESULT_{start}_{end}"

            # Substitute the placeholder into the query, replacing the parenthetical expression
            new_query = query[:start] + result_placeholder + query[end + 1 :]

            # Temporarily store the result in the inverted index under the placeholder name
            # This allows us to process it like any other term in subsequent operations
            self.inverted_index[result_placeholder] = inner_result

            # Recursively evaluate the modified query (with the placeholder)
            final_result = self._evaluate_query(new_query)

            # Clean up by removing the temporary placeholder from the index
            del self.inverted_index[result_placeholder]

            return final_result

        # STEP 2: Handle NOT operator (second highest precedence)
        if " NOT " in query:
            # Split the query at the first NOT operator
            # The left part is what we want to include, the right part is what we want to exclude
            parts = query.split(" NOT ", 1)
            left_part = parts[0].strip()  # Terms to include (may be empty)
            right_part = parts[1].strip()  # Terms to exclude

            # Get the set of all document IDs to use for negation
            all_docs = set(self.documents.keys())

            # Find documents that DON'T contain the term to be excluded
            if right_part in self.inverted_index:
                not_result = (
                    all_docs - self.inverted_index[right_part]
                )  # Set difference
            else:
                # If the term doesn't exist in any document, return all documents
                not_result = all_docs

            # If there's a left part (terms to include), combine it with the NOT result
            if left_part:
                if " AND " in left_part:
                    # If there's an AND in the left part, evaluate it first
                    left_result = self._evaluate_query(left_part)
                    # Then intersect with the NOT result
                    return left_result & not_result
                elif " OR " in left_part:
                    # If there's an OR in the left part, evaluate it first
                    left_result = self._evaluate_query(left_part)
                    # Then union with the NOT result
                    return left_result | not_result
                else:
                    # If it's a single term, get its document set
                    if left_part in self.inverted_index:
                        # Intersect with the NOT result
                        return self.inverted_index[left_part] & not_result
                    # If the term doesn't exist, return empty set
                    return set()
            else:
                # If there's no left part, just return the NOT result
                return not_result

        # STEP 3: Handle AND operator (third highest precedence)
        if " AND " in query:
            # Split the query at AND operators
            parts = query.split(" AND ")
            # Start with all documents and then narrow down
            result = set(self.documents.keys())

            # Process each part of the AND expression
            for part in parts:
                part = part.strip()
                if part.startswith("RESULT_"):
                    # If it's a result placeholder from parentheses processing
                    part_result = self.inverted_index[part]
                elif " OR " in part:
                    # If the part contains OR, evaluate it recursively
                    part_result = self._evaluate_query(part)
                else:
                    # If it's a single term, get its document set
                    part_result = self.inverted_index.get(part.lower(), set())

                # Intersect with the current result (AND operation)
                result &= part_result

            return result

        # STEP 4: Handle OR operator (lowest precedence)
        if " OR " in query:
            # Split the query at OR operators
            parts = query.split(" OR ")
            # Start with no documents and then expand
            result = set()

            # Process each part of the OR expression
            for part in parts:
                part = part.strip()
                if part.startswith("RESULT_"):
                    # If it's a result placeholder from parentheses processing
                    part_result = self.inverted_index[part]
                else:
                    # If it's a single term, get its document set
                    part_result = self.inverted_index.get(part.lower(), set())

                # Union with the current result (OR operation)
                result |= part_result

            return result

        # STEP 5: Handle single term query (base case)
        query = query.lower().strip()
        # Return the set of documents containing the term
        # If the term doesn't exist, return an empty set
        return self.inverted_index.get(query, set())

    def search(self, query):
        """
        Perform boolean search on the indexed documents.

        Query format examples:
        - "term1 AND term2"
        - "term1 OR term2"
        - "term1 AND NOT term2"
        - "term1 OR (term2 AND term3)"

        Args:
            query (_type_): _description_

        Returns:
            _type_: _description_
        """
        doc_ids = self._evaluate_query(query)
        return {doc_id: self.documents[doc_id] for doc_id in doc_ids}
