import sqlite3
import numpy as np
import pandas as pd
import faiss
import os
import json
import pickle
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, SystemMessage
from mistralai import Mistral
from langchain.embeddings import OpenAIEmbeddings  # Replace with your model
from langchain.schema import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import Levenshtein


api_key = os.environ["MISTRAL_API_KEY"]
llm = Mistral(api_key=api_key)

FAISS_INDEX_PATH_PREFIX = "./data/database/faiss_db_prefix"
FAISS_INDEX_PATH_USER_PROMPT = "./data/database/faiss_db_userprompts"

weighted_similarity_threshold = 0.5

embedding_model = HuggingFaceEmbeddings(model_name="thenlper/gte-large")

mistral_llm = Mistral(api_key=os.environ["MISTRAL_API_KEY"])


TABLE_CREATION_SQL_QUERY = '''
    CREATE TABLE IF NOT EXISTS PromptTemplates (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        intent TEXT NOT NULL,
        concept TEXT NOT NULL,
        document_type TEXT NOT NULL,
        context TEXT NOT NULL,
        question_type TEXT NOT NULL,
        metadata JSON,
        keywords TEXT NOT NULL,
        prefix TEXT NOT NULL,
        user_prompt TEXT NOT NULL,
        refined_prompt TEXT NOT NULL,
        UNIQUE(user_prompt, refined_prompt)
    );
    '''

INSERT_INTO_DB_SQL_QUERY = """
    INSERT INTO PromptTemplates (intent, concept, document_type, context, question_type, metadata, keywords, prefix, user_prompt, refined_prompt)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

DELETE_ALL_SQL_QUERY = """
    DELETE FROM PromptTemplates;
    """

DELETE_BY_ID_SQL_QUERY = """
    DELETE FROM PromptTemplates WHERE id = ?;
    """

def get_db_connection(db_name="./data/database/prompt_flow.db"):
    """Establishes and returns a database connection and cursor."""
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    return conn, cursor

def print_table_schema(db_name="prompt_flow.db", table_name="PromptTemplates"):
    """Print the schema of a specific table."""
    conn, cursor = get_db_connection(db_name)

    cursor.execute(f"PRAGMA table_info({table_name});")

    columns = cursor.fetchall()

    print(f"Schema of table '{table_name}':")
    for column in columns:
        print(f"Column Name: {column[1]}, Type: {column[2]}, Not Null: {column[3]}, Default Value: {column[4]}, Primary Key: {column[5]}")

    conn.close()

def checkdb(k=15, db_name = "prompt_flow.db"):
    conn, cursor = get_db_connection(db_name)
    cursor.execute("SELECT * FROM PromptTemplates")
    rows = cursor.fetchall()
    i = 0
    for row in rows:
        if i >= k:
            break
        print(row)
        i += 1

def delete_all_entries():
    """Deletes all records from the PromptTemplates table."""
    conn, cursor = get_db_connection()

    try:
        cursor.execute(DELETE_ALL_SQL_QUERY)
        conn.commit()
        print("All entries deleted successfully.")
    except sqlite3.Error as e:
        print(f"Error deleting entries: {e}")
    finally:
        conn.close()


def delete_entry_by_id(entry_id, db_name="prompt_flow.db"):
    """Deletes an entry from the PromptTemplates table by ID."""
    conn, cursor = get_db_connection()

    try:
        cursor.execute(DELETE_BY_ID_SQL_QUERY, (entry_id,))
        conn.commit()

        if cursor.rowcount > 0:
            print(f"Entry with ID {entry_id} has been deleted.")
        else:
            print(f"No entry found with ID {entry_id}.")

    except sqlite3.Error as e:
        print(f"Database error: {e}")

    finally:
        conn.close()


def delete_table(table_name):
    """Deletes a table from the SQLite database."""
    conn, cursor = get_db_connection()

    try:
        cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
        conn.commit()
        print(f"Table '{table_name}' deleted successfully.")
    except sqlite3.Error as e:
        print(f"Error deleting table: {e}")
    finally:
        conn.close()

def create_prompt_templates_table():
    """Creates the PromptTemplates table if it doesn't exist."""
    conn, cursor = get_db_connection()
    cursor.execute(TABLE_CREATION_SQL_QUERY)
    print("\n ----- SQLite Table PromptTemplates CREATED -------")
    conn.commit()
    conn.close()


def generate_json_response(query, context='', system=system, human=human):
    if context != '':
        context = 'Given the context:\n' + context

    chat_response = mistral_llm.chat.complete(
        model= "mistral-large-latest",
        messages = [
                {
                    "role": "system",
                    "content": system,
                    "role": "user",
                    "content": context + '\n' + human + '\n' + query,
                },
            ]
        )
    response = chat_response.choices[0].message.content
    return response


def extract_json(response_str):
    """Extracts and parses the first valid JSON object from a string response."""
    try:
        start = response_str.find("{")
        end = response_str.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("No valid JSON found in the response string.")

        json_str = response_str[start:end+1]
        return json.loads(json_str)

    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        return None
    except ValueError as e:
        print(f"Error: {e}")
        return None


def json_to_sentence(json_obj):
    sentences = [f"The {key.replace('_', ' ')} of this prompt is '{value}'." for key, value in json_obj.items()]
    return " ".join(sentences)

def process_and_store_queries(queries, refined_prompts, embedding_model, db_name="prompt_flow.db"):
    """Processes multiple user queries, generates responses, and stores them in the database."""

    conn, cursor = get_db_connection()

    for i, user_query in enumerate(queries):
        output_json_str = generate_json_response(user_query)
        output_json = extract_json(output_json_str)
        prefix = json_to_sentence(output_json)

        keywords_str = json.dumps(output_json['keywords'])

        # user_prompt_embedding = embedding_model.embed_query(user_query)  # List
        # prefix_embedding = embedding_model.embed_query(prefix)  # List
        # user_prompt_embedding_blob = sqlite3.Binary(pickle.dumps(user_prompt_embedding))
        # prefix_embedding_blob = sqlite3.Binary(pickle.dumps(prefix_embedding))


        data = (
            output_json['intent'],
            output_json['concept'],
            output_json['document_type'],
            output_json['context'],
            output_json['question_type'],
            None,  # Metadata (if needed, modify accordingly)
            keywords_str,
            prefix,
            user_query,
            refined_prompts[i]  # Using the corresponding refined prompt
        )

        cursor.execute(INSERT_INTO_DB_SQL_QUERY, data)
        print(f"Inserted record {i+1}/{len(queries)} successfully.")

    conn.commit()
    conn.close()
    print("All queries processed and stored.")

def fetch_embeddings_from_db(db_name = "prompt_flow.db"):
    """Fetch IDs and prefix embeddings from SQLite."""
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    cursor.execute("SELECT id,user_prompt, prefix FROM PromptTemplates")
    data = cursor.fetchall()
    conn.close()

    ids_list = []
    user_prompt_list = []
    prefix_list = []
    for row in data:
        if len(row) < 3:  # If prefix_embedding is missing, handle it properly
          print(f"Skipping row due to missing data: {row}")
          continue

        ids_list.append(row[0])
        user_prompt_list.append(row[1])
        prefix_list.append(row[2])

    return ids_list, user_prompt_list, prefix_list


def store_embeddings_in_faiss(faiss_index_path, ids, texts):
    """Store embeddings in FAISS with metadata for retrieval."""

    if ids is None or texts is None:
        print("Skipping FAISS storage due to missing embeddings.")
        return

    # can add more metadata as per need
    documents = [Document(page_content=text, metadata={"id": id_}) for id_, text in zip(ids, texts)]

    faiss_db = FAISS.from_documents(documents, embedding_model)
    faiss_db.save_local(faiss_index_path)

    print(f"Stored {len(ids)} embeddings in FAISS with metadata.")

# Function to calculate lexical similarity using TF-IDF and Cosine Similarity
def lexical_similarity(query, document):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([query, document])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return similarity[0][0]  # Return the similarity score (0 to 1)

# Function to calculate Levenshtein Distance similarity
def levenshtein_similarity(query, document):
    levenshtein_distance = Levenshtein.distance(query, document)
    similarity = 1 - (levenshtein_distance / max(len(query), len(document)))  # Convert distance to similarity
    return similarity  # Higher means more similar

# Function to calculate the weighted average of the three similarity scores
def weighted_average(semantic_score, lexical_score, levenshtein_score):
    average_score = (semantic_score + lexical_score + levenshtein_score) / 3
    return average_score


# Function to find the best matching prompt pair
def get_best_prompt_pair(retrieve_result, new_query, weighted_similarity_threshold):
    best_prompt_id = []
    best_user_prompts = []
    best_refined_prompts = []
    best_faiss_similarities = []
    best_lexical_scores = []
    best_levenshtein_scores = []
    best_weighted_scores = []

    conn, cursor = get_db_connection()

    # for i, (doc, score) in enumerate(retrieve_result, 1):
    #     prompt_id = doc.page_content

    #     if prompt_id is None:
    #         print(f"{i}. No valid ID found (Semantic Score: {score:.4f})")
    #         continue
    for prompt_id, score in zip(retrieved_prompt_ids, retrieved_scores):
        # print(id_)
        # print(score)
        # print(1- score)

        cursor.execute("SELECT user_prompt, refined_prompt FROM PromptTemplates WHERE id = ?", (prompt_id,))
        prompt_text = cursor.fetchone()

        if prompt_text:
            # Invert FAISS score for consistency (1 / (1 + FAISS distance))
            faiss_similarity = 1 - score

            # Calculate lexical similarity score between the query and the retrieved document
            lexical_score = lexical_similarity(new_query, prompt_text[0])

            # Calculate Levenshtein similarity score
            levenshtein_score = levenshtein_similarity(new_query, prompt_text[0])

            # Calculate the weighted average of the three scores
            weighted_score = weighted_average(faiss_similarity, lexical_score, levenshtein_score)

            # Print individual similarity scores and the weighted average score
            print(f"{prompt_id}")
            print(f"   User Prompt: {prompt_text[0]}")
            print(f"   Refined Prompt: {prompt_text[1]}")
            print(f"   FAISS Similarity: {faiss_similarity:.4f}")
            print(f"   Lexical Score: {lexical_score:.4f}")
            print(f"   Levenshtein Score: {levenshtein_score:.4f}")
            print(f"   Weighted Average Score: {weighted_score:.4f}")

            # Apply the thresholding on the weighted average score
            if weighted_score > weighted_similarity_threshold:
                print(f"   This prompt passes the threshold (Weighted Score: {weighted_score:.4f})\n")

                # Store the best prompt details
                best_prompt_id.append(prompt_id)
                best_user_prompts.append(prompt_text[0])
                best_refined_prompts.append(prompt_text[1])
                best_faiss_similarities.append(faiss_similarity)
                best_lexical_scores.append(lexical_score)
                best_levenshtein_scores.append(levenshtein_score)
                best_weighted_scores.append(weighted_score)
            else:
                print(f"   This prompt does not pass the threshold (Weighted Score: {weighted_score:.4f})\n")
        else:
            print(f"{prompt_id}. Prompt not found (Semantic Score: {score:.4f})")

    # Close the database connection
    conn.close()

    # Return the best prompt details
    return best_prompt_id, best_user_prompts, best_refined_prompts, best_faiss_similarities, best_lexical_scores, best_levenshtein_scores, best_weighted_scores


# Function to find the best matching prompt pair
def get_best_prompt_pair1(retrieve_result, new_query, weighted_similarity_threshold):
    best_prompts = {}  # Dictionary to store results

    conn, cursor = get_db_connection()

    for prompt_id, score in zip(retrieved_prompt_ids, retrieved_scores):
        cursor.execute("SELECT user_prompt, refined_prompt FROM PromptTemplates WHERE id = ?", (prompt_id,))
        prompt_text = cursor.fetchone()

        if prompt_text:
            # Invert FAISS score for consistency (1 / (1 + FAISS distance))
            faiss_similarity = 1 - score

            # Calculate lexical similarity score between the query and the retrieved document
            lexical_score = lexical_similarity(new_query, prompt_text[0])

            # Calculate Levenshtein similarity score
            levenshtein_score = levenshtein_similarity(new_query, prompt_text[0])

            # Calculate the weighted average of the three scores
            weighted_score = weighted_average(faiss_similarity, lexical_score, levenshtein_score)

            # Print individual similarity scores and the weighted average score
            print(f"{prompt_id}")
            print(f"   User Prompt: {prompt_text[0]}")
            print(f"   Refined Prompt: {prompt_text[1]}")
            print(f"   FAISS Similarity: {faiss_similarity:.4f}")
            print(f"   Lexical Score: {lexical_score:.4f}")
            print(f"   Levenshtein Score: {levenshtein_score:.4f}")
            print(f"   Weighted Average Score: {weighted_score:.4f}")

            # Apply the thresholding on the weighted average score
            if weighted_score > weighted_similarity_threshold:
                print(f"   This prompt passes the threshold (Weighted Score: {weighted_score:.4f})\n")

                # Store the best prompt details in a dictionary
                best_prompts[prompt_id] = {
                    "user_prompt": prompt_text[0],
                    "refined_prompt": prompt_text[1],
                    "faiss_similarity": float(faiss_similarity),
                    "lexical_score": float(lexical_score),
                    "levenshtein_score": float(levenshtein_score),
                    "weighted_score": float(weighted_score),
                }
            else:
                print(f"   This prompt does not pass the threshold (Weighted Score: {weighted_score:.4f})\n")
        else:
            print(f"{prompt_id}. Prompt not found (Semantic Score: {score:.4f})")

    # Close the database connection
    conn.close()

    # Return the dictionary with all passing prompts
    return best_prompts


def generate_refined_prompt(new_query, best_prompts_dict):
    """
    Generate a refined prompt based on the best-matching prompt from the dictionary.

    :param new_query: The new user query.
    :param best_prompts_dict: A dictionary containing the best-matching prompts.
    :return: A refined prompt generated by the model.
    """

    if not best_prompts_dict:
        print("No prompts passed the threshold.")
        return None

    # Select the best prompt (e.g., highest weighted score)
    best_prompt_id = max(best_prompts_dict, key=lambda k: best_prompts_dict[k]["weighted_score"])
    best_user_prompt = best_prompts_dict[best_prompt_id]["user_prompt"]
    best_refined_prompt = best_prompts_dict[best_prompt_id]["refined_prompt"]

    system_message = """
    You are an expert prompt engineer. Your task is to modify an existing refined prompt template
    to better match a new user query. The refined prompt should retain the original structure
    but be customized for the new query while ensuring that it asks for the relevant details clearly and precisely.

    IMPORTANT: Only return the modified refined prompt text. Do not add any explanations, introductions, or extra text.
    """

    human_message = f"""
    Given the following:
    - Sample User Prompt: {best_user_prompt}
    - Sample Refined Prompt: {best_refined_prompt}
    - New User Prompt: {new_query}

    Please modify the refined prompt template so that it accurately reflects the details needed for the new user prompt.
    Only return the modified refined prompt, with no extra text or explanations.
    """

    chat_response = llm.chat.complete(
        model="mistral-large-latest",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": human_message}
        ]
    )

    # Extract the refined prompt suggestion from the model's response
    response = chat_response.choices[0].message.content.strip()
    return response
