import os
import pymongo
import nltk
import numpy
from io import BytesIO
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

local_folder_path = 'results_folder'
cosmosdb_connection_string = "mongodb://azureindex:8kSPAOoQoUGksqV08xAf1INaV1arBcWepSm6BMjsJLyYgTWacTwkzlQMjMRT4ZEd6O4thWBWJ3jhACDbkLLcBA==@azureindex.mongo.cosmos.azure.com:10255/?ssl=true&replicaSet=globaldb&retrywrites=false&maxIdleTimeMS=120000&appName=@azureindex@"
client = pymongo.MongoClient(cosmosdb_connection_string)
db = client["Indexer"]
collection = db["Words"]
indexed_pages_collection = db["Indexed pages"]

nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words("english"))
ps = PorterStemmer()
global_doc_number = 0

tf_calculated = False

def extract_text_from_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    text = soup.get_text(separator=" ", strip=True)
    return text.lower()

def preprocess_text(text):
    words = nltk.word_tokenize(text)
    words = [ps.stem(word) for word in words if word.isalnum() and word not in stop_words]
    return words

def calculate_tf(documents):
    if not any(documents):
        return None, None

    vectorizer = TfidfVectorizer()
    tf_matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()
    return tf_matrix, feature_names

for root, dirs, files in os.walk(local_folder_path):
    for file_name in files:
        if file_name.endswith(".html"):
            global_doc_number += 1  
            doc_number = global_doc_number  

            subfolder_name = os.path.relpath(root, local_folder_path)

            text_file_name = f"{subfolder_name}.txt"
            text_file_path = os.path.join(local_folder_path, text_file_name)

            if os.path.exists(text_file_path):
                print(f"Text file {text_file_name} already exists. Doing indexing...")
                file_path = os.path.join(root, file_name)
                with open(file_path, 'r', encoding='utf-8') as file:
                    html_content = file.read()
                text = extract_text_from_html(html_content)
                if text is not None:
                    words = preprocess_text(text)

                    word_freq = Counter(words)

                    tf_matrix, feature_names = calculate_tf([text])

                    if tf_matrix is not None and feature_names is not None:
                        tf_calculated = True

                        if not indexed_pages_collection.find_one({"subfolder_name": subfolder_name}):
                            for word, freq in word_freq.items():
                                word_index = numpy.where(feature_names == word)[0]

                                if word_index.size > 0:  
                                    tf_value = tf_matrix[0, word_index[0]] 

                                    collection.update_one(
                                        {"word": word},
                                        {
                                            "$push": {
                                                "info_list": {
                                                    "doc_no": doc_number,
                                                    "frequency": freq,
                                                    "tf": tf_value,
                                                    "link": subfolder_name
                                                }
                                            },
                                            "$setOnInsert": {"word": word}
                                        },
                                        upsert=True
                                    )
                            indexed_pages_collection.insert_one({"link": subfolder_name, "doc_no": doc_number})
                        else:
                            print(f"TF not calculated for {subfolder_name}: Already indexed.")
            else:
                file_path = os.path.join(root, file_name)
                with open(file_path, 'r', encoding='utf-8') as file:
                    html_content = file.read()

                text = extract_text_from_html(html_content)

                
                text_file_path = os.path.join(local_folder_path, f"{subfolder_name}.txt")
                if os.path.exists(text_file_path):
                    print(f"Text file {text_file_name} already exists. Skipping extraction.")
                else:
                    with open(text_file_path, 'w', encoding='utf-8') as txt_file:
                        txt_file.write(text)

                if text is not None:
                    words = preprocess_text(text)

                    word_freq = Counter(words)

                    tf_matrix, feature_names = calculate_tf([text])

                    if tf_matrix is not None and feature_names is not None:
                        tf_calculated = True
                        if not indexed_pages_collection.find_one({"subfolder_name": subfolder_name}):
                            for word, freq in word_freq.items():
                                word_index = numpy.where(feature_names == word)[0]

                                if word_index.size > 0:  
                                    tf_value = tf_matrix[0, word_index[0]]  
                                    collection.update_one(
                                        {"word": word},
                                        {
                                            "$push": {
                                                "info_list": {
                                                    "doc_no": doc_number,
                                                    "frequency": freq,
                                                    "tf": tf_value,
                                                    "link": subfolder_name
                                                }
                                            },
                                            "$setOnInsert": {"word": word}
                                        },
                                        upsert=True
                                    )

                            indexed_pages_collection.insert_one({"link": subfolder_name, "doc_no": doc_number})
                        else:
                            print(f"TF not calculated for {subfolder_name}: Already indexed.")


if not tf_calculated:
    print("TF not calculated for any website in the local folder.")

print("Indexing complete.")