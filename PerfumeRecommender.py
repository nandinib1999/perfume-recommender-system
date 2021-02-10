import pandas as pd
import numpy as np
import pickle
from textwrap import wrap
import re
import os

import matplotlib.pyplot as plt
from skimage import io

import utils as ut

from gensim.models.doc2vec import Doc2Vec
from gensim.test.utils import get_tmpfile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from scipy import sparse

def load_models(input_data_path='final_perfume_data_temp.csv', model_dir="models"):
    print('Load models..')
    global doc2vec, tf_idf, svd, svd_feature_matrix, doctovec_feature_matrix, df, sentiment
    doc2vec = Doc2Vec.load(os.path.join(model_dir, "doc2vec_model"))
    tf_idf = pickle.load(open(os.path.join(model_dir, "tf-idf_vectorizer.pkl"), "rb"))
    svd = pickle.load(open(os.path.join(model_dir, "svd_model.pkl"), "rb"))
    svd_feature_matrix = pickle.load(open(os.path.join(model_dir, "lsa_embeddings.pkl"), "rb"))
    doctovec_feature_matrix = pickle.load(open(os.path.join(model_dir, "doc2vec_embeddings.pkl"), "rb"))
    df = pd.read_csv(input_data_path, engine='python')
    df['Notes'].fillna(' ', inplace=True)


def tfidf_embedding_vector(text):
    message_array = tf_idf.transform([text]).toarray()
    message_array = svd.transform(message_array)
    message_array = message_array[:,0:2000].reshape(1, -1)
    return message_array


def doctovec_embedding_vector(text):
    message_array = doc2vec.infer_vector(doc_words=text.split(" "), epochs=200)
    message_array = message_array.reshape(1, -1)
    return message_array


def get_similarity_scores(message_array, embeddings):
    cosine_sim_matrix = pd.DataFrame(cosine_similarity(X=embeddings, Y=message_array, dense_output=True))
    cosine_sim_matrix.set_index(embeddings.index, inplace=True)
    cosine_sim_matrix.columns = ["cosine_similarity"]
    return cosine_sim_matrix


def get_ensemble_similarity_scores(text):
    text = ut.preprocess_message(text)
    tf_idf_embedding = tfidf_embedding_vector(text)
    doct2vec_embedding = doctovec_embedding_vector(text)

    tfidf_similarity = get_similarity_scores(tf_idf_embedding, svd_feature_matrix)
    doc2vec_similarity = get_similarity_scores(doct2vec_embedding, doctovec_feature_matrix)

    ensemble_similarity = pd.merge(doc2vec_similarity, tfidf_similarity, left_index=True, right_index=True)
    ensemble_similarity.columns = ["doc2vec_similarity", "tfidf_similarity"]
    ensemble_similarity['ensemble_similarity'] = (ensemble_similarity["doc2vec_similarity"] + ensemble_similarity["tfidf_similarity"])/2
    ensemble_similarity.sort_values(by="ensemble_similarity", ascending=False, inplace=True)
    return ensemble_similarity


def find_similar_perfumes(text, n):
    print(text, n)
    n = int(n)
    similar_perfumes = get_ensemble_similarity_scores(love_message)
    print(similar_perfumes)
    similar_perfumes = similar_perfumes.drop_duplicates(subset='Name', keep='first')

    return similar_perfumes.iloc[:n, :]

def details_of_recommendations(recommended_perfumes):
    details = []
    for indx, row in recommended_perfumes.iterrows():
        name = row['Name']
        notes = df[df['Name'] == name]['Notes'].values[0].encode("ascii", errors="ignore").decode()
        image_url = df[df['Name'] == name]['Image URL'].values[0]
        details.append([name, notes, image_url])
    return details
        

