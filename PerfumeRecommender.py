import pandas as pd
import numpy as np
import pickle
from textwrap import wrap
import re
import os

import matplotlib.pyplot as plt
from skimage import io

import nltk
nltk.download('vader_lexicon')
import utils as ut
from nltk.sentiment.vader import SentimentIntensityAnalyzer as sentiment_analyzer

from gensim.models.doc2vec import Doc2Vec
from gensim.test.utils import get_tmpfile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from scipy import sparse

def load_models(input_data_path='final_perfume_data_temp.csv', model_dir="models"):
    global doc2vec, tf_idf, svd, svd_feature_matrix, doctovec_feature_matrix, df, sentiment
    doc2vec = Doc2Vec.load(os.path.join(model_dir, "doc2vec_model"))
    tf_idf = pickle.load(open(os.path.join(model_dir, "tf-idf_vectorizer.pkl"), "rb"))
    svd = pickle.load(open(os.path.join(model_dir, "svd_model.pkl"), "rb"))
    svd_feature_matrix = pickle.load(open(os.path.join(model_dir, "lsa_embeddings.pkl"), "rb"))
    doctovec_feature_matrix = pickle.load(open(os.path.join(model_dir, "doc2vec_embeddings.pkl"), "rb"))
    df = pd.read_csv(input_data_path, engine='python')
    df['Notes'].fillna(' ', inplace=True)
    sentiment = sentiment_analyzer()


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

def get_sentiment(text):
    sentences = re.split('\.|\but',text)
    sentences = [x for x in sentences if x != ""]
    love_message = ""
    hate_message = ""
    for s in sentences:
        sentiment_scores = sentiment.polarity_scores(s)
        if sentiment_scores['neg'] > 0:
            hate_message = hate_message + s
        else:
            love_message = love_message + s
    return love_message, hate_message


def get_dissimilarity_scores(text):
    text = ut.preprocess_message(text)
    tf_idf_embedding = tfidf_embedding_vector(text)
    doct2vec_embedding = doctovec_embedding_vector(text)

    tfidf_dissimilarity = get_similarity_scores(tf_idf_embedding, svd_feature_matrix)
    doc2vec_dissimilarity = get_similarity_scores(doct2vec_embedding, doctovec_feature_matrix)

    ensemble_dissimilarity = pd.merge(doc2vec_dissimilarity, tfidf_dissimilarity, left_index=True, right_index=True)
    ensemble_dissimilarity.columns = ["doc2vec_dissimilarity", "tfidf_dissimilarity"]
    ensemble_dissimilarity['ensemble_dissimilarity'] = (ensemble_dissimilarity["doc2vec_dissimilarity"] + ensemble_dissimilarity["tfidf_dissimilarity"])/2
    ensemble_dissimilarity.sort_values(by="ensemble_dissimilarity", ascending=False, inplace=True)
    return ensemble_dissimilarity


def find_similar_perfumes(text, n):
    love_message, hate_message = get_sentiment(text)
    similar_perfumes = get_ensemble_similarity_scores(love_message)
    dissimilar_perfumes = get_dissimilarity_scores(hate_message)
    dissimilar_perfumes = dissimilar_perfumes[dissimilar_perfumes['ensemble_dissimilarity'] > .3]
    similar_perfumes = similar_perfumes.drop(dissimilar_perfumes.index).reset_index()
    similar_perfumes = similar_perfumes.drop_duplicates(subset='Name', keep='first')

    return similar_perfumes.iloc[:n, :]

def view_recommendations(recommended_perfumes, n):
    fig, axes = plt.subplots(nrows=1, ncols=n, figsize=(15,10))
    ax = axes.ravel()

    for i in range(len(recommended_perfumes)):
        single_title = recommended_perfumes.Name.tolist()[i]
        single_perfume = df[df['Name']==single_title]
        name = single_perfume.Name.values[0].encode("ascii", errors="ignore").decode()
        notes = single_perfume.Notes.values[0].encode("ascii", errors="ignore").decode()
        title = "{} \n Notes: {}".format(name, notes)

        perfume_image = single_perfume['Image URL'].values[0]
        image = io.imread(perfume_image)
        ax[i].imshow(image)
        ax[i].set_yticklabels([])
        ax[i].set_xticklabels([])
        ax[i].set_title("\n".join(wrap(title, 20)))
        ax[i].axis('off')

    plt.show()

