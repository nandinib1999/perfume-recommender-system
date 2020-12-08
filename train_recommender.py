import pandas as pd
import matplotlib.pyplot as plt
from skimage import io
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import re
import gensim
from gensim.test.utils import get_tmpfile
import nltk
import os
import time
import utils as ut


def train_models(input_data_path='final_perfume_data_temp.csv', model_dir="models"):
	start_time = time.time()
	df1 = pd.read_csv(input_data_path, engine='python')
	df1['Notes'].fillna(' ', inplace=True)
	df1 = ut.preprocess_columns(df1, "Notes", [ut.make_lower_case, ut.remove_punctuation, ut.rem_numbers, ut.stem_words])
	df1 = ut.preprocess_columns(df1, 'Description', [ut.make_lower_case, ut.remove_punctuation, ut.decontractions, ut.remove_stop_words, ut.stem_words, ut.fullstops, ut.rem_numbers])

	df1['all_details'] = df1['Description'] + " " + df1['Notes']

	# This is where our pickle files will be stored
	if not os.path.exists(model_dir):
		os.mkdir(model_dir)

	print(df1['Notes'])
	print(df1['Description'])

	# ****************************************************************************
	print('Tf-IDF Vectorization...')
	tf = TfidfVectorizer(analyzer='word', 
	                     min_df=10,
	                     ngram_range=(1, 2),
	                     stop_words='english')
	tf.fit(df1['all_details'])
	tfidf = tf.transform(df1['all_details'])
	print("Shape of TFIDF vectorizer ", tfidf.shape)
	print('Saving the vectorizer')
	with open(os.path.join(model_dir, "tf-idf_vectorizer.pkl"), "wb") as pkl_head:
		pickle.dump(tf, pkl_head)

	# ****************************************************************************
	print()
	print('Dimensionality Reduction using SVD...')
	svd = TruncatedSVD(n_components=2000)
	latent_matrix = svd.fit_transform(tfidf)
	print("Shape of the Latent Matrix ", latent_matrix.shape)
	print('Saving the SVD Model..')
	with open(os.path.join(model_dir, "svd_model.pkl"), "wb") as pkl_head:
		pickle.dump(svd, pkl_head)

	# ****************************************************************************
	doc_labels = df1.Name
	svd_feature_matrix = pd.DataFrame(latent_matrix ,index=doc_labels)
	print("Shape of the LSA Feature Matrix ", svd_feature_matrix.shape)
	print('Saving the LSA Embeddings Model..')
	with open(os.path.join(model_dir, "lsa_embeddings.pkl"), "wb") as pkl_head:
		pickle.dump(svd_feature_matrix, pkl_head)


	# ****************************************************************************
	descriptions = df1.Description.values.tolist()
	print()
	print("Gensim doc2vec Model..")

	documents = []
	for i in range(len(df1)):
	    mystr = descriptions[i]
	    documents.append(re.sub("[^\w]", " ",  mystr).split())

	formatted_documents = [gensim.models.doc2vec.TaggedDocument(doc, [i]) for i, doc in enumerate(documents)]

	model = gensim.models.doc2vec.Doc2Vec(vector_size=512, min_count=5, epochs=1000, seed=0, window=3, dm=1)
	model.build_vocab(formatted_documents)

	model.train(formatted_documents, total_examples=model.corpus_count, epochs=model.epochs)

	fname = get_tmpfile(os.path.join(model_dir, "doc2vec_model"))
	model.save(os.path.join(model_dir, "doc2vec_model"))
	model = gensim.models.doc2vec.Doc2Vec.load(os.path.join(model_dir, "doc2vec_model"))


	# ****************************************************************************
	doc2vec_feature_matrix = pd.DataFrame(model.docvecs.vectors_docs, index=df1.Name)
	print("Shape of the doc2vec Feature Matrix ", doc2vec_feature_matrix.shape)

	print("Saving the doc2vec Embeddings Model..")
	with open(os.path.join(model_dir, "doc2vec_embeddings.pkl"), "wb") as pkl_head:
		pickle.dump(doc2vec_feature_matrix, pkl_head)

	total_time = (time.time() - start_time)/60
	print()
	print("Training Completed.. Total time taken ", str(total_time))