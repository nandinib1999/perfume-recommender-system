import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import RegexpTokenizer

def stem_words(text):
    text = text.split()
    stemmer = SnowballStemmer('english')
    stemmed_words = [stemmer.stem(word) for word in text]
    text = " ".join(stemmed_words)
    return text

def make_lower_case(text):
    return text.lower()

def remove_stop_words(text):
    text = text.split()
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops]
    text = " ".join(text)
    return text

def remove_punctuation(text):
    tokenizer = RegexpTokenizer(r'\w+')
    text = tokenizer.tokenize(text)
    text = " ".join(text)
    return text

def preprocess_message(message):
    message = make_lower_case(message)
    message = remove_stop_words(message)
    message = remove_punctuation(message)
    message = stem_words(message)
    return message

def preprocess_columns(dataframe, column_name, preprocess_funcs):
    for preprocess_func in preprocess_funcs:
        dataframe[column_name] = dataframe[column_name].apply(func=preprocess_func)

    return dataframe