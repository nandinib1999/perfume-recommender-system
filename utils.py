import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re

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

def decontractions(text):
    '''Performs decontractions in the doc'''
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"couldn\'t", "could not", text)
    text = re.sub(r"shouldn\'t", "should not", text)
    text = re.sub(r"wouldn\'t", "would not", text)
    # general
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)

    return text

def rem_numbers(text):
    '''Removes numbers and irrelevant text like xxxx*'''
    text = re.sub(r'\d','',text)
    return text

def fullstops(text):
    text = re.sub('\.', ' .', text)
    return text

def remove_punctuation(text):
    punctuations = '''!()-[]{};:'"\,<>/?@#$%^&*~''' # full stop is not removed
    new_text = []
    for word in text.split(' '):
        if word in punctuations: 
            pass
        else:
            new_text.append(word)
    return ' '.join(new_text)

def preprocess_message(message):
    message = make_lower_case(message)
    message = remove_stop_words(message)
    message = remove_punctuation(message)
    message = decontractions(message)
    message = fullstops(message)
    message = stem_words(message)
    return message

def preprocess_columns(dataframe, column_name, preprocess_funcs):
    for preprocess_func in preprocess_funcs:
        dataframe[column_name] = dataframe[column_name].apply(func=preprocess_func)

    return dataframe