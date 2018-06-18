import pandas as pd
import numpy as np
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer

def open_file(file, path):
    """
    Wrapper to open files in path
    :param file: (str) file name
    :param path: (str) path to directory with json files
    :return: list of dictionaries combining all documents in corpus
    """
    f = open(path + file)
    file_content = f.read()
    return(json.loads(file_content))

def clean_text(text):
    """
    Fxn to clean text column
    :param text: str with raw text
    :return: str of text with replaced values
    """
    # strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('-', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = re.sub(r'[0-9]+', ' ', text)
    text = re.sub(re.compile('[^A-Za-z0-9 ]+'), '', text) # strip special chars
    text = text.strip(' ')
    return text

def multiple_replace(text, adict):
    """
    Fxn to replace multiple substrings of a string
    :param text: str of input text
    :param adict: dictionary with conditions
    :return: str with replaced values
    """
    rx = re.compile('|'.join(map(re.escape, adict)))
    def one_xlat(match):
        return adict[match.group(0)]
    return rx.sub(one_xlat, text)

adict = {
      "(" : "",
      ")" : "",
      "-" : ""
        }

def get_tfidf_dta(df,variable):
    """
    Fxn to create binarised DataFrame with most important terms in a text column
    :param df: DataFrame
    :param variable: text column
    :return: DataFrame with most important terms and respective tfid values
    """
    Text = [multiple_replace(w,adict) for w in df[variable].tolist()]
    v = TfidfVectorizer(input=Text, ngram_range=(1, 1), min_df=0.01, stop_words=stop_words)
    mat = v.fit_transform(Text)
    m = pd.DataFrame(mat.todense())
    m.columns = v.get_feature_names()
    m = df[[variable]].join(m)
    return(m)

def top_tfidf_feats(row, features, top_n=25):
    """
    Fxn to get top n tfidf values in row and return them with their corresponding feature names.
    :param features: list of features
    :param top_n: integer specifying the
    :return: DataFrame with top n features and tfidf values
    """
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return(df)