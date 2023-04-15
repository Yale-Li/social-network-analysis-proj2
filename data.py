import string
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# define a function to preprocess the text
def preprocess_text(text):
    # remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # convert to lowercase
    text = text.lower()
    # remove whitespace
    text = text.strip()
    return text


def bagofwords(train, test):

    # create a CountVectorizer object
    vectorizer = CountVectorizer()

    vectorizer.fit(train)

    # transform the data into bag-of-words representations
    train_bow = vectorizer.transform(train)
    test_bow = vectorizer.transform(test)
    return (train_bow, test_bow)


def tfidf(train, test):
    # create a TfidfVectorizer object
    tfidf = TfidfVectorizer()

    # fit the vectorizer to the training data and transform the data
    train_tfidf = tfidf.fit_transform(train)

    # transform the testing data using the fitted vectorizer
    train_tfidf = tfidf.transform(test)
    test_tfidf = tfidf.transform(test)
    
    return (train_tfidf, test_tfidf)
