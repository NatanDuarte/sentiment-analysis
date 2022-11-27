import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

from nltk import tokenize, FreqDist

from pickle import dump


def classify_text(text, text_column, classification_column):
    vectorizer = CountVectorizer(max_features=50)
    bag_of_words = vectorizer.fit_transform(text[text_column])

    train, test, train_class, test_class = train_test_split(
        bag_of_words, text[classification_column])

    logistic_regressor = LogisticRegression()
    logistic_regressor.fit(train, train_class)

    return (
        logistic_regressor.score(test, test_class),
        logistic_regressor,
        vectorizer
    )


reviews = pd.read_csv('dataset/imdb-reviews-pt-br.csv')
reviews['classification'] = reviews['sentiment'].replace(
    ['neg', 'pos'], [0, 1])

accuracy, model, vec = classify_text(reviews, 'text_pt', 'classification')

if accuracy >= 0.6:
    with open("models/regressor_model.pkl", "wb") as file:
        dump(model, file, protocol=None, fix_imports=True)
        print("model saved successfully")
    with open("models/regressor_vectorizer.pkl", "wb") as file:
        dump(vec, file, protocol=None, fix_imports=True)
        print("vectorizer saved successfully")
else:
    print('model has no enough performance')

print(f'current accuracy: {(accuracy * 100):.2f}%')
