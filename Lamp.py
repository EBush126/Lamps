from asyncio.windows_utils import pipe
from copyreg import pickle
from json import load
import numpy as np
import pandas as pd
from sklearn import pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix,  accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle
from pickle import load

data_file = 'Lamp_imputed.csv'

# This function loads the csv file, splits the data into train/test splits
# parameters: movie reviews csv filename
# return: same as the return values of the sklearn train_test_split function  
def load_split(data_file):
    df = pd.read_csv(data_file)
    X = df.loc[:,['Review']]
    y = df.IsRecommended
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,stratify=y)
    return X_train, X_test, y_train, y_test


# This function trains a linearSVC model from the training data you obtained from the splits
# parameters: X_train, y_train
# return: mean cross validation accuracy and your full pipeline object 
# that has your trained vectorizer and classifer 
def train(X_train, y_train):
    X_train_docs = [doc for doc in X_train.Review]
    pipeline = Pipeline([
        ('vect', TfidfVectorizer(ngram_range = (1, 3), stop_words = "english", max_features = 1000)),
        ('pac', PassiveAggressiveClassifier(C = 6, average = True, class_weight = 'balanced', fit_intercept = False, shuffle =  True, tol = 1e-07))
    ])
    pipeline.fit(X_train_docs, y_train)
    pickle.dump(pipeline, open('pipeline.pkl', 'wb'))
    #joblib.dump(pipeline, 'pipeline.pkl')
    scores = cross_val_score(pipeline, X_train_docs, y_train, cv=5)
    mean_cross_val_accuracy = np.mean(scores)
    return mean_cross_val_accuracy, pipeline

#with open('pipeline.pkl', 'wb') as f:
#    pickle.dumps(pipeline, f)


#pickle.dump(pipeline, open('pipeline.pkl', 'wb'))
# saving only when want to use pipeline to other dataset

# This function validates your model by applying your vectorizer and model to the test set
# parameters: X_test, y_test, pipeline
# return: test accuracy
def validate(X_test, y_test, pipeline):
    X_test_docs = [doc for doc in X_test.Review]
    y_test_pred = pipeline.predict(X_test_docs)
    test_accuracy =  metrics.accuracy_score(y_test, y_test_pred)
    cm = confusion_matrix(y_test, y_test_pred)
    return test_accuracy, cm

# Applying pipeline

def predict_isRecommended(to_predict):
    with open('pipeline.pkl', 'rb') as f:
        pipe = pickle.load(f)

    #for review in data_file:
    X_docs = [doc for doc in to_predict.Review]
    predict_isRecommended = pipe.predict(X_docs)
    #to_predict['Prediction'] = predict_isRecommended
    predict_isRecommended=pd.Series(predict_isRecommended)
    df = pd.DataFrame(
        {'ProductID': to_predict['ProductID'],
        'Review': to_predict['Review'],
        'Rating': to_predict['Rating'],
        'IsRecommended': to_predict['IsRecommended'],
        'Prediction': predict_isRecommended}
    )
    return df

