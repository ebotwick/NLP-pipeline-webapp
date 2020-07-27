import sys
import sqlalchemy
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import nltk
import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix, precision_score, recall_score
import pickle
# import warnings


def load_data(database_filepath):
    engine = create_engine('sqlite:////home/workspace/{}'.format(database_filepath))
    conn = engine.connect()
    df = pd.read_sql_table('messagesCategorizedCommandLine', conn)
    X = df['message']
    Y = df.iloc[:,4:] 
    cat_names = Y.columns
    return X, Y, cat_names


def tokenize(text):
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokenized = nltk.word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokes = []
    
    for tok in tokenized:
        cleaned = lemmatizer.lemmatize(tok).strip()
        clean_tokes.append(cleaned)
    
    return clean_tokes


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('moc', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
#     pipeline.fit(Xtrain, Ytrain)
    
                      
    return pipeline                 
def evaluate_model(pipeline, Xtest, Ytest, cat_names):
    yhat = pipeline.predict(Xtest)                        
    for i in range(len(Ytest.columns)):
        print("Performance for {} category".format(Ytest.columns[i]))
        print(classification_report(Ytest.iloc[:,i], yhat[:,i]))
    
                            
    preclist = []
    reclist = []
    f1list = []
    for i in range(len(Ytest.columns)):
        preclist.append(precision_score(Ytest.iloc[:,i], yhat[:,i]))
        reclist.append(recall_score(Ytest.iloc[:,i], yhat[:,i]))
        f1list.append(f1_score(Ytest.iloc[:,i], yhat[:,i]))        
                            
    print("Average Precision: ", np.around(np.mean(preclist),3))
    print("Average Recall: ", np.around(np.mean(reclist),3))
    print("Average F1: ", np.around(np.mean(f1list),3))                      


def save_model(pipeline, model_filepath):
    
    pickle.dump(pipeline, open(model_filepath, 'wb'))
    return print('Model pickled!')

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, cat_names = load_data(database_filepath)
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2)
        nltk.download(['punkt', 'wordnet'])
        print('Building model...')
        pipeline = build_model()
        
        print('Training model...')
        pipeline.fit(Xtrain, Ytrain)
        
        print('Evaluating model...')
        evaluate_model(pipeline, Xtest, Ytest, cat_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(pipeline, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
