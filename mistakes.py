import numpy as np
import pandas as pd

def get_mistakes(clf, X_q1q2, y):

    predictions = np.around(clf.predict(X_q1q2)).astype(int)   
    incorrect_predictions = predictions!=y
    incorrect_indices = np.where(incorrect_predictions)[0]
    
    if np.sum(incorrect_predictions)==0:
        print("no mistakes in this df")
    else:
        return incorrect_indices, predictions
    
    
def print_mistake_k(k, mistake_indices, predictions, df):
    print(df.iloc[mistake_indices[k]].question1)
    print(df.iloc[mistake_indices[k]].question2)
    print("true class:", df.iloc[mistake_indices[k]].is_duplicate)
    print("prediction:", predictions[mistake_indices[k]])
    
    
def print_mistake_k_and_tokens(k, mistake_indices, predictions,
                               X_q1q2, count_vect, clf, df):
    q1 = df.iloc[mistake_indices[k]].question1
    q2 = df.iloc[mistake_indices[k]].question2
    
    print(q1)
    print(count_vect.tokenize(q1))
    print()
    print(q2)
    print(count_vect.tokenize(q2))
    print()
    print("true class:", df.iloc[mistake_indices[k]].is_duplicate)
    print("prediction:", predictions[mistake_indices[k]])
    print()
    print("Probability vector: [P(0|x), P(1|x)]:")
    print(clf.predict_proba(X_q1q2)[mistake_indices[k],:])
    
    
def hist_errors(mistake_indices, predictions,
                               X_q1q2, count_vect, clf, df):
    qs = df.iloc[mistake_indices][['question1', 'question2']]
    qs['true_class']=df.iloc[mistake_indices].is_duplicate
    qs['prediction']=predictions[mistake_indices]
    qs['P(1|x)']=clf.predict_proba(X_q1q2)[mistake_indices,:][:,1]
    qs = qs.reset_index(drop=True)
    return qs
