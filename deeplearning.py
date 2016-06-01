import pandas as pd
import numpy as np
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import train_test_split
from sklearn.metrics import *
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import *
from sklearn.feature_selection import SelectKBest
from keras.models import Sequential
from keras.layers.core import *
from keras.optimizers import Adadelta
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn import decomposition, pipeline, metrics, grid_search


scores = ['precision', 'recall', 'f-score', 'support']
def get_cross_validation_scores(clf, X, y, n_folds=2, labels=['M', 'R'], NN=False):
    cross_validation_scores = np.zeros((len(labels) + 1, len(scores), n_folds))
    i = 0
    for train, test in StratifiedKFold(y, n_folds=n_folds):
        clf.fit(X[train], y[train])
        if NN:
            y_pred = model.predict_classes(X[test])
        else:
            y_pred = clf.predict(X[test])
        sc = precision_recall_fscore_support(y[test], y_pred, labels=labels)
        tsc = precision_recall_fscore_support(y[test], y_pred)
        for s in range(len(scores)):
            for l in range(len(labels) + 1):
                if (l == len(labels)):
                    continue
                cross_validation_scores[l][s][i] = sc[s][l]
        cross_validation_scores[len(labels)][0][i] = precision_score(y[test], y_pred)
        cross_validation_scores[len(labels)][1][i] = recall_score(y[test], y_pred)
        cross_validation_scores[len(labels)][2][i] = f1_score(y[test], y_pred)
        i += 1        
    return cross_validation_scores

def mean(x): return "%.5f" % (sum(x) / len(x))
def cross_validation_report(cv, labels):
    row_format ="{:>15}" * (len(scores) + 1)
    print row_format.format("", *scores)
    for l in range(len(labels)):
        print row_format.format(labels[l], *map(mean, cv[l]))
    cv_tr = np.zeros((len(scores)))
    for s in range(len(scores)):
        cv_tr[s] = 0.0
        for l in range(len(labels)): cv_tr[s] += float(map(mean, cv[l])[s])
    cv_tr = map(lambda x: x / len(labels), cv_tr)
    print row_format.format("total/av", *cv_tr)


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train_ready = train
test_ready = test

def create_model():
    model = Sequential()
    model.add(Dense(64, input_dim=5, init='uniform', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
    return model

model = KerasClassifier(build_fn=create_model, nb_epoch=50)

q = train_ready[["206_USAGE","207_USAGE","208_USAGE","209_USAGE","210_USAGE","TARGET", "CONTRACT_KEY"]]
# q["TARGET"] = q["TARGET"].apply(lambda x: True if x == 1 else False)
#grid = GridSearchCV(estimator=model, param_grid=dict(n_estimators=ns))
X = q[(q.keys().difference(['TARGET', "CONTRACT_KEY"]))].values
y = q["TARGET"].values
#y.shape = (len(y), 1)


# svd = TruncatedSVD(n_components=200)
scl = StandardScaler()
#rbm = BernoulliRBM()
# kbst = SelectKBest(k=100)
mdl = model
# Create the pipeline 
clf = pipeline.Pipeline([('scl', scl), ('svm', mdl)])

labels = [0, 1]
cv = get_cross_validation_scores(clf, X, y, labels=labels)
cross_validation_report(cv, labels)
