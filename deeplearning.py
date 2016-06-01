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
from sklearn.linear_model import LinearRegression
from keras.optimizers import SGD 



def encode_onehot(df, cols):
    """
    One-hot encoding is applied to columns specified in a pandas DataFrame.
    
    Modified from: https://gist.github.com/kljensen/5452382
    
    Details:
    
    http://en.wikipedia.org/wiki/One-hot
    http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
    
    @param df pandas DataFrame
    @param cols a list of columns to encode
    @return a DataFrame with one-hot encoding
    """
    vec = DictVectorizer()
    
    vec_data = pd.DataFrame(vec.fit_transform(df[cols].to_dict(outtype='records')).toarray())
    vec_data.columns = vec.get_feature_names()
    vec_data.index = df.index
    
    df = df.drop(cols, axis=1)
    df = pd.concat([df, vec_data], axis=1)
    return df



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
contract = pd.read_csv("contract_ref.csv")
train_ready = train
test_ready = test



def convert_gender(g):
        return {"Female": "F", "f": "F", "F": "F", "Male": "M", "m": "M", "M":"M",
                "Not Entered":"Unknown","Unknown":"Unknown"}[g]
contract["GENDER"] = contract["GENDER"].apply(convert_gender)
contract["VALUE_SEGMENT"][pd.isnull(contract["VALUE_SEGMENT"])] = "Unknown"
contract["HANDSET_NAME"][pd.isnull(contract["HANDSET_NAME"])] = "Other"
contract["AGE"][contract["AGE"] < 10] = 99
contract["AGE"][contract["AGE"] == 99] = contract["AGE"].median()
#contract["APPLE"] = contract["HANDSET_NAME"].apply(lambda x: x.lower().count("apple") > 0)
#contract = contract.drop(["HANDSET_NAME"], axis=1)
def convert_segment(s):
    return {"Core": 1, "Med-Low": 2, "Med-High": 3, "High":4,
           "Premium": 5, "Platinum": 6, "Unknown": 0}[s]
contract["VALUE_SEGMENT"] = contract["VALUE_SEGMENT"].apply(convert_segment)
contract = encode_onehot(contract, ["GENDER"])
contract["AGE_UNKNOWN"] = contract["AGE"] == 99
contract = encode_onehot(contract, ["HANDSET_NAME"])
# contract["APPLE"] = contract["HANDSET_NAME"].apply(lambda x: x.lower().count("apple") > 0)
# contract["GALAXY"] = contract["HANDSET_NAME"].apply(lambda x: x.lower().count("galaxy") > 0)
contract = contract.drop(["RATE_PLAN"], axis=1)


def feature_engineer_usage(ds):
    global contract
    cols = ["206_USAGE", "207_USAGE", "208_USAGE", "209_USAGE", "210_USAGE"]
    s_cols = ["206_SESSION_COUNT", "207_SESSION_COUNT", "208_SESSION_COUNT", "209_SESSION_COUNT", "210_SESSION_COUNT"]
    ds["211_USAGE"] = 0
    ds["211_SESSION_COUNT"] = 0
    ds["tmp"] = ''
    ds["s_tmp"] = ''
    def predict(s, m):
        vals = map(int, s.split(",")[:-1])
        reg = LinearRegression()
        t = np.array(range(0, 5))
        t.shape = (len(t), 1)
        reg.fit(t, vals)
        return reg.predict(m)
#         return np.poly1d(np.polyfit(t, vals, 3))(6)
    for i in cols: ds['tmp'] += ds[i].astype('string') + ','
    for i in s_cols: ds['s_tmp'] += ds[i].astype('string') + ','
    
    ds['211_USAGE'] = ds.apply(lambda row: predict(row['tmp'], 6), axis=1)
    ds['211_SESSION_COUNT'] = ds.apply(lambda row: predict(row['s_tmp'], 6), axis=1)
    ds["MEAN"] = 0
    ds["s_MEAN"] = 0
    for i in cols: ds["MEAN"] += ds[i]
    for i in s_cols: ds["s_MEAN"] += ds[i]
    ds["MEAN"] /= 5.0
    ds["s_MEAN"] /= 5.0
#     df = train[["CONTRACT_KEY", "211_USAGE", "MEAN","TARGET", "210_USAGE"]]
    df = ds
    df = ds.drop(["tmp","s_tmp", "MEAN", "s_MEAN"], axis=1)
#     df.merge(contract)
    return df

train_ready = feature_engineer_usage(train)
test_ready = feature_engineer_usage(test)
train_ready.merge(contract)


q = train_ready
# q["TARGET"] = q["TARGET"].apply(lambda x: True if x == 1 else False)
#grid = GridSearchCV(estimator=model, param_grid=dict(n_estimators=ns))
X = q[(q.keys().difference(['TARGET', "CONTRACT_KEY"]))].values
y = q["TARGET"].values
#y.shape = (len(y), 1)



def create_model():
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model = Sequential()
    model.add(Dense(100, input_dim=X.shape[1], init='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
    return model

model = KerasClassifier(build_fn=create_model, nb_epoch=500, batch_size=100)


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
