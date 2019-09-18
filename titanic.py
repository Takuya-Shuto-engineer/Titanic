import numpy as np
from numpy import random
import pandas as pd
import os
import os.path as path
from argparse import ArgumentParser
from sklearn import preprocessing

import tensorflow as tf
import keras
from keras import backend as K, regularizers
from tensorflow import name_scope as scope
from keras.layers import BatchNormalization, Input, Activation, Dense
from keras.models import Model
from keras.callbacks import CSVLogger, EarlyStopping
from keras.regularizers import l2

def model(shape = []):
    with scope("input") as _n:
        _i = Input(shape = shape, name = _n)
    with scope("Layer01") as _n:
        with scope("FC") as _n:
            _x = Dense(32, kernel_regularizer = l2(0.02), name = _n)(_i)
        with scope("Activation") as _n:
            _x = Activation("relu", name = _n)(_x)
        with scope("BatchNormalization") as _n:
            _x = BatchNormalization(name = _n)(_x)
    with scope("Layer02") as _n:
        with scope("FC") as _n:
            _x = Dense(1, kernel_regularizer = l2(0.02), name = _n)(_x)
        with scope("Activation") as _n:
            _x = Activation("relu", name = _n)(_x)
        with scope("BatchNormalization") as _n:
            _x = BatchNormalization(name = _n)(_x)
    with scope("Output") as _n:
        _o = Activation("sigmoid", name = _n)(_x)
    return Model(_i, _o)

def damage_table(df):
    _null_idx = df.isnull().sum()
    _percent = 100 * df.isnull().sum() / len(df)
    _damage_table = pd.concat([_null_idx, _percent], axis = 1)
    _damage_table_rename_columns = _damage_table.rename(columns = {0: "欠損値", 1: "%"})
    return _damage_table_rename_columns

# Argument Parse
parser = ArgumentParser()
parser.add_argument("--outdir", type = str, required = True, help = "出力先のディレクトリ")
parser.add_argument("--epochs", type = int, required = True, help = "エポック数")
parser.add_argument("--threshold", type = float, required = True, help = "閾値")
parser.add_argument("--patience", type = int, required = True, help = "指定回数epoch回してもlossが減らない場合早期終了")
parser.add_argument("--batchsize", type = int, required = True, help = "バッチサイズ")
args = parser.parse_args()
    
# preprocessing
_train = pd.read_csv("train.csv")
_test = pd.read_csv("test.csv")

_train_damage = damage_table(_train)
_test_damage = damage_table(_test)

_train["Age"] = _train["Age"].fillna(random.normal(_train["Age"].median(), _train["Age"].std()))
_train["Embarked"] = _train["Embarked"].fillna("S")

_train_damage = damage_table(_train)

_train["Sex"][_train["Sex"] == "male"] = 0
_train["Sex"][_train["Sex"] == "female"] = 1
_train["Embarked"][_train["Embarked"] == "S"] = 0
_train["Embarked"][_train["Embarked"] == "C"] = 1
_train["Embarked"][_train["Embarked"] == "Q"] = 2

_test["Age"] = _test["Age"].fillna(random.normal(_test["Age"].median(), _test["Age"].std()))
_test["Sex"][_test["Sex"] == "male"] = 0
_test["Sex"][_test["Sex"] == "female"] = 1
_test["Embarked"][_test["Embarked"] == "S"] = 0
_test["Embarked"][_test["Embarked"] == "C"] = 1
_test["Embarked"][_test["Embarked"] == "Q"] = 2
_test.Fare[152] = random.normal(_test.Fare.median(), _test.Fare.std())

_mm = preprocessing.MinMaxScaler()

# Extract features and labels
_train_x = _train[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "Embarked"]].values
_train_x = _mm.fit_transform(_train_x)
_train_y = _train["Survived"].values
_test_x = _test[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "Embarked"]].values
_test_x = _mm.fit_transform(_test_x)
# 正解はわからない
# _test_y = _test["Survived"].values 

_config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

with tf.Session(config = _config) as _session:
    K.set_session(_session)
    _model = model(shape = (_train_x.shape[1],))
    _model.summary()
    _model.compile(optimizer = "Adam", loss = "binary_crossentropy", metrics = [])
    _callbacks = []
    _callbacks.append(CSVLogger(path.join(args.outdir, "train.log")))
    _callbacks.append(EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True))
    
    _model.fit(x = _train_x, y = _train_y, batch_size = args.batchsize, epochs = args.epochs, verbose = 1, callbacks = _callbacks, shuffle = True)
    _model.save(path.join(args.outdir, "model.h5"))
    
    # predict
    _prediction = _model.predict(_test_x)
    for _i, _elm in enumerate(_prediction):
        if _elm > args.threshold:
            _prediction[_i] = 1
        elif _elm <= args.threshold:
            _prediction[_i] = 0
        else:
            _prediction[_i] = -1
    _prediction = np.reshape(_prediction, (_prediction.shape[0], ))
    print(_prediction)
    
    _passenger_id = np.array(_test["PassengerId"]).astype(int)
    
    _result = pd.DataFrame(_prediction.astype(int), _passenger_id, columns = ["Survived"])
    _result.to_csv(path.join(args.outdir, "titanic_nn.csv"), index_label = ["PassengerId"])
    