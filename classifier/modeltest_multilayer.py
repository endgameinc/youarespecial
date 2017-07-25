import common
import numpy as np

# this will take a LONG time the first time you run it (and cache features to disk for next time)
# it's also chatty.  Parts of feature extraction require LIEF, and LIEF is quite chatty.
# the output you see below is *after* I've already run feature extraction, so that
#   X and sample_index are being read from cache on disk
X, y, sha256list = common.extract_features_and_persist() 

# split our features, labels and hashes into training and test sets
from sklearn.model_selection import train_test_split
np.random.seed(123)
X_train, X_test, y_train, y_test, sha256_train, sha256_test = train_test_split(
    X, y, sha256list, test_size=1000)

# StandardScaling the data can be important to multilayer perceptron
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_train)

###########
# sanity check: random forest classifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
rf = RandomForestClassifier(
    n_estimators=40, n_jobs=-1, max_depth=30).fit(X_train, y_train)
y_pred = rf.predict_proba(X_test)[:,-1] # get probabiltiy of malicious (last class == last column )
common.summarize_performance(y_pred, y_test, "RF Classifier")
from sklearn.externals import joblib
joblib.dump(rf, 'random_forest.pkl')

# simple multilayer perceptron
X_train = scaler.transform(X_train) # scale for multilayer perceptron
X_test = scaler.transform(X_test)

import simple_multilayer
from keras.callbacks import LearningRateScheduler
model = simple_multilayer.create_model(input_shape=(
    X_train.shape[1], ), input_dropout=0.1, hidden_dropout=0.1, hidden_layers=[4096, 2048, 1024, 512])
model.fit(X_train, y_train,
          batch_size=128,
          epochs=20,
          verbose=1,
          callbacks=[LearningRateScheduler(
              lambda epoch: common.schedule(0.2, 0.5, 5))],
          validation_data=(X_test, y_test))
y_pred = model.predict(X_test)
common.summarize_performance(y_pred, y_test, "Multilayer perceptron")
model.save('multilayer.h5')
