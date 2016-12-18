import os
from pprint import pprint
from time import time
from datetime import datetime

import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import Activation
from keras.layers.advanced_activations import PReLU, ParametricSoftplus
from keras.layers.core import Dropout
from keras.layers.core import MaxoutDense
from keras.layers.noise import GaussianNoise
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1, l2
from keras.optimizers import Nadam
from keras.optimizers import Adadelta

from keras.callbacks import Callback
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint

from keras import backend as K


def exp(x):
    return K.exp(x)

def expmae(y_true, y_pred):
    return K.mean(K.abs(K.exp(y_true) - K.exp(y_pred)))


def modified_reference_model(input_dim):
    model = Sequential()
    model.add(Dense(600, input_dim=input_dim, init='he_normal'))
    model.add(Activation('tanh'))
    model.add(BatchNormalization(mode=0))
    model.add(Dropout(0.3))
    model.add(Dense(200, init='he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization(mode=0))
    model.add(Dropout(0.2))
    # model.add(MaxoutDense(output_dim=20, nb_feature=12))
    model.add(Dense(100, init='he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization(mode=0))
    model.add(Dropout(0.2))
    model.add(Dense(50, init='he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization(mode=0))
    model.add(Dropout(0.2))
    model.add(Dense(1, init='he_normal'))
    model.add(Activation('softplus'))
    # model.add(Activation('relu'))
    # model.add(Activation(exp))
    # model.add(ParametricSoftplus(alpha_init=0.99, beta_init=0.0001))
    optimizer = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
    # optimizer = Nadam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.00001)
    model.compile(loss='mae', optimizer=optimizer)
    print(model.summary())
    return(model)


class BagDNN(object):
    def __init__(self, base_model_folder):
        self.base_model_folder = base_model_folder
        self.base_model_fnames = []
        self.errors = []

    def fit(self, train_x, train_y, n_folds=5, random_state=1234):
        self.base_model_fnames = []
        for k in range(3):
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state+k)
            now = datetime.now().strftime("%Y%m%d%H")
            self.errors = []
            for idx, (subtr_idx, valid_idx) in enumerate(kf.split(train_x)):
                subtrain_x = train_x[subtr_idx, :]
                subtrain_y = train_y[subtr_idx]
                validation_x = train_x[valid_idx, :]
                validation_y = train_y[valid_idx]

                best_model_fname = self.base_model_folder + 'modified_reference_model_' + now + '_fold_' + str(idx) + '.h5'
                model, mae = self._train_model(subtrain_x, subtrain_y, validation_x, validation_y, best_model_fname)
                self.errors.append(mae)
        print('mae of each fold:')
        pprint(self.errors)

    def _train_model(self, subtrain_x, subtrain_y, validation_x, validation_y, best_model_fname):
        input_dim = subtrain_x.shape[1]
        print(input_dim)
        model = modified_reference_model(input_dim)
        # best_model_fname = '../modified_reference_model.h5'
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            verbose=1,
            factor=0.3,
            patience=5,
            cooldown=3,
            min_lr=1e-8
        )
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=24,
            verbose=1,
        )
        model_cp = ModelCheckpoint(
            best_model_fname,
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
        )
        # subtrain_y = np.log(subtrain_y + 200)
        # raw_validation_y = validation_y
        # validation_y = np.log(validation_y + 200)
        model.fit(
            subtrain_x,
            subtrain_y,
            nb_epoch=500,
            batch_size=64,
            validation_data=(validation_x, validation_y),
            callbacks=[reduce_lr, early_stopping, model_cp],
        )
        model = load_model(best_model_fname)
        validation_pred = model.predict(validation_x, verbose=0)
        loss = mean_absolute_error(validation_y, validation_pred)
        self.base_model_fnames.append(os.path.abspath(best_model_fname))
        return model, loss

    def predict(self, test_x):
        predictions = self.predict_individually(test_x)
        predictions_mean = np.mean(predictions, axis=1)
        return predictions_mean

    def predict_individually(self, test_x):
        t1 = time()
        models = [load_model(fname) for fname in self.base_model_fnames]
        t2 = time()
        print('Load model: %.2f seconds' % (t2 - t1))
        predictions = []
        for model in models:
            current_prediction = model.predict(test_x, verbose=1)
            # current_prediction = np.exp(current_prediction) - 200
            predictions.append(current_prediction.flatten())
        predictions = np.column_stack(predictions)
        return predictions
