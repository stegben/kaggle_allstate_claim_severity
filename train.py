import sys

import numpy as np
import pandas as pd

import joblib

from sklearn.model_selection import KFold

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers.advanced_activations import PReLU
from keras.layers.core import Dropout
from keras.layers.noise import GaussianNoise
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1, l2
from keras.optimizers import Nadam


NB_EPOCH_MODIFIED_REFERENCE_MODEL = 50
NB_EPOCH_REFERENCE_MODEL = 50
NB_EPOCH_MY_MODEL = 50


def reference_model(input_dim):
    model = Sequential()
    model.add(Dense(400, input_dim=input_dim, init='he_normal'))
    model.add(PReLU())
    model.add(Dropout(0.4))
    model.add(Dense(200, init='he_normal'))
    model.add(PReLU())
    model.add(Dropout(0.2))
    model.add(Dense(1, init='he_normal'))
    model.compile(loss='mae', optimizer='adadelta')
    return(model)


def modified_reference_model(input_dim):
    model = Sequential()
    model.add(Dense(300, input_dim=input_dim, init='he_normal'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.4))
    model.add(Dense(200, init='he_normal'))
    model.add(PReLU())
    model.add(Dropout(0.4))
    model.add(Dense(200, init='he_normal'))
    model.add(PReLU())
    model.add(Dropout(0.4))
    model.add(Dense(1, init='he_normal'))
    model.add(Activation('softplus'))
    model.compile(loss='mae', optimizer='adadelta')
    return(model)


def my_model(input_dim):
    model = Sequential([
        Dense(1000, input_dim=input_dim, W_regularizer=l1(0.0002), init='he_normal'),
        Activation('tanh'),
        # PReLU(init='zero', weights=None),
        Dropout(0.2),
        Dense(400, init='he_normal', W_regularizer=l2(0.0002)),
        PReLU(init='zero', weights=None),
        Dropout(0.2),
        Dense(200, init='he_normal', W_regularizer=l2(0.0002)),
        PReLU(init='zero', weights=None),
        Dropout(0.2),
        Dense(100, init='he_normal', W_regularizer=l2(0.0002)),
        PReLU(init='zero', weights=None),
        Dense(1),
        Activation('softplus'),
        # Activation('relu')
    ])
    optimizer = Nadam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.00005)
    model.compile(optimizer=optimizer, loss='mae')
    return model


def train_modified_reference_model(subtrain_x, subtrain_y, validation_x, validation_y):
    input_dim = subtrain_x.shape[1]
    print(input_dim)
    model = modified_reference_model(input_dim)
    model.fit(
        subtrain_x,
        subtrain_y,
        nb_epoch=NB_EPOCH_MODIFIED_REFERENCE_MODEL,
        batch_size=32,
        validation_data=(validation_x, validation_y)
    )
    loss = model.evaluate(validation_x, validation_y, verbose=0)
    return model, loss


def train_reference_model(subtrain_x, subtrain_y, validation_x, validation_y):
    input_dim = subtrain_x.shape[1]
    print(input_dim)
    model = reference_model(input_dim)
    model.fit(
        subtrain_x,
        subtrain_y,
        nb_epoch=NB_EPOCH_REFERENCE_MODEL,
        batch_size=32,
        validation_data=(validation_x, validation_y)
    )
    loss = model.evaluate(validation_x, validation_y, verbose=0)
    return model, loss


def train_my_model(subtrain_x, subtrain_y, validation_x, validation_y):
    input_dim = subtrain_x.shape[1]
    print(input_dim)
    model = my_model(input_dim)
    model.fit(
        subtrain_x,
        subtrain_y,
        nb_epoch=NB_EPOCH_MY_MODEL,
        batch_size=32,
        validation_data=(validation_x, validation_y)
    )
    loss = model.evaluate(validation_x, validation_y, verbose=0)
    return model, loss


def ensemble_models_predict(models, test_x):
    model_num = len(models)
    prediction = None
    for model in models:
        current_prediction = model.predict(test_x)
        if prediction is None:
            prediction = current_prediction
        else:
            prediction += current_prediction
    prediction /= 10
    return prediction.flatten()


def main():
    data_fname = sys.argv[1]
    sub_fname = sys.argv[2]

    print('Read data...')
    data = joblib.load(data_fname)
    df_train = data['train']
    df_test = data['test']

    train_x = df_train.drop('loss', axis=1).values
    train_y = df_train['loss'].values

    print('===== train model')
    models = []
    reference_model_loss = []
    modified_reference_model_loss = []
    my_model_loss = []
    kf = KFold(n_splits=5, shuffle=False, random_state=None)
    for subtr_idx, valid_idx in kf.split(train_x):
        subtrain_x = train_x[subtr_idx, :]
        subtrain_y = train_y[subtr_idx]
        validation_x = train_x[valid_idx, :]
        validation_y = train_y[valid_idx]
        model, mae = train_reference_model(subtrain_x, subtrain_y, validation_x, validation_y)
        models.append(model)
        reference_model_loss.append(mae)
        model, mae = train_modified_reference_model(subtrain_x, subtrain_y, validation_x, validation_y)
        models.append(model)
        modified_reference_model_loss.append(mae)
        model, mae = train_my_model(subtrain_x, subtrain_y, validation_x, validation_y)
        models.append(model)
        my_model_loss.append(mae)
        print('my_model_loss', my_model_loss)
        print('reference_model_loss', reference_model_loss)
        print('modified_reference_model_loss', modified_reference_model_loss)
    print('average my_model_loss', np.mean(my_model_loss))
    print('average reference_model_loss', np.mean(reference_model_loss))
    print('average modified_reference_model_loss', np.mean(modified_reference_model_loss))

    test_x = df_test.drop('id', axis=1).values
    test_id = df_test['id']
    pred = ensemble_models_predict(models, test_x)
    pd.DataFrame({'id': test_id, 'loss': pred}).to_csv(sub_fname, index=False)


if __name__ == '__main__':
    main()
