import sys

import pandas as pd

import joblib

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers.advanced_activations import PReLU
from keras.layers.core import Dropout
from keras.layers.noise import GaussianNoise
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1, l2
from keras.optimizers import Nadam


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


def train_model(train_x, train_y):
    input_dim = train_x.shape[1]
    print(input_dim)
    model = modified_reference_model(input_dim)
    # model = reference_model(input_dim)
    # model = my_model(input_dim)

    model.fit(train_x, train_y, nb_epoch=50, batch_size=128, validation_split=0.1)
    return model


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
    model = train_model(train_x, train_y)

    test_x = df_test.drop('id', axis=1).values
    test_id = df_test['id']
    pred = model.predict(test_x)
    pd.DataFrame({'id': test_id, 'loss': pred.flatten()}).to_csv(sub_fname, index=False)


if __name__ == '__main__':
    main()
