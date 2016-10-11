import sys

import joblib

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
# from keras.optimizers import Nadam

def train_model(train_x, train_y):
    input_dim = train_x.shape[1]

    model = Sequential([
        Dense(2048, input_dim=input_dim, init='he_normal'),
        Activation('relu'),
        Dense(1),
        Activation('softplus'),
    ])

    model.compile(optimizer='rmsprop', loss='mae')
    model.fit(train_x, train_y, nb_epoch=10, batch_size=32, validation_split=0.2)
    return model


def main():
    data_fname = sys.argv[1]

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
    pred = model.predict_proba()

if __name__ == '__main__':
    main()
