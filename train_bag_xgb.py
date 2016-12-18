import os
import sys
from datetime import datetime
import pickle as pkl

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

import joblib

from bag_xgb import BagXGB


def main():
    data_fname = sys.argv[1]
    sub_fname = sys.argv[2]

    print('Read data...')
    data = joblib.load(data_fname)
    df_train = data['train']
    df_validation = data['validation']
    df_test = data['test']

    train_x = df_train.drop('loss', axis=1).values
    validation_x = df_validation.drop('loss', axis=1).values
    train_y = df_train['loss'].values
    validation_y = df_validation['loss'].values

    print('===== train model')
    model = BagXGB(base_model_folder='../allstate_basemodel/.base_models/')
    model.fit(train_x, train_y, n_folds=10, random_state=1234)

    validation_pred = model.predict(validation_x)
    mae = mean_absolute_error(validation_y, validation_pred)
    print(mae)
    model_fname = '../allstate_basemodel/%4.0f_xgb_with_' % mae
    model_fname = model_fname + os.path.split(data_fname)[1].split('.')[0] + '.pkl'
    with open(model_fname, 'wb') as f:
        pkl.dump(model, f)

    test_x = df_test.drop('id', axis=1).values
    test_id = df_test['id']
    pred = model.predict(test_x)
    pd.DataFrame({'id': test_id, 'loss': pred}).to_csv(sub_fname, index=False)


if __name__ == '__main__':
    main()
