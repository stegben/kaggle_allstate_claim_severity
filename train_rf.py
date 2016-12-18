import sys
from datetime import datetime

import numpy as np
import pandas as pd

import joblib

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


def ensemble_models_predict(models, test_x):
    model_num = len(models)
    prediction = None
    for model in models:
        current_prediction = model.predict(test_x)
        if prediction is None:
            prediction = current_prediction
        else:
            prediction += current_prediction
    prediction /= model_num
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
    model_loss = []
    kf = KFold(n_splits=5, shuffle=True, random_state=1234)
    now = datetime.now().strftime("%Y%m%d%H")
    for idx, (subtr_idx, valid_idx) in enumerate(kf.split(train_x)):
        subtrain_x = train_x[subtr_idx, :]
        subtrain_y = train_y[subtr_idx]
        validation_x = train_x[valid_idx, :]
        validation_y = train_y[valid_idx]

        rgs = RandomForestRegressor(n_estimators=100, criterion='mae', max_depth=8, n_jobs=28, verbose=2, max_features=0.01)
        rgs.fit(subtrain_x, subtrain_y)

        pred = rgs.predict(validation_x)
        loss = mean_absolute_error(validation_y, pred)

        best_model_fname = '../rf_model_' + now + '_fold_' + str(idx) + '.pkl'
        joblib.dump(rgs, best_model_fname)
        models.append(rgs)
        model_loss.append(loss)

        print('model_loss', model_loss)
    print('average model_loss', np.mean(model_loss))

    test_x = df_test.drop('id', axis=1).values
    test_id = df_test['id']
    pred = ensemble_models_predict(models, test_x)
    pd.DataFrame({'id': test_id, 'loss': pred}).to_csv(sub_fname, index=False)


if __name__ == '__main__':
    main()
