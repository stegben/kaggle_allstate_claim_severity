import sys
from datetime import datetime

import numpy as np
import pandas as pd

import joblib

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

import xgboost as xgb


def train_xgb_model(subtrain_x, subtrain_y, validation_x, validation_y, best_model_fname):
    print(subtrain_x.shape)
    params = {
        'min_child_weight': 1,
        'learning_rate': 0.03,
        'colsample_bytree': 0.9,
        'subsample': 0.9,
        'gamma': 1,
        'silent': 0,
        'seed': 1234,
        # 'booster': 'gblinear',
        # 'booster': 'gbtree',
        'max_depth': 9,
        'objective': 'reg:linear',
        'nthread': 10,
        'n_estimators': 2000,
    }

    # xgsubtrain = xgb.DMatrix(subtrain_x, label=subtrain_y, )
    # xgval = xgb.DMatrix(validation_x, label=validation_y)
    # rgs = xgb.train(params, xgsubtrain, early_stopping_rounds=10, eval=(xgval, 'eval'))

    rgs = xgb.XGBModel(**params)
    rgs.fit(
        subtrain_x, subtrain_y,
        eval_set=[(subtrain_x, subtrain_y), (validation_x, validation_y)],
        eval_metric='mae',
        early_stopping_rounds=30,
        verbose=True,
    )
    return rgs, mean_absolute_error(validation_y, rgs.predict(validation_x))


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

        best_model_fname = '../xgb_model_' + now + '_fold_' + str(idx) + '.pkl'
        model, mae = train_xgb_model(subtrain_x, subtrain_y, validation_x, validation_y, best_model_fname)
        models.append(model)
        joblib.dump(model, best_model_fname)
        model_loss.append(mae)
        print('model_loss', model_loss)
    print('average model_loss', np.mean(model_loss))

    test_x = df_test.drop('id', axis=1).values
    test_id = df_test['id']
    pred = ensemble_models_predict(models, test_x)
    pd.DataFrame({'id': test_id, 'loss': pred}).to_csv(sub_fname, index=False)


if __name__ == '__main__':
    main()
