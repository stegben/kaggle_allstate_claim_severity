import os
import uuid
from pprint import pprint
from time import time
from datetime import datetime
import pickle as pkl

import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

import xgboost as xgb


def load_model(fname):
    model = None
    with open(fname, 'rb') as f:
        model = pkl.load(f)
    return model


def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    return 'mae', mean_absolute_error(np.exp(preds), np.exp(labels))


def logregobj(preds, dtrain):
    labels = dtrain.get_label()
    con = 2
    x =preds-labels
    grad = con*x / (np.abs(x)+con)
    hess = con**2 / (np.abs(x)+con)**2
    return grad, hess


class BagXGB(object):
    def __init__(self, base_model_folder):
        self.base_model_folder = base_model_folder
        self.base_model_fnames = []
        self.errors = []

    def fit(self, train_x, train_y, n_folds=5, random_state=1234):
        self.base_model_fnames = []
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        now = datetime.now().strftime("%Y%m%d%H")
        self.errors = []
        for idx, (subtr_idx, valid_idx) in enumerate(kf.split(train_x)):
            subtrain_x = train_x[subtr_idx, :]
            subtrain_y = train_y[subtr_idx]
            validation_x = train_x[valid_idx, :]
            validation_y = train_y[valid_idx]

            best_model_fname = self.base_model_folder + str(uuid.uuid4()) + 'xgb_model_' + now + '_fold_' + str(idx) + '.h5'
            model, mae = self._train_model(subtrain_x, subtrain_y, validation_x, validation_y, best_model_fname)
            self.errors.append(mae)
        print('mae of each fold:')
        pprint(self.errors)

    def _train_model(self, subtrain_x, subtrain_y, validation_x, validation_y, best_model_fname):
        params = {
            'min_child_weight': 1,
            'learning_rate': 0.03,
            'colsample_bytree': 0.7,
            'subsample': 0.8,
            'gamma': 1,
            'silent': 0,
            'seed': 1234,
            # 'booster': 'gblinear',
            # 'booster': 'gbtree',
            'max_depth': 11,
            'objective': 'reg:linear',
            'nthread': 12,
            # 'n_estimators': 2000,
        }

        # rgs = xgb.train(params, xgsubtrain, early_stopping_rounds=10, eval=(xgval, 'eval'))
        subtrain_y = np.log(subtrain_y + 200)
        raw_validation_y = validation_y
        validation_y = np.log(validation_y + 200)
        xgsubtrain = xgb.DMatrix(subtrain_x, label=subtrain_y, )
        xgval = xgb.DMatrix(validation_x, label=validation_y)
        watch_list = [(xgsubtrain, 'subtrain'), (xgval, 'val')]
        rgs = xgb.train(params, xgsubtrain, 2000, watch_list, obj=logregobj, feval=evalerror, early_stopping_rounds=50)

        # rgs = xgb.XGBModel(**params)
        # rgs.fit(
        #     subtrain_x, subtrain_y,
        #     eval_set=[(subtrain_x, subtrain_y), (validation_x, validation_y)],
        #     eval_metric=evalerror,
        #     early_stopping_rounds=50,
        #     verbose=True,
        # )
        self.base_model_fnames.append(os.path.abspath(best_model_fname))
        validation_pred = np.exp(rgs.predict(xgval)) - 200
        loss = mean_absolute_error(raw_validation_y, validation_pred)
        with open(best_model_fname, 'wb') as f:
            pkl.dump(rgs, f)
        return rgs, loss

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
        xgtest = xgb.DMatrix(test_x)
        for model in models:
            current_prediction = model.predict(xgtest)
            current_prediction = np.exp(current_prediction) - 200
            predictions.append(current_prediction.flatten())
        predictions = np.column_stack(predictions)
        return predictions
