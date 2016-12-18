import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./titan_allstate_claim_severity/raw_data/train.csv')

columns = df.columns

# cont_col_combinations = itertools.combinations([col for col in columns if 'cont' in col], 2)
# for col_combination in cont_col_combinations:
#     col1 = col_combination[0]
#     col2 = col_combination[1]
#     new_feat_1 = (df[col2] / df[col1])
#     new_feat_1 = new_feat_1.fillna(new_feat_1.mean())
#     new_feat_name_1 = col2 + '_divide_' + col1
#     df[new_feat_name_1] = new_feat_1

all_columns = df.columns

for col in all_columns:
    if 'cont' not in col:
        continue
    # new_col = col + '_cut'
    # df[new_col] = pd.Series(pd.cut(df[col].values, 10000))
    # loss_count = df.groupby(new_col)['loss'].count()
    # loss_mae = df.groupby(new_col)['loss'].agg(lambda x: (x - x.mean()).abs().mean())
    # plt.figure()
    # plt.scatter(loss_mae, loss_count)
    # plt.xlim([0, 5000])
    # plt.ylim([0,20000])
    # plt.savefig(col + '.png')
    plt.figure()
    plt.scatter(df[col].values, df['loss'].values)
    plt.savefig(col + '_loss.png')

