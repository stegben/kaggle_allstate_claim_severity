import sys
import itertools

import pandas as pd
import joblib

def extract_data(train_fname, test_fname, rand_state=1234):
    """
    Return:
    =======
    df_train: (features, label)
    df_valid: (features, label)
    df_test: (features, id)
    """
    print('read data...')
    df_train = pd.read_csv(train_fname)
    df_test = pd.read_csv(test_fname)

    df_train['is_test'] = False
    df_test['is_test'] = True

    df = pd.concat((df_train, df_test), axis=0)

    columns = df.columns
    print('===== start process data')
    # basic transformation
    for col in columns:
        print('process data column {}'.format(col))
        if 'cat' in col:
            dummy = pd.get_dummies(df[col], drop_first=True, prefix=col)
            df = pd.concat([df, dummy], axis=1).drop(col, axis=1)
        if col == 'cont2':
            print('{} has {} unique values'.format(col, len(df[col].value_counts())))
            dummy = pd.get_dummies(df[col], drop_first=True, prefix=col)
            df = pd.concat([df, dummy], axis=1)
        if 'cont' in col:
            target = df[col]
            df[col] = (target - target.mean()) / target.std()

    cont_col_combinations = itertools.combinations([col for col in columns if 'cont' in col], 2)
    for col_combination in cont_col_combinations:
        col1 = col_combination[0]
        col2 = col_combination[1]
        new_feat_1 = (df[col2] / df[col1])
        new_feat_1 = new_feat_1.fillna(new_feat_1.mean())
        new_feat_name_1 = col2 + '_divide_' + col1
        df[new_feat_name_1] = new_feat_1

        new_feat_2 = (df[col2] * df[col1])
        new_feat_name_2 = col2 + '_multiply_' + col1
        df[new_feat_name_2] = new_feat_2


    df_train = df[~df['is_test']].drop('id', axis=1)
    df_test = df[df['is_test']].drop('loss', axis=1)
    return df_train, df_test


def main():
    train_fname = sys.argv[1]
    test_fname = sys.argv[2]
    data_fname = sys.argv[3]

    # df_train, df_valid, df_test = extract_data(train_fname, test_fname, 1234)
    df_train, df_test = extract_data(train_fname, test_fname, 1234)
    joblib.dump({'train': df_train, 'test': df_test}, data_fname)



if __name__ == '__main__':
    main()
