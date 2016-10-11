import sys

import pandas as pd


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
            dummy = pd.get_dummies(df[col], dummy_na=True, drop_first=True, prefix=col)
            df = pd.concat([df, dummy], axis=1).drop(col, axis=1)
        if col == 'cont2':
            print('cont2 has {} unique values'.format(len(df[col].value_counts())))
            dummy = pd.get_dummies(df[col], dummy_na=True, drop_first=True, prefix=col)
            df = pd.concat([df, dummy], axis=1)
        if 'cont' in col:
            target = df[col]
            df[col] = (target - target.mean()) / target.std()

    df_train = df[~df['is_test']].drop('id')
    df_test = df[df['is_test']]


def main():
    train_fname = sys.argv[1]
    test_fname = sys.argv[2]
    data_fname = sys.argv[3]

    # df_train, df_valid, df_test = extract_data(train_fname, test_fname, 1234)
    df_train, df_test = extract_data(train_fname, test_fname, 1234)

    joblib.dump(data_fname, {'train': df_train, 'test': df_test})
    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    main()
