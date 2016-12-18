import sys
import itertools

import pandas as pd
import joblib


def split_train(df_train, train_ratio=0.9999, random_state=12334):
    df_subtrain = df_train.sample(frac=train_ratio, random_state=random_state)
    df_validation = df_train.drop(df_subtrain.index)
    return df_subtrain, df_validation


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
    """
    select_cat = ["cat80","cat87","cat57","cat12","cat79","cat10","cat7","cat89","cat2","cat72",
             "cat81","cat11","cat1","cat13","cat9","cat3","cat16","cat90","cat23","cat36",
             "cat73","cat103","cat40","cat28","cat111","cat6","cat76","cat50","cat5",
             "cat4","cat14","cat38","cat24","cat82","cat25"]
    # new_columns = []
    cont_col_combinations = itertools.combinations([col for col in select_cat], 2)
    for col_combination in cont_col_combinations:
        col1 = col_combination[0]
        col2 = col_combination[1]
        new_feat_1 = (df[col2] + df[col1])
        new_feat_1 = new_feat_1.fillna(-1)
        new_feat_name_1 = col2 + '__' + col1
        df[new_feat_name_1] = new_feat_1
    # for col in columns:
    #     if 'cat' in col:
    #         if col not in select_cat:
    #             continue
    #     new_columns.append(col)
    # df = df[new_columns]
    columns = df.columns
    """
    print('===== start process data')
    # basic transformation
    for col in columns:
        print('process data column {}'.format(col))
        if 'cat' in col:
            # if len(df[col].unique()) < 30:
            #     print('onehot {}'.format(col))
            #     dummy = pd.get_dummies(df[col], drop_first=False, prefix=col)
            #     df = pd.concat([df, dummy], axis=1)
            # if len(df[col].unique()) < 3:
            #     df = df.drop(col, axis=1)
            #     continue
            # df[col] = pd.factorize(df[col].values, sort=True)[0]
            # target = df[col]
            # df[col] = (target - target.mean()) / target.std()
            dummy = pd.get_dummies(df[col], drop_first=False, prefix=col)
            df = pd.concat([df.drop(col, axis=1), dummy], axis=1)
        # if col in ['cont2', 'cont7', 'cont14']:
        # if col in ['cont2', 'cont3', 'cont4','cont5', 'cont6','cont8','cont9', 'cont14']:
        # if col in ['cont2']:
        if col in []:
            unique_value = len(df[col].value_counts())
            print('{} has {} unique values'.format(col, unique_value))
            if unique_value > 100:
                print('cut 100')
                proc_sr = pd.Series(pd.cut(df[col].values, 100), index=df.index)
            else:
                proc_sr = df[col]
            dummy = pd.get_dummies(proc_sr, drop_first=False, prefix=col)
            df = pd.concat([df, dummy], axis=1)
        if 'cont' in col:
            target = df[col]
            df[col] = (target - target.mean()) / target.std()

    """
    cont_col_combinations = itertools.combinations([col for col in columns if 'cont' in col], 2)
    for col_combination in cont_col_combinations:
        col1 = col_combination[0]
        col2 = col_combination[1]
        new_feat_1 = (df[col2] / df[col1])
        new_feat_1 = new_feat_1.fillna(-1)
        new_feat_name_1 = col2 + '_divide_' + col1
        df[new_feat_name_1] = new_feat_1

        new_feat_2 = (df[col2] * df[col1])
        new_feat_name_2 = col2 + '_multiply_' + col1
        df[new_feat_name_2] = new_feat_2
    """
    import ipdb; ipdb.set_trace()
    df_train = df[~df['is_test']].drop('id', axis=1)
    df_test = df[df['is_test']].drop('loss', axis=1)
    return df_train, df_test


def main():
    train_fname = sys.argv[1]
    test_fname = sys.argv[2]
    data_fname = sys.argv[3]

    # df_train, df_valid, df_test = extract_data(train_fname, test_fname, 1234)
    df_train, df_test = extract_data(train_fname, test_fname, 1234)
    df_train, df_validation = split_train(df_train, train_ratio=0.95, random_state=678)
    joblib.dump({'train': df_train, 'test': df_test, 'validation': df_validation}, data_fname)


if __name__ == '__main__':
    main()
