import numpy as np
import pandas as pd

def delete80na_col(train):
    over80 = []
    other = []
    for col in list(train.columns.values):
        per = train[col].isnull().sum()/len(train[col])
        if per > 0.4:
#             print (">80")
            over80.append(col)
        else:
            other.append(col)
    return over80, other

def my_fillna(X):
#     acc = 0
    for col in list(X.columns.values):
        X[col].fillna(100, inplace = True)
    return X

def preprocess(train, test):
    df_all = pd.concat([train,test])
    category_cols = train.select_dtypes(exclude=[np.number]).columns.tolist() + test.select_dtypes(exclude=[np.number]).columns.tolist()
    for header in category_cols:
#     df[header] = df[header].astype('category').cat.codes
#     test[header] = test[header].astype('category').cat.codes
        df_all[header] = df_all[header].astype('category').cat.codes.astype('int')
        df_all[header] = pd.to_numeric(df_all[header])

        train[header] = train[header].astype('category').cat.codes
        train[header] = pd.to_numeric(train[header])
        test[header] = test[header].astype('category').cat.codes
        test[header] = pd.to_numeric(test[header])


    # over80, other = delete80na_col(train)
    # train_delete_over80 = train.drop(over80, axis = 1)
    # test = test.reindex(columns = train_delete_over80.columns, fill_value = 0)


    my_fillna(train)
    my_fillna(test)
    test = test.reindex(columns = train.columns, fill_value = 0)

    return train, test
