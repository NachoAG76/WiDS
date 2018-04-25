import pandas as pd
from copy import deepcopy
import csv
from sklearn import metrics
from sklearn.metrics import confusion_matrix


train_selfea = pd.read_csv("train.csv",low_memory=False)

def classify(clf,x_n_train, y_n_train, x_n_test,y_n_test,test, useall = False):
    if useall:
        clf.fit(x_n_train, y_n_train)
        # test_n = test.reindex(columns = x_n_train.columns)
        test_n = test
    else:
        clf.fit(x_n_train, y_n_train)

        y_pred = clf.predict(x_n_test)

        cm = confusion_matrix(y_n_test, y_pred)
        print (cm)

        y_prob = clf.predict_proba(x_n_test)
        print(metrics.roc_auc_score(y_n_test, y_prob[:,1]))

        test_n = test.reindex(columns = x_n_test.columns)

    y_pred_f1 = clf.predict(test_n)
    y_prob = clf.predict_proba(test_n)
    return y_prob


class InitData():
    def __init__(self):
        print("start reading data")
        # train_selfea = pd.read_csv("train.csv",low_memory=False)
        print("read train")
        test = pd.read_csv("test.csv",low_memory=False)
        print("read test")
        # new_feature = pd.read_csv("new_feature.csv")
        # new_feature = new_feature["Column Name"].tolist()

        # lst = ['DL14','DG5_4','AA15','AA14','DG8a','AA7','DG4','GN2','DG1','MT10','GN3','GN5','MT2','GN4','DG3','FL4','DL1','DG6','DL0']
        # lst = lst + new_feature

        lst = ['DL14','DG5_4','AA15','AA14','DG8a','AA7','DG4','GN2','DG1','MT10','GN3','GN5','MT2','GN4','DG3','FL4','DL1','DG6','DL0']
        add_feature = ['DG5_9','DL0', 'DL1', 'DL2_new', 'G2P1_11_new', 'DL4_22', "MT1A_m", "MT1A_f"]
        add_feature2 = ["GN2_new","FF14_6_new","FF14_5_new","FF14_4_new","FF14_3_new","MT18_5_new","MT18_4_new","GN5"]
        lst = lst + add_feature + add_feature2

        self.train = train_selfea
        self.test = test
        self.features = lst

        process_data(train_selfea)
        process_data(test)
        X_new = train_selfea[lst]
        Y_new = train_selfea.is_female

        self.processed_Xtr = X_new
        self.processed_Ytr = Y_new
        self.processed_test = test
        print("finished")


    def get_train(self, pro = False):
        """
        if pro is default(False), return the original train data, the same as the
        original csv file

        if pro is true, return the processed train with only selected features
        """
        if pro:
            return deepcopy(self.processed_Xtr)
        return deepcopy(self.train)

    def get_labels(self):
        """Return the train.is_female"""
        return deepcopy(train_selfea.is_female)

    def get_test(self, pro = False):
        """
        if pro is default(False), return the original test data, the same as the
        original csv file

        if pro is true, return the processed test with only selected features
        """
        if pro:
            return deepcopy(self.processed_test)
        return deepcopy(self.test)

    def get_features(self):
        """Return a list of selected features, both human selected and random forest generated"""
        return deepcopy(self.features)

def process_DL2(df,pr = False):
    f = "DL2"
    male = []
    female = []
    middle = []
    for choice in range(1,33):
        perc = rate_feature(f,choice)
        if perc < 0.3:
            male.append(choice)
    if pr:
        print(male)
    feature_DL2 = df.loc[df.DL2.isin(male), "DL2_new"]=1
    df["DL2_new"] = feature_DL2
    if pr:
        print((df[["DL2_new","DL2"]]).head())


def process_MT17_6(df, pr = False):
    f = 'MT17_6'
    choices = range(1,7)
    male = []
    female = []
    for choice in choices:
        perc = rate_feature(f,choice)
        if perc < 0.3:
            male.append(choice)
        if pr:
            print(perc)

    df.loc[df.MT1A.isin(male), f +"_m"]=1
#     df.loc[df.MT1A.isin(female), "MT1A_f"]=1
    df.MT17_6_m = df.MT17_6_m.fillna(0)
#     df.MT1A_f = df.MT1A_f.fillna(0)
    if pr:
        print("male",male)
        print((df[[f,f +"_m"]]).head())

def process_MT1A(df,pr = False):
    f = 'MT1A'
    choices = [1,2,3,4,5,8,99]
    male = []
    female = []
    for choice in choices:
        perc = rate_feature(f,choice)
        if perc < 0.3:
            male.append(choice)
        elif perc < 0.7:
            pass
        else:
            female.append(choice)
    if pr:
        print("male",male)
        print("female",female)
    df.loc[df.MT1A.isin(male), "MT1A_m"]=1
    df.loc[df.MT1A.isin(female), "MT1A_f"]=1
    df.MT1A_m = df.MT1A_m.fillna(0)
    df.MT1A_f = df.MT1A_f.fillna(0)
    if pr:
        print((df[["MT1A","MT1A_m","MT1A_f"]]).head())


def process_cols(df,pr=False):
    cols = [("FF14_6",1),("FF14_5",1),("FF14_4",1),("FF14_3",1),("MT18_5",1),("MT18_4",1)]
    for pair in cols:
        col, choice = pair
        col_new = col+"_new"
        df.loc[df[col] == choice, col_new]=choice
        df.loc[df[col] != choice, col_new]=choice+1
#         df.[col_new] = df.col_new.fillna(0)
        if pr:
            print((df[[col,col_new]]).head())


def process_GN2(df,pr = False):
    df.loc[df.GN2 == 2, "GN2_new"]=1
    df.loc[df.GN2 != 2, "GN2_new"]=0
    df.GN2_new = df.GN2_new.fillna(0)
    if pr:
        print((df[["GN2","GN2_new"]]))


def process_G2P1_11(df,pr = False):
    df.loc[df.G2P1_11 == 1, "G2P1_11_new"]=1
    df.G2P1_11_new = df.G2P1_11_new.fillna(2)
    if pr:
        print((df[["G2P1_11","G2P1_11_new"]]).head())


def rate_feature(f,choice):
    # among people who chose choice in f, what percent is female
    df=(train_selfea[["is_female",f]])
    # df[ :,lambda dh
    feature = df.loc[df[f] ==choice]
    a = list(feature["is_female"])
    if len(a) == 0:
        return 0
    return a.count(1) * 1.0 / len(a)

def process_data(df):
    process_MT1A(df)
    process_DL2(df)
    process_G2P1_11(df)
    process_GN2(df)
    process_cols(df)

def write_file(fname,y_prob):
    with open(fname,'w') as f:
        fieldnames = ["test_id","is_female"]
        wri = csv.DictWriter(f, delimiter=',',fieldnames= fieldnames)
        acc = 0
        wri.writeheader()
        for i in y_prob:
            wri.writerow({"test_id":acc, "is_female": i[1]})
            acc = acc+1
