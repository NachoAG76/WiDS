import pandas as pd


new_feature = pd.read_csv("new_feature.csv")
new_feature = new_feature["Column Name"].tolist()

lst = ['DL14','DG5_4','AA15','AA14','DG8a','AA7','DG4','GN2','DG1','MT10','GN3','GN5','MT2','GN4','DG3','FL4','DL1','DG6','DL0']
lst = lst + new_feature

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
    df=(train[["is_female",f]])
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