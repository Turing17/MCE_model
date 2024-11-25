from sklearn.ensemble import RandomForestClassifier
import os
import pandas as pd
from sklearn.metrics import accuracy_score

path_home = os.path.dirname(__file__)
path_boot = os.path.dirname(path_home)
path_radiomics_f_csv = os.path.join(path_home, 'rf_csv/csv_test')


def train_p_r(list_id_train, list_id_val, p, roi):
    # 初始化log
    df_log = pd.DataFrame(columns=['ID', 'label', 'prob_label_0', 'prob_label_1', 'result'])

    path_temp = os.path.join(path_radiomics_f_csv, f'{p}_{roi}.csv')
    data = pd.read_csv(path_temp)
    data_train = data[data['ID'].isin(list_id_train)]
    data_val = data[data['ID'].isin(list_id_val)]
    X_train = data_train[data_train.columns[2:]]
    y_train = data_train['label']
    X_val = data_val[data_val.columns[2:]]
    y_val = data_val['label']
    best_acc = 0
    best_acc_r = 0
    for r in range(10):
        rf_train = RandomForestClassifier(n_estimators=500, random_state=r)
        rf_train.fit(X_train, y_train)
        y_pred = rf_train.predict(X_val)
        acc_temp = accuracy_score(y_val, y_pred)
        if acc_temp > best_acc:
            best_acc = acc_temp
            best_acc_r = r
    rf = RandomForestClassifier(n_estimators=500, random_state=best_acc_r)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_val)
    y_prob = rf.predict_proba(X_val)
    df_log['ID'] = list_id_val
    df_log['label'] = y_val.tolist()
    df_log['prob_label_0'] = y_prob[:, 0].tolist()
    df_log['prob_label_1'] = y_prob[:, 1].tolist()
    list_result = []
    for i, v in enumerate(y_pred.tolist()):
        if v == y_val.tolist()[i]:
            list_result.append('Right')
        else:
            list_result.append('Wrong')
    df_log['result'] = list_result
    return df_log


def get_radiomics_score(test_seed, dataset):
    path_trainval_txt = os.path.join(path_boot, f'bin/nametxt/brats{dataset}')
    path_log = os.path.join(path_home, 'log', f'{test_seed}')
    if not os.path.exists(path_log):
        os.mkdir(path_log)
    with open(os.path.join(path_trainval_txt, "train_name.txt"), "r") as f:
        list_id_train = [i.split()[0] for i in f.readlines() if i.split()[0] != '\n']
    with open(os.path.join(path_trainval_txt, "val_name.txt"), "r") as f:
        list_id_val = [i.split()[0] for i in f.readlines() if i.split()[0] != '\n']
    for p in ('flair', 't1ce', 't2'):
        for roi in ('14', '142'):
            df_log = train_p_r(list_id_train, list_id_val, p, roi)
            df_log.to_csv(os.path.join(path_log, f'{dataset}_{p}_{roi}.csv'),index=False)
    return df_log


