import pandas as pd
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.multioutput import MultiOutputClassifier

def predict(model: MultiOutputClassifier,
            X_train: pd.DataFrame,
            X_test: pd.DataFrame,
            y_train: pd.DataFrame,
            y_test: pd.DataFrame) -> (list, list):
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    cf_train = multilabel_confusion_matrix(y_pred_train, y_train)
    TN, FP, FN, TP, prec = [], [], [], [], []
    for matr in cf_train:
        tn, fp, fn, tp = matr.ravel()
        TN.append(tn)
        FP.append(fp)
        FN.append(fn)
        TP.append(tp)
        prec.append(sum(TP) / (sum(TP) + sum(FP)))

    TN, FP, FN, TP, rec = [], [], [], [], []
    cf_test = multilabel_confusion_matrix(y_pred_test, y_test)
    for matr in cf_test:
        tn, fp, fn, tp = matr.ravel()
        TN.append(tn)
        FP.append(fp)
        FN.append(fn)
        TP.append(tp)
        rec.append(sum(TP) / (sum(TP) + sum(FN)))
    return prec, rec
