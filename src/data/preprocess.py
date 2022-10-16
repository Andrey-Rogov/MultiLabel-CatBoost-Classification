import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def fill_sex(data: pd.DataFrame) -> pd.DataFrame:
    param_sex = {np.NaN: 0, 'М': 0, 'Ж': 1}
    new_data = np.array([param_sex[val] for val in data['Пол']])
    data['Пол'] = new_data
    return data


def smoking_and_alcohol(data: pd.DataFrame) -> pd.DataFrame:
    param_smoke = {'Курит': 0,
                   'Бросил(а)': 1,
                   'Никогда не курил(а)': 2,
                   'Никогда не курил': 2}
    new_data = np.array([param_smoke[val] for val in data['Статус Курения']])
    data['Статус Курения'] = new_data

    param_alco = {'употребляю в настоящее время': 0,
                  'ранее употреблял': 1,
                  'никогда не употреблял': 2}
    new_data = np.array([param_alco[val] for val in data['Алкоголь']])
    data['Алкоголь'] = new_data

    data['Возраст курения'].fillna(0, inplace=True)
    data['Сигарет в день'].fillna(0, inplace=True)
    data['Возраст алког'].fillna(0, inplace=True)
    return data


def drop_unnecessary_cols(data: pd.DataFrame) -> pd.DataFrame:
    if 'ID_y' in data.columns:
        data.drop(['ID', 'ID_y'], axis=1, inplace=True)
    else:
        data.drop(['ID'], axis=1, inplace=True)
    return data


def education_label(data: pd.DataFrame) -> pd.DataFrame:
    data['Образование'] = data['Образование'].str[:2].astype(np.int64)
    return data


def fill_na(data: pd.DataFrame) -> pd.DataFrame:
    #     not_full = []  # To get know which columns should be processed
    #     for col in data.columns:
    #         if data[col].dropna().shape[0] != 955:  # 638 if running for test set
    #             not_full.append(col)
    #     print(not_full)
    data = fill_sex(data)
    data = smoking_and_alcohol(data)
    return data


def labels_encoding(data: pd.DataFrame, cols: list) -> pd.DataFrame:
    for col in cols:
        enc = LabelEncoder()
        enc.fit(data[col].unique())
        a = enc.transform(data[col])
        data[col] = a
    return data


class PreprocessTransformer:
    def __init__(self):
        self.train_data = None
        self.test_data = None

    def fit(self, x_train, x_test):
        self.train_data = x_train
        self.test_data = x_test
        return self

    def transform(self):
        self.train_data = fill_na(self.train_data)
        self.test_data = fill_na(self.test_data)

        self.train_data = drop_unnecessary_cols(self.train_data)
        self.test_data = drop_unnecessary_cols(self.test_data)

        self.train_data = education_label(self.train_data)
        self.test_data = education_label(self.test_data)

        self.train_data = labels_encoding(self.train_data, ['Семья', 'Этнос', 'Национальность', 'Религия', 'Профессия',
                                                            'Время засыпания', 'Время пробуждения', 'Частота пасс кур'])
        self.test_data = labels_encoding(self.test_data, ['Семья', 'Этнос', 'Национальность', 'Религия', 'Профессия',
                                                          'Время засыпания', 'Время пробуждения', 'Частота пасс кур'])
        return self.train_data, self.test_data
