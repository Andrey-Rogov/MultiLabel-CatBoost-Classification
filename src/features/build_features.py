import pandas as pd
import numpy as np
import datetime


def passive_smoking(data: pd.DataFrame) -> pd.DataFrame:
    param = {np.NaN: 0,
             '1-2 раза в неделю': 2,
             '3-6 раз в неделю': 4,
             'не менее 1 раза в день': 10,
             '4 и более раз в день': 35,
             '2-3 раза в день': 25}
    new_data = np.array([param[val] for val in data['Частота пасс кур']])
    data['Пасс кур в неделю'] = new_data
    for idx in range(data.shape[0]):
        if data.iloc[idx]['Пассивное курение'] == 0 and data.iloc[idx]['Пасс кур в неделю'] != 0:
            data.loc[idx, 'Пасс кур в неделю'] = 0
    return data


def time_diff(sleep: datetime.time(), wakeup: datetime.time()) -> int:
    diff = datetime.timedelta(hours=wakeup.hour, minutes=wakeup.minute) - datetime.timedelta(hours=sleep.hour,
                                                                                             minutes=sleep.minute)
    if str(diff).startswith('-1 day'):
        return int(str(diff)[7:9].split(':')[0])
    else:
        return int(str(diff).split(':')[0])


def sleep_wakeup(data: pd.DataFrame) -> pd.DataFrame:
    sleep = []
    wakeup = []
    for col1, col2 in zip(data['Время засыпания'].values, data['Время пробуждения'].values):
        hour, minute, sec = map(np.int32, col1.split(':'))
        val = datetime.time(hour, minute, sec)
        sleep.append(val)

        hour, minute, sec = map(np.int32, col2.split(':'))
        val = datetime.time(hour, minute, sec)
        wakeup.append(val)
    dream_quality = []
    for sl, wk in zip(sleep, wakeup):
        test = time_diff(sl, wk)
        if test in (6, 7, 8):
            dream_quality.append(0)
        elif test > 8:
            dream_quality.append(1)
        else:
            dream_quality.append(2)
    data['Качество сна'] = dream_quality
    return data


class FeaturesTransformer:
    def __init__(self):
        self.train_data = None
        self.test_data = None

    def fit(self, x_train, x_test):
        self.train_data = x_train
        self.test_data = x_test
        return self

    def transform(self):
        self.train_data = sleep_wakeup(self.train_data)
        self.test_data = sleep_wakeup(self.test_data)

        self.train_data = passive_smoking(self.train_data)
        self.test_data = passive_smoking(self.test_data)
        return self.train_data, self.test_data
