import streamlit as st
from typing import List, Dict

import PySimpleGUI as sg #стремлид
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
# pip install pandas, PySimpleGUI, openpyxl, matplotlib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

import statsmodels.api as sm
# pip install numpy, sklearn, keras, statsmodels
from sklearn.linear_model import LinearRegression as LinearCorrModel

import warnings
warnings.simplefilter(action='ignore')

from database import Database
# from settings import DEFAULT_USER, DEBUG_MODE, DATA_SQL_TABLE, TITLE, Y_LIMIT, X_LIMIT
from settings import DEFAULT_USER, DEBUG_MODE, DATA_SQL_TABLE, TITLE, Y_LIMIT, X_LIMIT, AboutProgram1, AboutProgram2, AboutProgram3, AboutProgram4, AboutProgram5, AboutProgram6, AboutProgram7



import streamlit as st

def prepare_model(neurons: int, input_dim: int, activation='sigmoid'):
    """
    Вспомогательная фугкция подготовки модели НС
    """
    model = Sequential()
    model.add(Dense(neurons, input_dim=input_dim, activation='relu'))
    model.add(Dense(neurons, activation='relu'))
    model.add(Dense(1, activation=activation))
    return model


def model_plot(Y_COLS: List, history_dict: Dict, values: Dict, verbous=True):
    """
    Вспомогательная фукгция графика метрик модели
    """
    plt.figure(figsize=(8, 8))
    i = 1
    for y_col in Y_COLS:
        plt.subplot(len(Y_COLS), 1, i)
        plt.title(f"Модель для {y_col}. Тип: {values[y_col]}.")
        plt.plot(history_dict[y_col].history['mae'],
                 label='Средняя абсолютная ошибка - train')
        plt.plot(history_dict[y_col].history['val_mae'],
                 label='Средняя абсолютная ошибка - test')
        plt.xlabel('Эпоха обучения')
        plt.ylabel('Ошибка')
        plt.legend()
        i += 1

    plt.tight_layout()
    if verbous:
        plt.show()
    else:
        plt.savefig('model_plot.png')


def data_plot(test_size, Y_COLS, model_dict, normalized_learn_df_x, max_y_dict, min_y_dict, learn_df, verbous=True):
    """
    Вспомогательная фукция графика данных
    """
    plt.figure(figsize=(15, 8))
    i = 1

    for y_col in Y_COLS:
        pred = model_dict[y_col].predict(normalized_learn_df_x).flatten()
        # Возвращем данные к нашей размерности - обратное нормирование
        pred = pred * max_y_dict[y_col] + min_y_dict[y_col]
        max_len = len(pred)
        limit = int(max_len * (1 - test_size / 100))

        tmp_df = pd.DataFrame(columns=['pred', 'y'])
        tmp_df['pred'] = pred
        tmp_df['pred'] = tmp_df['pred'].round(0)
        tmp_df['y'] = learn_df[y_col]

        # Очищаем от всплесков
        tmp_df = tmp_df.drop(np.where(abs(tmp_df['y'] - tmp_df['pred']) / tmp_df['y'] > 0.1)[0])
        plt.subplot(len(Y_COLS), 1, i)
        plt.plot(tmp_df['pred'], ':o', label='Прогноз данных НС') #FIXME 19/05

        plt.plot(tmp_df.loc[:limit, 'y'], 'ro', label='Данные обучения НС')
        plt.axvline(x=limit)
        plt.xlabel('index')
        plt.ylabel(y_col)
        plt.legend()
        i += 1
    plt.tight_layout()
    if verbous:
        plt.show()
    else:
        plt.savefig('data_plot.png')


def all_plot(test_size, Y_COLS, model_dict, normalized_learn_df_x, max_y_dict, min_y_dict, learn_df, history_dict, values, size=None):
    """
    Вспомогательная фукгция графика метрик модели
    """
    plt.figure(figsize=(18, 9))
    i = 1
    plt.suptitle('Обзорная панель', fontsize=18)

    if size: #FIXME 19/05
        max_len = size
        limit = int(max_len * (1 - test_size / 100))

    for y_col in Y_COLS:
        ########
        ax = plt.subplot(len(Y_COLS), 2, i)
        ax.set_title(f"Модель для {y_col}. Тип: {values[y_col]}.")
        ax.plot(history_dict[y_col].history['mae'],
                 label='Средняя абсолютная ошибка - train')
        ax.plot(history_dict[y_col].history['val_mae'],
                 label='Средняя абсолютная ошибка - test')
        ax.set_xlabel('Эпоха обучения')
        ax.set_ylabel('Ошибка')
        ax.legend()
        ##########
        pred = model_dict[y_col].predict(normalized_learn_df_x).flatten()
        # Возвращем данные к нашей размерности - обратное нормирование
        pred = pred * max_y_dict[y_col] + min_y_dict[y_col]
        if not size: #FIXME 19/05
            max_len = len(pred)
            limit = int(max_len * (1 - test_size / 100))

        tmp_df = pd.DataFrame(columns=['pred', 'y'])
        tmp_df['pred'] = pred
        tmp_df['pred'] = tmp_df['pred'].round(0)
        tmp_df['y'] = learn_df[y_col]

        # Очищаем от всплесков
        tmp_df = tmp_df.drop(np.where(abs(tmp_df['y'] - tmp_df['pred']) / tmp_df['y'] > 0.1)[0])
        plt.subplot(len(Y_COLS), 2, i + 1)
        plt.plot(tmp_df['pred'].loc[:max_len], ':o', label='Прогноз данных НС')

        plt.plot(tmp_df.loc[:limit, 'y'], 'ro', label='Данные обучения НС')
        plt.axvline(x=limit)
        plt.xlabel('index')
        plt.ylabel(y_col)
        plt.legend()

        i += 2

    plt.tight_layout()
    plt.show()


def regress_data_plot(test_size, Y_COLS, model_dict, normalized_learn_df_x, max_y_dict, min_y_dict, learn_df, verbous=True, size=None):
    plt.figure(figsize=(15, 8))
    i = 1
    plt.suptitle('Результаты регрессионного анализа', fontsize=18)

    if size: #FIXME 19/05
        max_len = size
        limit = int(max_len * (1 - test_size / 100))

    for y_col in Y_COLS:
        pred = model_dict[y_col].predict(normalized_learn_df_x).flatten()
        # Возвращем данные к нашей размерности - обратное нормирование
        pred = pred * max_y_dict[y_col] + min_y_dict[y_col]

        if not size: #FIXME 19/05
            max_len = len(pred)
            limit = int(max_len * (1 - test_size / 100))

        tmp_df = pd.DataFrame(columns=['pred', 'y'])
        tmp_df['pred'] = pred
        tmp_df['pred'] = tmp_df['pred'].round(0)
        tmp_df['y'] = learn_df[y_col]

        # Очищаем от всплесков
        tmp_df = tmp_df.drop(np.where(abs(tmp_df['y'] - tmp_df['pred']) / tmp_df['y'] > 0.1)[0])
        plt.subplot(len(Y_COLS), 1, i)
        plt.plot(tmp_df['pred'].loc[:max_len], ':o', label='Прогноз данных')

        plt.plot(tmp_df.loc[:limit, 'y'], 'ro', label='Данные для анализа')
        plt.axvline(x=limit)
        plt.xlabel('index')
        plt.ylabel(y_col)
        plt.legend()
        i += 1
    plt.tight_layout()
    if verbous:
        plt.show()
    else:
        plt.savefig('regress_data_plot.png')


def corr_data_plot(test_size, Y_COLS, model_dict, normalized_learn_df_x, max_y_dict, min_y_dict, learn_df, verbous=True, size=None):
    plt.figure(figsize=(15, 8))
    i = 1
    plt.suptitle('Результаты корреляционного анализа', fontsize=18)

    if size: #FIXME 19/05
        max_len = size
        limit = int(max_len * (1 - test_size / 100))

    for y_col in Y_COLS:
        pred = model_dict[y_col].predict(normalized_learn_df_x).flatten()
        # Возвращем данные к нашей размерности - обратное нормирование
        pred = pred * max_y_dict[y_col] + min_y_dict[y_col]

        if not size: #FIXME 19/05
            max_len = len(pred)
            limit = int(max_len * (1 - test_size / 100))

        tmp_df = pd.DataFrame(columns=['pred', 'y'])
        tmp_df['pred'] = pred
        tmp_df['pred'] = tmp_df['pred'].round(0)
        tmp_df['y'] = learn_df[y_col]

        # Очищаем от всплесков
        tmp_df = tmp_df.drop(np.where(abs(tmp_df['y'] - tmp_df['pred']) / tmp_df['y'] > 0.1)[0])
        plt.subplot(len(Y_COLS), 1, i)
        plt.plot(tmp_df['pred'].loc[:max_len], ':o', label='Прогноз данных')

        plt.plot(tmp_df.loc[:limit, 'y'], 'ro', label='Данные для анализа')
        plt.axvline(x=limit)
        plt.xlabel('index')
        plt.ylabel(y_col)
        plt.legend()
        i += 1
    plt.tight_layout()
    if verbous:
        plt.show()
    else:
        plt.savefig('corr_data_plot.png')



def neuro_window(learn_df):
    X_LIMIT = 0
    Y_LIMIT = 0
    cols = learn_df.columns.tolist()
    for col in cols:
        if col.startswith('X'):
            X_LIMIT += 1
        elif col.startswith('Y'):
            Y_LIMIT += 1

    neuros_dict = {
        'Регрессия': 'relu',
        'Кластеризация': 'sigmoid'
    }
    neuros_list = list(neuros_dict.keys())
    database = Database()
    # learn_df = None
    normalized_df = None
    normalized_learn_df_x = None
    model = None
    pred = None
    X_COLS = [f'X{i}' for i in range(1, X_LIMIT + 1)]
    Y_COLS = [f'Y{i}' for i in range(1, Y_LIMIT + 1)]
    
    st.text('Нейросети', justification='center', font=("Helvetica", 20))
    st.text('_' * 92)

    for y_col in Y_COLS:
        st.text(f'Тип нейронной сети для {y_col}'), st.selectbox(neuros_list, default_value=neuros_list[0], key=y_col)
   
        st.text('% данных на теcтовую выборку:'), st.text_input(25, size=(5, 1), key='test')
        st.text('Число нейронов входящего слоя нейросети:'), st.text_input(64, size=(5, 1), key='neurons')
        st.text('Число эпох для обучения нейросети:'), st.text_input(100, size=(5, 1), key='epohs')
        st.button('Нормировать данные'), st.button('Обучить модель'), st.button('Ошибки при обучении')
        st.button('Прогнозирование данных'), st.button('Сохранить png')
        st.text('_' * 92)
        st.text('Y для обзорной панели:')
    
    checkbox_list = []
    for y_col in Y_COLS:
        checkbox_list.append(st.checkbox(y_col, default=True, key=f"_{y_col}"))
        checkbox_list
        st.text('Число данных(размер выборки) для панели:'), st.text_input('', size=(5, 1), key='size')
        st.button('Обзорная панель'), st.button('Сохранить в Excell')
        # [sg.Submit('Сделать прогноз по модели'), sg.Submit('Сохранить результат прогноза в Excell')],
        st.text_area(size=(90, 15), key='-OUTPUT-')
        st.button('Выход')
    
    window = st.window(TITLE, layout)
    df_flag = False
    while True:
        event, values = window.read()
        if event in (None, 'Выход', sg.WIN_CLOSED):
            database.close()
            window.close()
            return

        if event == 'Загрузить обучающую выборку':
            file_path = values['file']
            if not file_path:
                sg.popup('Выберите файл')
                continue
            if not file_path.endswith('.xlsx'):
                sg.popup('Поддерживается только .XLSX')
                continue
            learn_df = pd.read_excel(file_path)
            print(learn_df)
            sg.popup('Выборка загружена')

        if event == 'Нормировать данные':
            if learn_df is None:
                sg.popup('Загрузите обучающую выборку')
                continue
            if not database.check_table(DATA_SQL_TABLE):
                sg.popup('Сохраните предварительно основные данные в БД')
                continue
            df = database.read_data()

            # Очищаемся от пустых данных
            df = df.dropna()
            learn_df = learn_df.dropna()

            # Нормализуем
            normalized_df = (df - df.mean()) / df.std()
            normalized_learn_df_x = (learn_df[X_COLS] - learn_df[X_COLS].mean()) / learn_df[X_COLS].std()
            min_y_dict = {}
            max_y_dict = {}
            normalized_y_dict = {}
            for y_col in Y_COLS:
                min_y_dict[y_col] = learn_df[y_col].min()
                max_y_dict[y_col] = learn_df[y_col].max() - min_y_dict[y_col]
                normalized_y_dict[y_col] = (learn_df[y_col] - min_y_dict[y_col]) / max_y_dict[y_col]

            print(normalized_learn_df_x)
            print('Данные нормированы')
            sg.popup('Данные нормированы')

        if event == 'Обучить модель':
            test_size = int(values['test'])
            neurons = int(values['neurons'])
            epochs = int(values['epohs'])

            if normalized_learn_df_x is None:
                sg.popup('Нормируйте данные')
                continue

            # Отделяем Х
            X = np.asarray(normalized_learn_df_x).astype(float)
            input_dim = len(X_COLS)
            Y_dict = {}
            model_dict = {}
            history_dict = {}
            for y_col in Y_COLS:
                # Отделяем У
                Y_dict[y_col] = np.asarray(normalized_y_dict[y_col]).astype(float)

                # Делим обучающую выборку на train и test
                X_train, X_test, Y_train, Y_test = train_test_split(X, Y_dict[y_col], test_size=test_size / 100, random_state=22)
                # print('X_train, X_test, Y_train, Y_test')

                # model = Sequential()
                # model.add(Dense(int(values['neurons']), input_dim=len(X_COLS), activation='relu'))
                # model.add(Dense(int(values['neurons']), activation='relu'))
                # model.add(Dense(1, activation='relu')
                # model = Sequential()
                # model.add(Dense(int(values['neurons']), input_dim=len(X_COLS), activation='relu'))
                # model.add(Dense(int(values['neurons']), activation='relu'))
                # model.add(Dense(1, activation='sigmoid'))

                # Строим простую полносвязную нейронную сеть. Выходной слой с одним линейным нейроном — для задачи регрессии.
                model_dict[y_col] = prepare_model(neurons, input_dim, activation=neuros_dict[values[y_col]])
                print("Описание модели")
                print(model_dict[y_col].summary())

                # Компилируем модель
                # Т.к. задача регрессии, удобнее использовать mean square error(средне-квадратичная ошибка).
                # В качестве метрики берем mean absolute error (средний модуль ошибки)
                model_dict[y_col].compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

                # Проводим обучение в заданное число эпох
                history_dict[y_col] = model_dict[y_col].fit(X_train, Y_train, epochs=epochs, validation_split=0.1, verbose=2)
                print(f"Обучение модели для {y_col} завершено")
            print("Обучение завершено")
            sg.popup("Обучение завершено")

        # if event == 'Сделать прогноз по модели':
        #     if model is None:
        #         sg.popup('Обучите модель')
        #         continue
        #     pred =

def main():
    learn_df = None
    uploadfile= st.file_uploader("Выберите файл", type=['txt', 'csv', 'xlsx'])
    load_button = st.button("загрузить", key="button_load") 
    if load_button:
        if learn_df is not None:
            learnf_df = pd.read_excel(uploadfile)
            st.write(learnf_df)
            neuro_window(learnf_df)


if __name__ == "__main__":
    main()
    
    