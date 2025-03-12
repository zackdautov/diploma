import streamlit as st
from typing import List, Dict

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
# pip install pandas, PySimpleGUI, openpyxl, matplotlib
import tensorflow as tf
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


if 'clicks' not in st.session_state:
    st.session_state['clicks'] = {}

if 'message_cons' not in st.session_state:
    st.session_state['message_cons'] = {}

if 'dataex' not in st.session_state:
    st.session_state['dataex'] = {}

if 'data_norma' not in st.session_state:
    st.session_state['data_norma'] = {}
   

if 'history_dicts' not in st.session_state:
    st.session_state['history_dicts'] = {}

if 'model_dicts' not in st.session_state:
    st.session_state['model_dicts'] = {}

def model_dicts(y_cols,key):
    st.session_state.model_dicts[y_cols] = key

def history_dicts(y_cols,key):
    st.session_state.history_dicts[y_cols] = key

def data_norma(df_x,min,max,normalized_y_dict):
    st.session_state.df_x = df_x
    st.session_state.min = min
    st.session_state.max = max
    st.session_state.normalized_y_dict = normalized_y_dict

def click(key):
    st.session_state.clicks[key] = True

def message_cons(key):
    st.session_state.message_cons = f"{st.session_state.message_cons} \n {str(key)}"

def dataex(key):
    st.session_state.dataex = key

def prepare_model(neurons: int, input_dim: int, activation='sigmoid'):
    """
    Вспомогательная фугкция подготовки модели НС
    """
    model = Sequential()
    model.add(Dense(neurons, input_dim=input_dim, activation='relu'))
    model.add(Dense(neurons, activation='relu'))
    model.add(Dense(1, activation=activation))
    return model


def model_plot(Y_COLS: List[str], history_dict: Dict, values: dict, verbose=True):
    """
    Вспомогательная функция графика метрик модели
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
    if verbose:
        st.pyplot()  
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
        st.pyplot() 
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



def neuro_window():
    learn_df = None
    load_button = False
    uploadfile= st.file_uploader("Выберите файл", type=['txt', 'csv', 'xlsx'])
    load_button = st.button("загрузить", key="button_load") 
    if load_button:
        click("load_button")
    if st.session_state.clicks["load_button"]:
        learn_df = pd.read_excel(uploadfile)
        dataex(learn_df)
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
        
        normalized_df = None
        normalized_learn_df_x = None
        model = None
        pred = None
        X_COLS = [f'X{i}' for i in range(1, X_LIMIT + 1)]
        Y_COLS = [f'Y{i}' for i in range(1, Y_LIMIT + 1)]
        
        st.text('Нейросети')
        
        values = {}
        for y_col in Y_COLS:
            st.write(f'Тип нейронной сети для {y_col}')
            values[y_col] =  st.selectbox("",neuros_list, key=y_col)
    
        test = st.text_input('% данных на теcтовую выборку:', 25)
        neurons = st.text_input('Число нейронов входящего слоя нейросети:', 64)
        epohs = st.text_input('Число эпох для обучения нейросети:', 100)
        
        normal_button = st.button('Нормировать данные')
        learn_button = st.button('Обучить модель')
        errors_button = st.button('Ошибки при обучении')
        predicrion_button = st.button('Прогнозирование данных')
        st.text('Y для обзорной панели:')
        
        checkbox_list = []
        for y_col in Y_COLS:
            checkbox_list.append(st.checkbox(y_col, value=True, key=f"_{y_col}"))
        checkbox_list
        size = st.text_input('Число данных(размер выборки) для панели:', '')
        st.button('Обзорная панель')
        save_excell = st.button('Сохранить в Excell')
            # [sg.Submit('Сделать прогноз по модели'), sg.Submit('Сохранить результат прогноза в Excell')],
        
        
        
        df_flag = False
    # while True:
    #     event, values = window.read()
        

        if normal_button:
            if learn_df is None:
                message_cons('Загрузите обучающую выборку')
            elif not database.check_table(DATA_SQL_TABLE):
                st.error('Сохраните предварительно основные данные в БД')
            else:
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
                    data_norma(normalized_learn_df_x,min_y_dict,max_y_dict,normalized_y_dict)

                st.write(normalized_learn_df_x)
                message_cons("Данные нормированы")
                

        if learn_button:
            test_size = int(test)
            neurons = int(neurons)
            epochs = int(epohs)
            normalized_learn_df_x = st.session_state.df_x
            if normalized_learn_df_x is None:
                st.error('Нормируйте данные')
            else:

            # Отделяем Х
                X = np.asarray(normalized_learn_df_x).astype(float)
                input_dim = len(X_COLS)
                Y_dict = {}
                model_dict = {}
                history_dict = {}
                #normalized_learn_df_x = st.session_state.df_x
                min_y_dict = st.session_state.min
                max_y_dict = st.session_state.max
                normalized_y_dict = st.session_state.normalized_y_dict
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
                    # message_cons("Описание модели")
                    # message_cons(str(model_dict[y_col].summary()))

                    # Компилируем модель
                    # Т.к. задача регрессии, удобнее использовать mean square error(средне-квадратичная ошибка).
                    # В качестве метрики берем mean absolute error (средний модуль ошибки)
                    model_dict[y_col].compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

                    # Проводим обучение в заданное число эпох
                    history_dict[y_col] = model_dict[y_col].fit(X_train, Y_train, epochs=epochs, validation_split=0.1, verbose=2)
                    st.session_state.history_dicts[y_col] = model_dict[y_col].fit(X_train, Y_train, epochs=epochs, validation_split=0.1, verbose=2)
                    st.session_state.model_dicts[y_col] = model_dict[y_col]
                    message_cons(f"Обучение модели для {y_col} завершено")
                st.write("Обучение завершено")
        
        if save_excell:
            history_dict = {}
            model_dict = {}
            
            if not st.session_state.history_dicts[y_col]:
                message_cons('Обучите модель')
            else:
                normalized_learn_df_x = st.session_state.df_x
                for y_col in Y_COLS:
                    history_dict[y_col] = st.session_state.history_dicts[y_col] 
                    model_dict[y_col] = st.session_state.model_dicts[y_col] 
                max_y_dict = st.session_state.max
                min_y_dict = st.session_state.min
                for y_col in Y_COLS:  
                    pred = model_dict[y_col].predict(normalized_learn_df_x).flatten()
                    # Возвращем данные к нашей размерности - обратное нормирование
                    learn_df[y_col] = pred * max_y_dict[y_col] + min_y_dict[y_col]
                res_file = "result12.xlsx"
                learn_df[X_COLS + Y_COLS].to_excel(res_file)
                print(f"Результат сохранен в {res_file}")
                message_cons(f"Результат сохранен в {res_file}")
        
        if errors_button:
            history_dict = {}

            if not st.session_state.history_dicts[y_col]:
                message_cons('Обучите модель')
            else:
                for y_col in Y_COLS:
                    history_dict[y_col] = st.session_state.history_dicts[y_col]
                model_plot(Y_COLS, history_dict, values)

        if predicrion_button:
            history_dict = {}
            model_dict = {}
            test_size = int(test)
            if not st.session_state.history_dicts[y_col]:
                message_cons('Обучите модель')
            else:
                for y_col in Y_COLS:
                    history_dict[y_col] = st.session_state.history_dicts[y_col]
                    model_dict[y_col] = st.session_state.model_dicts[y_col] 
                data_plot(test_size, Y_COLS, st.session_state.model_dicts, st.session_state.df_x, st.session_state.max, st.session_state.min, st.session_state.dataex,
                      verbous=True)
        
        st.text_area("console",str(st.session_state.message_cons))
        exit_neuro = st.button('Выход', key="exit")
        if exit_neuro:
            database.close()
            return

def main():
    neuro_window()


if __name__ == "__main__":
    main()
    st.write(st.session_state)
    