import streamlit as st
from typing import List, Dict

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
warnings.filterwarnings("ignore")

from database import Database
# from settings import DEFAULT_USER, DEBUG_MODE, DATA_SQL_TABLE, TITLE, Y_LIMIT, X_LIMIT
from settings import DEFAULT_USER, DEBUG_MODE, DATA_SQL_TABLE, TITLE, Y_LIMIT, X_LIMIT, AboutProgram1, AboutProgram2, AboutProgram3, AboutProgram4, AboutProgram5, AboutProgram6, AboutProgram7

st.set_page_config(
    page_title='НГДУ "Джалильнефть"',
    page_icon=":bar_chart:",
    layout="wide")

st.set_option('deprecation.showPyplotGlobalUse', False)


st.markdown('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)

if 'clicks' not in st.session_state:
    st.session_state['clicks'] = {}

if 'dataex1' not in st.session_state:
    st.session_state['dataex1'] = {}

if 'dataex3' not in st.session_state:
    st.session_state['dataex3'] = {}

if 'userser' not in st.session_state:
    st.session_state['userser'] = {}

if 'df_flag' not in st.session_state:
    st.session_state['df_flag'] = {}

if 'message_cons' not in st.session_state:
    st.session_state['message_cons'] = {}

if 'message_cons_corr' not in st.session_state:
    st.session_state['message_cons_corr'] = {}

if 'logs' not in st.session_state:
    st.session_state['logs'] = ""

if 'dataex' not in st.session_state:
    st.session_state['dataex'] = {}

if 'data_norma' not in st.session_state:
    st.session_state['data_norma'] = {}

if 'data_reg' not in st.session_state:
    st.session_state['data_reg'] = {}

if 'data_corr' not in st.session_state:
    st.session_state['data_corr'] = {}
   

if 'history_dicts' not in st.session_state:
    st.session_state['history_dicts'] = {}

if 'model_dicts' not in st.session_state:
    st.session_state['model_dicts'] = {}

if 'model_dicts_reg' not in st.session_state:
    st.session_state['model_dicts_reg'] = {}

if 'model_dicts_corr' not in st.session_state:
    st.session_state['model_dicts_corr'] = {}

def model_dicts(y_cols,key):
    st.session_state.model_dicts[y_cols] = key

def model_dicts_reg(y_cols,key):
    st.session_state.model_dicts_reg[y_cols] = key

def model_dicts_corr(y_cols,key):
    st.session_state.model_dicts_corr[y_cols] = key

def history_dicts(y_cols,key):
    st.session_state.history_dicts[y_cols] = key

def data_norma(df_x,min,max,normalized_y_dict):
    st.session_state.df_x = df_x
    st.session_state.min = min
    st.session_state.max = max
    st.session_state.normalized_y_dict = normalized_y_dict

def data_reg(df_x,min,max,normalized_y_dict):
    st.session_state.reg_df_x = df_x
    st.session_state.reg_min = min
    st.session_state.reg_max = max
    st.session_state.reg_normalized_y_dict = normalized_y_dict

def data_corr(df_x,min,max,normalized_y_dict):
    st.session_state.corr_df_x = df_x
    st.session_state.corr_min = min
    st.session_state.corr_max = max
    st.session_state.corr_normalized_y_dict = normalized_y_dict



def click(key):
    st.session_state.clicks[key] = True

def message_cons(key):
    st.session_state.message_cons = f"{st.session_state.message_cons} \n {str(key)}"

def message_cons_corr(key):
    st.session_state.message_cons_corr = f"{st.session_state.message_cons_corr} \n {str(key)}"

def logs(key):
    st.session_state.logs = f"{st.session_state.logs} {str(key)}"

def dataex1(key):
    st.session_state.dataex1 = key

def dataex3(key):
    st.session_state.dataex3 = key

def unclick(key):
    st.session_state.clicks[key] = False

def df_flag():
    st.session_state.f_flag = False

def userses(key):
    st.session_state.userser = key

def dataex(key):
    st.session_state.dataex = key


def login():
    st.sidebar.subheader("Авторизация")
    login = st.sidebar.text_input("Введите имя пользователя")
    pasw = st.sidebar.text_input("Введите пароль", type="password")
    login_button = st.sidebar.button("Войти",on_click=click, args=(f'button_login',) ) 

    database = Database()
    if login_button:
            
        if len(login) == 0 or len(pasw) == 0:
            st.error('Поля логина и пароля не должны быть пустыми')
        user = database.get_user(login)
        if not user:
            st.error('Пользователя не существует')
        if pasw == str(user[3]):
            database.close()
            userses(user)
            return(user)
        else:
         st.error('Неверный пароль')

def register():
    database = Database()
    st.sidebar.subheader("Регистрация")
    st.sidebar.text('Поля * обязательны для заполнения')

    login = st.sidebar.text_input("Логин*")
    paswrd = st.sidebar.text_input("Пароль*", type="password")
    name = st.sidebar.text_input("Имя*")
    surname = st.sidebar.text_input("Фамилия")
    role = st.sidebar.text_input("Права*(Гость/Админ)")
    register_button = st.sidebar.button("Зарегистрироваться", key="button_register") 


    if register_button:
        pasw =  paswrd
        sname = surname    
        if len(login) == 0 or len(pasw) == 0 or len(name) == 0 or len(role) == 0:
            st.error('Поля * не должны быть пустыми')
        
        user = database.get_user(login)
        if user:
            st.error('Пользователь существует')
              
        else:
            database.insert_user(name, sname, login, pasw, role)
            st.error('Пользователь создан')

def main_window(user):
    database = Database()
    st.sidebar.subheader(f"{user[0]}")
    st.sidebar.button("Выйти",on_click=click, args=(f'button_logout',) ) 
    #if st.session_state.clicks.get(f'button_logout', True):
    #    st.session_state['userser'] = False

    st.title(f'Добро пожаловать, {user[0]}')
    role = user[4]
    st.header(f'Вашу уровень прав в программе: {role}')
    st.text('Загрузить данные из файла:')
    uploaded_file1 = st.file_uploader("Выберите файл", type=[ 'xlsx'])
    #st.text('Загрузить данные для нейросети, регрессии и корреляции:')
#uploaded_file = st.file_uploader("Выберите файл", type=['txt', 'csv', 'xlsx'])
    load_button = st.button("загрузить", key="button_load") 
    coll1, coll2, coll3= st.columns(3)
    with coll1:
        savedata_button = st.button("Сохранить данные в БД", key="button_savedata") 
    with coll2:
        loaddata_button = st.button("Загрузить данные из БД", key="button_loaddata") 
    with coll3:
        cleardata_button = st.button("Очистить данные в БД", key="button_cleardata") 


    df_flag = False
    if load_button:
        if uploaded_file1 is not None:
            file_path = uploaded_file1
            excel_data_df = pd.read_excel(uploaded_file1)
            dataex1(excel_data_df)
            st.write(st.session_state.dataex1)
            st.session_state.df_flag = True

    if savedata_button:
        if not st.session_state.df_flag:
            st.error('Нет данных для сохранения')
        database.drop_table(DATA_SQL_TABLE)
        database.save_data(st.session_state.dataex1)
        st.write('Данные сохранены в БД')

    if loaddata_button:
        if not database.check_table(DATA_SQL_TABLE):
            st.error('База данных пустая')
        df = database.read_data()
        print(df)
        st.write('Данные прочитаны из БД')
        
         
    if cleardata_button:
        database.drop_table(DATA_SQL_TABLE)
        st.write('Данные очищены в БД')
    col1, col2, col3, col4, col5, col6= st.columns(6)
    with col1:
        statdata_button = st.button("Статистика", args=[{"style":"white-space: nowrap;"}], key="button_statdata") 
    with col2:
        normdata_button = st.button("Нормальность",args=[{"style": "white-space: nowrap;"}], key="button_normdata") 
    with col3:
        diagrdata_button = st.button("Диаграмма",args=[{"style": "white-space: nowrap;"}], key="button_diagrdata") 
    with col4:
        neirodata_button = st.button("Нейросети",args=[{"style": "white-space: nowrap;"}], key="button_neirodata") 
    with col5:
        regrdata_button = st.button("Регрессионный анализ",args=[{"style": "white-space: nowrap;"}], key="button_regrdata") 
    with col6:
        correldata_button = st.button("Корреляционный анализ",args=[{"style": "white-space: nowrap;"}], key="button_correldata")
        
    if statdata_button:
        if not database.check_table(DATA_SQL_TABLE):
            st.error('Сохраните предварительно данные в БД')
        stat_window()
    if normdata_button:
        if not database.check_table(DATA_SQL_TABLE):
            st.error('Сохраните предварительно данные в БД')
        norm_window()
    if diagrdata_button:
        if not database.check_table(DATA_SQL_TABLE):
            st.error('Сохраните предварительно данные в БД')
        diagrdata_window()
    if neirodata_button:
        logs("neirodata_button")
        st.session_state.logs.replace("correldata_button", "", 1) 
        st.session_state.logs.replace("regrdata_button", "", 1) 
    if (st.session_state.logs).find("neirodata_button") > 0:
        neuro_window()
    if regrdata_button:
        logs("regrdata_button")
        st.session_state.logs.replace("neirodata_button", "", 1) 
        st.session_state.logs.replace("correldata_button", "", 1) 
    if (st.session_state.logs).find("regrdata_button") > 0:
        regress_window()
    if correldata_button:
        logs("correldata_button")
        st.session_state.logs.replace("regrdata_button", "", 1) 
        st.session_state.logs.replace("neirodata_button", "", 1) 
    if (st.session_state.logs).find("correldata_button") > 0:
        corr_window()
        
def stat_window():
    database = Database()
    df = database.read_data()
    cols = df.columns.values
    stats = {}
    stats['Показатель'] = cols
    for col in cols:
        if 'Медиана' not in stats:
            stats['Медиана'] = []
        stats['Медиана'].append(df[col].median().round(4))
        if 'Среднее' not in stats:
            stats['Среднее'] = []
        stats['Среднее'].append(df[col].mean())
        if 'Мин' not in stats:
            stats['Мин'] = []
        stats['Мин'].append(df[col].min().round(4))
        if 'Макс' not in stats:
            stats['Макс'] = []
        stats['Макс'].append(df[col].max().round(4))
        if 'CтдОшСред' not in stats:
            stats['CтдОшСред'] = []
        stats['CтдОшСред'].append((df[col].median() - df[col].mean()).round(4))
        if 'Дисперсия' not in stats:
            stats['Дисперсия'] = []
        stats['Дисперсия'].append(df[col].var())
        if 'Ассиметрия' not in stats:
            stats['Ассиметрия'] = []
        stats['Ассиметрия'].append(df[col].skew().round(4))
        if 'Эксцесс' not in stats:
            stats['Эксцесс'] = []
        stats['Эксцесс'].append(df[col].kurtosis().round(4))
        if 'CтдОш' not in stats:
            stats['CтдОш'] = []
        stats['CтдОш'].append(df[col].std())
    stat_df = pd.DataFrame(stats, index=cols)

    headings = list(stats)
    values = stat_df.values.tolist()
 
    st.subheader('Статистические показатели')
    df = pd.DataFrame(values, columns=headings)
    st.table(df)

def norm_window():
    database = Database()
    df = database.read_data()
    cols = df.columns.values
    stats = {}
    stats['Показатель'] = cols
    for col in cols:
        if 'Нормальность по медиане' not in stats:
            stats['Нормальность по медиане'] = []
        param = abs((df[col].median() - df[col].mean()) / df[col].median()) * 100
        res = '-'
        if param < 10:
            res = '+'
        stats['Нормальность по медиане'].append(res)

        if 'Нормальность по ассим.' not in stats:
            stats['Нормальность по ассим.'] = []
        res = '-'
        if -1 < df[col].skew() < 1:
            res = '+'
        stats['Нормальность по ассим.'].append(res)

        if 'Нормальность по эксцессу' not in stats:
            stats['Нормальность по эксцессу'] = []
        res = '-'
        if 2 < df[col].kurtosis() < 4:
            res = '+'
        stats['Нормальность по эксцессу'].append(res)

    stat_df = pd.DataFrame(stats, index=cols)

    headings = list(stats)
    values = stat_df.values.tolist()

   
    st.subheader('Статистические показатели')
    df = pd.DataFrame(values, columns=headings)
    st.table(df)
    
    
def diagrdata_window():
    database = Database()
    df = database.read_data()
    cols = df.columns.values
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Диаграмма")
    ax.set_xlim([0, 50])
    ax.set_xlabel("X")
    ax.plot(df)
    ax.legend(cols)
    plt.tight_layout()
    st.pyplot(fig)
    database.close()


    
    
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
    print(test_size, Y_COLS, model_dict, normalized_learn_df_x, max_y_dict, min_y_dict, learn_df)
    i = 1
    plt.suptitle('Результаты регрессионного анализа', fontsize=18)

    if size: 
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
        st.pyplot() 
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
        st.pyplot()
    else:
        plt.savefig('corr_data_plot.png')



def neuro_window():
    learn_df = None
    uploadfile= st.file_uploader("Выберите файл", type=['txt', 'csv', 'xlsx'])
    load_neuro_button = st.button("загрузить", key = "load_neuro_button") 
    if load_neuro_button:
        logs('load_neuro_button')
    if (st.session_state.logs).find('load_neuro_button') > 0:
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
        
        st.title('Нейросети')
        
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
        exit_neuro = st.button('Выход',on_click=unclick, args=(f'neurodata_button',)) 
        if exit_neuro:
            database.close()
            return    
    
def forward_regression(X, y):
    threshold_in = 0.4
    initial_list = []
    included = list(X.columns)
    while True:
        changed = False
        excluded = list(set(X.columns) - set(included))
        new_pval = pd.Series(index=excluded)
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max()
        if worst_pval > 1 - threshold_in:
            changed = True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
        if not changed:
            break
    return included


def backward_regression(X, y):
    threshold_out = 0.05
    included = list(X.columns)
    while True:
        changed = False
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max()
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
        if not changed:
            break
    return included


def all_regression(X, y):
    return list(X.columns)


def regress_window():
    learn_df = None
    uploadfile2= st.file_uploader("Выберите файл", type=['xlsx'], key = "regres")
    load_regres_button = st.button("загрузить", key = "load_regres_button") 
    if load_regres_button:
        logs("load_regres_button")  
    if (st.session_state.logs).find("load_regres_button") > 0:
        learn_df = pd.read_excel(uploadfile2)
        dataex3(learn_df)
        X_LIMIT = 0
        Y_LIMIT = 0
        cols = learn_df.columns.tolist()
        for col in cols:
            if col.startswith('X'):
                X_LIMIT += 1
            elif col.startswith('Y'):
                Y_LIMIT += 1


        elim_dict = {
            'Без исключения': all_regression,
            'Forward regression': forward_regression,
            'Backward regression': backward_regression,

        }
        database = Database()
        # learn_df = None
        normalized_df = None
        normalized_learn_df_x = None
        model = None
        pred = None
        model_dict = []
        values = {}
        X_COLS = [f'X{i}' for i in range(1, X_LIMIT + 1)]
        Y_COLS = [f'Y{i}' for i in range(1, Y_LIMIT + 1)]
        
        st.title('Линейный регрессионный анализ')
            # [sg.Text('_' * 92)],
            # [sg.Text('Загрузить данные из файла:')],
            # [sg.InputText(size=(45, 1), key='file'), sg.FileBrowse('Обзор'), sg.Submit('Загрузить данные')],

        st.text(f'Независимые переменные (регрессоры):')
        st.text(X_COLS)
        st.text(f'Зависимые переменные: {Y_COLS}')
        test = st.text_input('% данных для тестирования:', 25)
        st.write('Метод исключения признаков')
        elim = st.selectbox( "" , elim_dict)
        normalreq_button = st.button('Нормировать данные')
        createreq_button = st.button('Построить модели')
        kefreq_button = st.button('Коэффициенты моделей')
        st.text('Y для диаграммы:')
        checkbox_list = []
        for y_col in Y_COLS:
            checkbox_list.append(st.checkbox(y_col, value=True, key=f"_{y_col}"))
        checkbox_list,
        size = st.text_input('Число данных(размер выборки) для панели:', '')
        predict_reg_button = st.button('Прогнозирование данных')
        save_reg_button = st.button('Сохранить png')
        save_excell = st.button('Сохранить в Excell')

        df_flag = False
        learn_df = pd.read_excel(uploadfile2)
        print(learn_df)
        #st.write('Данные загружены')
        test_size = int(test)
        
        if normalreq_button:
            if learn_df is None:
                st.error('Загрузите данные')
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
                    
                print(normalized_learn_df_x)
                print('Данные нормированы')

                # Features elimination
                X_COLS = elim_dict[elim](normalized_learn_df_x, normalized_y_dict[y_col])
                normalized_learn_df_x = normalized_learn_df_x[X_COLS]
                print("Были отобраны следующие признаки:")
                print(X_COLS)

                st.write('Данные нормированы, признаки отобраны')
                data_reg(normalized_learn_df_x,min_y_dict,max_y_dict,normalized_y_dict)

        if createreq_button:
            normalized_learn_df_x = st.session_state.reg_df_x
            min_y_dict = st.session_state.reg_min
            max_y_dict = st.session_state.reg_max
            normalized_y_dict = st.session_state.reg_normalized_y_dict

            if normalized_learn_df_x is None:
                st.error('Нормируйте данные')

            else:    
                X = np.asarray(normalized_learn_df_x).astype(float)
                input_dim = len(X_COLS)
                Y_dict = {}
                model_dict = {}
                history_dict = {}

                for y_col in Y_COLS:
                    print(f"Модель линейной регреcсии для {y_col}:")
                    model_dict[y_col] = LinearRegression().fit(X, normalized_y_dict[y_col])
                    r_sq = model_dict[y_col].score(X, normalized_y_dict[y_col])
                    model_dicts_reg(y_col,model_dict[y_col])
                    print('coefficient of determination:', r_sq)
                    print('intercept:', model_dict[y_col].intercept_)
                    print("-" * 30)

                print("Все модели линейной регрессии построены")
                st.write("Все модели линейной регрессии построены")

        if save_excell:
            model_dict = {}
            normalized_learn_df_x = st.session_state.reg_df_x
            min_y_dict = st.session_state.reg_min
            max_y_dict = st.session_state.reg_max
            normalized_y_dict = st.session_state.reg_normalized_y_dict
            for y_col in Y_COLS:
                model_dict[y_col] = st.session_state.model_dicts_reg[y_col]

            if not model_dict:
                st.error('Постройте модель')
            else:
                for y_col in Y_COLS:
                    pred = model_dict[y_col].predict(normalized_learn_df_x).flatten()
                    #Возвращем данные к нашей размерности - обратное нормирование
                    learn_df[y_col] = pred * max_y_dict[y_col] + min_y_dict[y_col]
                res_file = "result_reg.xlsx"
                learn_df[X_COLS + Y_COLS].to_excel(res_file)
                print(f"Результат сохранен в {res_file}")
                st.write(f"Результат сохранен в {res_file}")

        if kefreq_button:
            model_dict = {}
            normalized_learn_df_x = st.session_state.reg_df_x
            min_y_dict = st.session_state.reg_min
            max_y_dict = st.session_state.reg_max
            normalized_y_dict = st.session_state.reg_normalized_y_dict
            for y_col in Y_COLS:
                model_dict[y_col] = st.session_state.model_dicts_reg[y_col]

            if not model_dict:
                st.error('Постройте модель')
            else:    
                for y_col in Y_COLS:
                    print(f"Коэффициенты линейной регресии для {y_col}:")
                    print('slope:', model_dict[y_col].coef_)
                    print("-" * 30)
                st.write("Коэффициенты отображены")

        if predict_reg_button:
            model_dict = {}
            normalized_learn_df_x = st.session_state.reg_df_x
            min_y_dict = st.session_state.reg_min
            max_y_dict = st.session_state.reg_max
            normalized_y_dict = st.session_state.reg_normalized_y_dict
            for y_col in Y_COLS:
                model_dict[y_col] = st.session_state.model_dicts_reg[y_col]

            if not model_dict:
                st.error('Постройте модель')
            else:    
                try:
                    size = int(values['size'])
                except:
                    size = None
                new_y_cols = []
                print(values)
                for y_col in Y_COLS:
                    if values.get(f"_{y_col}"):
                        new_y_cols.append(y_col)
                regress_data_plot(test_size, Y_COLS, model_dict, normalized_learn_df_x, max_y_dict, min_y_dict, learn_df,
                            verbous=True)
        if save_reg_button:
            model_dict = {}
            normalized_learn_df_x = st.session_state.reg_df_x
            min_y_dict = st.session_state.reg_min
            max_y_dict = st.session_state.reg_max
            normalized_y_dict = st.session_state.reg_normalized_y_dict
            for y_col in Y_COLS:
                model_dict[y_col] = st.session_state.model_dicts_reg[y_col]

            if not model_dict:
                st.error('Постройте модель')
            else:
                regress_data_plot(test_size, Y_COLS, model_dict, normalized_learn_df_x, max_y_dict, min_y_dict, learn_df,
                            verbous=False)
                st.write("Графики сохранены в png файлы")

def corr_text(corr_df, x_cols, y_cols):
    res = ''
    for x in x_cols:
        for y in y_cols:
            res += f'{x}-{y}:{abs(corr_df.loc[x][y])}\n'
    return res


def filter_cols_by_corr(corr_df, x_cols, y_cols, corr_level):
    res = []
    for x in x_cols:
        flag = True
        for y in y_cols:
            if abs(corr_df.loc[x][y]) < corr_level:
                flag = False
        if flag:
            res.append(x)
    return res

def corr_window():
    learn_df = None
    uploadfile3= st.file_uploader("Выберите файл", type=['xlsx'], key = "regres")
    load_corr_button = st.button("загрузить", key = "load_corr_button") 
    if load_corr_button:
        logs("load_corr_button")  
    if (st.session_state.logs).find("load_corr_button") > 0:
        learn_df = pd.read_excel(uploadfile3)
        dataex3(learn_df)
        X_LIMIT = 0
        Y_LIMIT = 0
        cols = learn_df.columns.tolist()
        for col in cols:
            if col.startswith('X'):
                X_LIMIT += 1
            elif col.startswith('Y'):
                Y_LIMIT += 1

        cols = learn_df.columns.tolist()
        corr_cols = []
        for col in cols:
            if col.startswith('X') or col.startswith('Y'):
                corr_cols.append((col))
        corr_matrix = learn_df.loc[:, corr_cols].corr()

        database = Database()
        # learn_df = None
        normalized_df = None
        normalized_learn_df_x = None
        model = None
        pred = None
        model_dict = []
        values = {}
        X_COLS_START = [f'X{i}' for i in range(1, X_LIMIT + 1)]
        X_COLS = X_COLS_START
        Y_COLS = [f'Y{i}' for i in range(1, Y_LIMIT + 1)]
        res_text = corr_text(corr_matrix, X_COLS_START, Y_COLS)
        st.title('Корреляционный анализ')
        test = st.text_input('% данных для тестирования:', 25)
        corr_matrix_button = st.button('Корреляционная матрица')
        
        min_corr_lvl = st.slider('Настройка min уровня корреляции', 0.0, 1.0, 0.0, 0.01)
        normalcorr_button = st.button('Нормировать данные')
        createcorr_button = st.button('Построить модели')
        corr_ur_button = st.button('Корреляционные уравнения')
        st.text('Y для диаграммы:')
        checkbox_list = []
        for y_col in Y_COLS:
            checkbox_list.append(st.checkbox(y_col, value=True, key=f"_{y_col}"))
        checkbox_list,
        size = st.text_input('Число данных(размер выборки) для панели:', '')
        predict_corr_button = st.button('Прогнозирование данных')
        save_corr_button = st.button('Сохранить png')
        save_excell = st.button('Сохранить в Excell')
        df_flag = False
        old_level = 0

        if float(min_corr_lvl) != old_level:
            old_level = float(min_corr_lvl)
            X_COLS = filter_cols_by_corr(corr_matrix, X_COLS_START, Y_COLS, old_level)
            res_text = corr_text(corr_matrix, X_COLS, Y_COLS)
            message_cons_corr(res_text)

        #     if event in (None, 'Выход', sg.WIN_CLOSED):
        #         database.close()
        #         window.close()
        #         return

        if normalcorr_button:
            if learn_df is None:
                st.error('Загрузите данные')
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
                    
                message_cons_corr(normalized_learn_df_x)
                print('Данные нормированы')
                message_cons_corr('Данные нормированы, признаки отобраны')
                data_corr(normalized_learn_df_x,min_y_dict,max_y_dict,normalized_y_dict)

            # if event == 'Нормировать данные':
            #     if learn_df is None:
            #         sg.popup('Загрузите данные')
            #         continue
            #     if not database.check_table(DATA_SQL_TABLE):
            #         sg.popup('Сохраните предварительно основные данные в БД')
            #         continue
            #     df = database.read_data()

            #     # Очищаемся от пустых данных
            #     df = df.dropna()
            #     learn_df = learn_df.dropna()

            #     # Нормализуем
            #     normalized_df = (df - df.mean()) / df.std()
            #     normalized_learn_df_x = (learn_df[X_COLS] - learn_df[X_COLS].mean()) / learn_df[X_COLS].std()
            #     min_y_dict = {}
            #     max_y_dict = {}
            #     normalized_y_dict = {}
            #     for y_col in Y_COLS:
            #         min_y_dict[y_col] = learn_df[y_col].min()
            #         max_y_dict[y_col] = learn_df[y_col].max() - min_y_dict[y_col]
            #         normalized_y_dict[y_col] = (learn_df[y_col] - min_y_dict[y_col]) / max_y_dict[y_col]

            #     print(normalized_learn_df_x)
            #     print('Данные нормированы, признаки отобраны')
            #     sg.popup('Данные нормированы, признаки отобраны')

            
        if createcorr_button:
            normalized_learn_df_x = st.session_state.corr_df_x
            min_y_dict = st.session_state.corr_min
            max_y_dict = st.session_state.corr_max
            normalized_y_dict = st.session_state.corr_normalized_y_dict

            if normalized_learn_df_x is None:
                st.error('Нормируйте данные')

            else:    
            #     # Отделяем Х
                 X = np.asarray(normalized_learn_df_x).astype(float)
                 input_dim = len(X_COLS)
            #     Y_dict = {}
                 model_dict = {}
            #     history_dict = {}

                 for y_col in Y_COLS:
                     print(f"Корреляционная модель для {y_col}:")
                     model_dict[y_col] = LinearCorrModel().fit(X, normalized_y_dict[y_col])
                     r_sq = model_dict[y_col].score(X, normalized_y_dict[y_col])
                     model_dicts_corr(y_col,model_dict[y_col])
                     print('R^2', r_sq)
                     print("-" * 30)

                 print("Все корреляционные модели построены")
                 message_cons_corr("Все корреляционные модели построены")

        if save_excell:
            model_dict = {}
            normalized_learn_df_x = st.session_state.corr_df_x
            min_y_dict = st.session_state.corr_min
            max_y_dict = st.session_state.corr_max
            normalized_y_dict = st.session_state.corr_normalized_y_dict
            for y_col in Y_COLS:
                model_dict[y_col] = st.session_state.model_dicts_corr[y_col]

            if not model_dict:
                st.error('Постройте модель')
            else:
                for y_col in Y_COLS:
                    pred = model_dict[y_col].predict(normalized_learn_df_x).flatten()
                    #Возвращем данные к нашей размерности - обратное нормирование
                    learn_df[y_col] = pred * max_y_dict[y_col] + min_y_dict[y_col]
                res_file = "result_corr.xlsx"
                learn_df[X_COLS + Y_COLS].to_excel(res_file)
                print(f"Результат сохранен в {res_file}")
                message_cons_corr(f"Результат сохранен в {res_file}")

        if corr_ur_button:
            model_dict = {}
            normalized_learn_df_x = st.session_state.corr_df_x
            min_y_dict = st.session_state.corr_min
            max_y_dict = st.session_state.corr_max
            normalized_y_dict = st.session_state.corr_normalized_y_dict
            for y_col in Y_COLS:
                model_dict[y_col] = st.session_state.model_dicts_corr[y_col]

            if not model_dict:
                st.error('Постройте модель')
            else:
                for y_col in Y_COLS:
                    print(f"{y_col} = ", end='')
                    for i, coef in enumerate(model_dict[y_col].coef_):
                        if i != 0 and coef >= 0:
                            print('+', end='')
                        print(f"{coef}*{X_COLS[i]}", end='')
                    print()
                    print("-" * 30)
                message_cons_corr("Уравнения отображены")

        if predict_corr_button :
            test_size = int(test)
            model_dict = {}
            normalized_learn_df_x = st.session_state.corr_df_x
            min_y_dict = st.session_state.corr_min
            max_y_dict = st.session_state.corr_max
            normalized_y_dict = st.session_state.corr_normalized_y_dict
            for y_col in Y_COLS:
                model_dict[y_col] = st.session_state.model_dicts_corr[y_col]

            if not model_dict:
                st.error('Постройте модель')
            else:    
                try:
                    size = int(size)
                except:
                    size = None
                new_y_cols = []
                for y_col in Y_COLS:
                    if values.get(f"_{y_col}"):
                        new_y_cols.append(y_col)
                corr_data_plot(test_size, Y_COLS, model_dict, normalized_learn_df_x, max_y_dict, min_y_dict,
                                learn_df,
                                verbous=True)
        if save_corr_button:
            model_dict = {}
            normalized_learn_df_x = st.session_state.corr_df_x
            min_y_dict = st.session_state.corr_min
            max_y_dict = st.session_state.corr_max
            normalized_y_dict = st.session_state.corr_normalized_y_dict
            for y_col in Y_COLS:
                model_dict[y_col] = st.session_state.model_dicts_corr[y_col]

            if not model_dict:
                st.error('Постройте модель')
            else:
                corr_data_plot(test_size, Y_COLS, model_dict, normalized_learn_df_x, max_y_dict, min_y_dict,learn_df,
                            verbous=False)
                message_cons_corr("Графики сохранены в png файлы")

        if corr_matrix_button:
            cols = learn_df.columns.tolist()
            corr_cols = []
            for col in cols:
                if col.startswith('X') or col.startswith('Y'):
                    corr_cols.append((col))

            corr_matrix = learn_df.loc[:, corr_cols].corr()
            print(corr_matrix)

            f, ax = plt.subplots(figsize=(10, 10))
            sn.heatmap(corr_matrix, ax=ax, cmap="YlGnBu", linewidths=0.1, annot=True, annot_kws={"size": 6})
            plt.title("Корреляционная матрица")
            #plt.show()
            st.pyplot()
            
        st.text_area("console",str(st.session_state.message_cons_corr))

    
def main():

    if st.session_state.userser:
            main_window(st.session_state.userser)
    else:
        mode = st.sidebar.radio("Выберите режим:", ("Авторизация", "Регистрация"))
        if mode == "Авторизация":
            login()
        elif mode == "Регистрация":
            register()
    
    

if __name__ == "__main__":
    st.write(st.session_state)
    main()

    
    