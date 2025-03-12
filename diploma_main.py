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

if 'clicks' not in st.session_state:
    st.session_state['clicks'] = {}

def click(key):
    st.session_state.clicks[key] = True

def unclick(key):
    st.session_state.clicks[key] = False

if 'dataex' not in st.session_state:
    st.session_state['dataex'] = {}

if 'userser' not in st.session_state:
    st.session_state['userser'] = {}

if 'df_flag' not in st.session_state:
    st.session_state['df_flag'] = {}

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
    uploaded_file = st.file_uploader("Выберите файл", type=[ 'xlsx'])
    st.text('Загрузить данные для нейросети, регрессии и корреляции:')
    uploaded_file1 = st.file_uploader("Выберите файл", type=['txt', 'csv', 'xlsx'])
    load_button = st.button("загрузить", key="button_load") 
    col1, col2, col3= st.columns(3)
    with col1:
        savedata_button = st.button("Сохранить данные в БД", key="button_savedata") 
    with col2:
        loaddata_button = st.button("Загрузить данные из БД", key="button_loaddata") 
    with col3:
        cleardata_button = st.button("Очистить данные в БД", key="button_cleardata") 


    df_flag = False
    if load_button:
        if uploaded_file is not None:
            file_path = uploaded_file
            excel_data_df = pd.read_excel(uploaded_file)
            dataex(excel_data_df)
            st.write(st.session_state.dataex)
            st.session_state.df_flag = True

    if savedata_button:
        if not st.session_state.df_flag:
            st.error('Нет данных для сохранения')
        database.drop_table(DATA_SQL_TABLE)
        database.save_data(st.session_state.dataex)
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
    
    col4, col5, col6= st.columns(3)
    with col4:
        statdata_button = st.button("Статистика", args=[{"style":"white-space: nowrap;"}], key="button_statdata") 
    with col5:
        normdata_button = st.button("Нормальность",args=[{"style": "white-space: nowrap;"}], key="button_normdata") 
    with col6:
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
    main()
    st.write(st.session_state)
    