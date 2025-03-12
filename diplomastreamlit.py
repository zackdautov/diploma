import yaml
import streamlit as st
import streamlit_authenticator as stauth 
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


# Окно логина
def login_window():
    st.title("Авторизация")
    login = st.text_input("Введите имя пользователя")
    pasw = st.text_input("Введите пароль", type="password")
    login_button = st.button("Войти", key="button_login") 
    register_button = st.button("Регистрация", key="button_register") 

    database = Database()
    if login_button:
            
        if len(login) == 0 or len(pasw) == 0:
            st.error('Поля логина и пароля не должны быть пустыми')
        user = database.get_user(login)
        if not user:
            st.error('Пользователя не существует')
        if pasw == user[3]:
            database.close()
            return(user)
        else:
         st.error('Неверный пароль')

    if register_button:
        st.session_state['show_register_form'] = True
        

def register_window(database):
    st.title("Регистрация")
    st.text('Поля * обязательны для заполнения')

    login = st.text_input("Логин*")
    paswrd = st.text_input("Пароль*", type="password")
    name = st.text_input("Имя*")
    surname = st.text_input("Фамилия")
    role = st.text_input("Права*(Гость/Админ)")
    zaregister_button = st.button("Зарегистрироваться", key="button_zaregister") 


    if zaregister_button:
        pasw =  paswrd
        sname = surname    
        if len(login) == 0 or len(pasw) == 0 or len(name) == 0 or len(role) == 0:
            st.error('Поля * не должны быть пустыми')
        
        user = database.get_user(login)
        if user:
            st.error('Пользователь существует')
              
        else:
            database.insert_user(name, sname, login, pasw, role)       
                






if __name__ == "__main__":
    if DEBUG_MODE:
        main_window(DEFAULT_USER)
    else:
        user = login_window()
        if user:
            main_window(user)
    exit(0)
