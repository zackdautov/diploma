"""
Database class
"""

import sqlite3
import pandas as pd
from settings import DATA_SQL_TABLE, DB_NAME


class Database:
    def __init__(self, host=None, user=None, passwd=None, database=None, port=None):
        self.database = sqlite3.connect(DB_NAME)
        self.cursor = self.database.cursor()
        self.cursor.execute(
            '''CREATE TABLE IF NOT EXISTS logins (Name text PRIMARY KEY, Surname text, Login text, Password text, Role text)''')
        self.database.commit()

    def drop_table(self, table):
        self.cursor.execute(f"DROP TABLE IF EXISTS {table}")
        self.database.commit()

    def insert_user(self, name, surname, login, password, role):
        self.cursor.execute('''INSERT OR IGNORE INTO logins (Name,Surname,Login,Password,Role) values (?,?,?,?,?)''',
                       (name, surname, login, password, role))
        self.database.commit()

    def get_user(self, login):
        data = self.cursor.execute('''SELECT * FROM logins WHERE Login = ?''', (login,))
        return data.fetchone()

    def save_data(self, df):
        df.to_sql(name=DATA_SQL_TABLE, con=self.database, index=False)

    def read_data(self):
        df = pd.read_sql(f'select * from {DATA_SQL_TABLE}', self.database)
        return df

    def check_table(self, table):
        res = self.cursor.execute('''SELECT count(*) FROM sqlite_master WHERE type='table' AND name=?;''', (table,))
        return res.fetchone()[0]

    def close(self):
        self.cursor.close()
        self.database.close()
