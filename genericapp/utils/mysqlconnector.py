# pip install streamlit mysql-connector-python pandas python-dotenv groq
import mysql.connector
import pandas as pd

class MySqlConnector:
    """
    This class is used to connect to the MySQL database and Execute query and showing results
    """
    def __init__(self, host, user, password, database, port=3306):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.port = port
        self.columns_info = None
        self.tables = None

    def get_connection(self):
        """
        This method is used to connect to the MySQL database and return connection object
        """
        try:
            conn = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database
                port = self.port
            )
            return conn
        except mysql.connector.Error as err:
            return None

    def execute_pd_query(self, query):
        """
        This method is used to execute query and return results as pandas DataFrame
        """
        try:
            conn = self.get_connection()
            results = pd.read_sql_query(query, conn)
            return results
        except mysql.connector.Error as err:
            return err
        finally:
            conn.close()

    def execute_sql_query(self, query):
        """
        This method executes a SQL query and returns raw results
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute(query)
            result = cursor.fetchall()
            return result
        except mysql.connector.Error as err:
            return err
        finally:
            cursor.close()
            conn.close()

    def get_columns_info(self, table_name):
        """
        This method is used to get the columns info for a specific table
        """
        try:
            query = f"DESCRIBE {table_name}"
            result = self.execute_sql_query(query)
            return result
        except mysql.connector.Error as err:
            return err

    def get_basic_info(self):
        """
        This method will get all tables and all columns info from those tables
        """
        # Get all tables in public schema
        self.tables = self.execute_sql_query("SHOW TABLES")
        columns_info = []
        for table in self.tables:
            table_info = self.get_columns_info(table[0])
            columns_info.append({
                "table": table[0],
                "columns": table_info
            })
        self.columns_info = columns_info
        return self.tables, self.columns_info