# pip install streamlit psycopg2-binary pandas python-dotenv groq
import psycopg2
import pandas as pd
from psycopg2 import sql

class PostgreSQLConnector:
    """
    This class is used to connect to the PostgreSQL database, execute queries, and retrieve schema information.
    """
    def __init__(self, host, user, password, database, port=5432):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.port = port
        self.columns_info = None
        self.tables = None

    def get_connection(self):
        """
        Establishes a connection to the PostgreSQL database. Returns connection object or None if failed.
        """
        try:
            conn = psycopg2.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database,
                port=self.port
            )
            return conn
        except psycopg2.Error as err:
            print(f"Database connection error: {err}")
            return None

    def execute_pd_query(self, query):
        """
        Executes a SQL query and returns results as a pandas DataFrame.
        """
        conn = self.get_connection()
        if not conn:
            return pd.DataFrame()  # Return empty DataFrame on failure
        try:
            results = pd.read_sql_query(query, conn)
            return results
        except Exception as err:
            print(f"Query execution error: {err}")
            return pd.DataFrame()
        finally:
            conn.close()

    def execute_sql_query(self, query):
        """
        Executes a raw SQL query and returns the result as a list of tuples.
        """
        conn = self.get_connection()
        if not conn:
            return []
        try:
            with conn.cursor() as cursor:
                cursor.execute(query)
                result = cursor.fetchall()
                return result
        except Exception as err:
            print(f"SQL execution error: {err}")
            return []
        finally:
            conn.close()

    def get_columns_info(self, table_name):
        """
        Retrieves column details (name, type, nullable, etc.) for a given table. Uses PostgreSQL's information_schema.
        """
        query = f"""
        SELECT 
            column_name,
            data_type,
            is_nullable,
            column_default
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = %s
        ORDER BY ordinal_position;
        """
        conn = self.get_connection()
        if not conn:
            return []
        try:
            with conn.cursor() as cursor:
                cursor.execute(query, (table_name,))
                result = cursor.fetchall()
                return result
        except Exception as err:
            print(f"Error fetching column info: {err}")
            return []
        finally:
            conn.close()

    def get_basic_info(self):
        """
        Fetches all table names in the 'public' schema and their column details. Returns a list of tables and a structured column info list.
        """
        # Get all tables in public schema
        tables_query = """
        SELECT tablename 
        FROM pg_tables 
        WHERE schemaname = 'public';
        """
        self.tables = self.execute_sql_query(tables_query)
        self.tables = [row[0] for row in self.tables]  # Extract table names
        columns_info = []
        for table in self.tables:
            table_info = self.get_columns_info(table)
            columns_info.append({
                "table": table,
                "columns": [
                    {
                        "column_name": col[0],
                        "data_type": col[1],
                        "is_nullable": col[2],
                        "default_value": col[3]
                    }
                    for col in table_info
                ]
            })
        self.columns_info = columns_info
        return self.tables, self.columns_info