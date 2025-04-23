import sqlite3
from langchain.tools import Tool
from pydantic import BaseModel
from typing import List
conn = sqlite3.connect('db.sqlite')


def list_tables():
    """Return a list of table names in the SQLite database at *db_path*."""
    cursor = conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type='table'          -- only real tables
            AND name NOT LIKE 'sqlite_%'  -- skip SQLite’s internal tables
        ORDER BY name;
        """
    )
    return [row[0] for row in cursor.fetchall()]

def run_sqlite_query(query):
    c = conn.cursor()
    try:
        c.execute(query)
        return c.fetchall()
    except sqlite3.OperationalError as err:
        return f'The following error has occured: {str(err)}'

class RunQueryArgsSchema(BaseModel):
    query: str

run_query_tool = Tool.from_function(
    name='run_sqlite_query',
    description='Run a sqlite query.',
    func=run_sqlite_query,
    args_schema=RunQueryArgsSchema,
)

def describe_table(table_name):
    c = conn.cursor()
    # table_names_with_quotes = [f"\'{table}\'" for table in table_names]
    # tables = ', '.join(table_names_with_quotes)
    sql_cmd = f"SELECT sql from sqlite_master WHERE type = 'table' and name = '{table_name}';"
    # print(sql_cmd)
    rows = c.execute(sql_cmd)
    return '\n'.join([row[0] for row in rows if row[0] is not None])

class DescribeTableArgsSchema(BaseModel):
    table_name: str
    
describe_table_tool = Tool.from_function(
    name='describe_tables_tool',
    description='Given a table name, returns the schema of the table with that name',
    func=describe_table,
    args_schema=DescribeTableArgsSchema,
)

if __name__ == "__main__":
    # for tbl in list_tables():
    #     print(tbl)
    # tables = ', '.join([f"\'{table}\'" for table in list_tables()])
    # print(f"SELECT sql from sqlite_master WHERE type = 'table' and name IN ({tables});")
    print(describe_table('addresses'))