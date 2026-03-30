import sqlite3
import pandas as pd

conn = sqlite3.connect("smart_trolley.db")

df = pd.read_sql_query(
    "SELECT product_name, category, sub_category FROM products",
    conn
)

conn.close()

print(df.head(30))