import pandas as pd
import sqlite3
import datetime

# Load data from the Excel file
excel_file_path = 'par.xlsx'
data = pd.read_excel(excel_file_path)

# Identify and convert date columns
# Assuming the column containing date information is named 'date'
if 'date' in data.columns:
    def convert_mixed_dates(value):
        if isinstance(value, (int, float)):  # Likely an Excel serial number
            return datetime.datetime(1899, 12, 30) + datetime.timedelta(days=value)
        else:
            return pd.to_datetime(value, errors='coerce')  # Attempt to convert any valid string format

    # Apply conversion and format it to 'YYYY-MM-DD'
    data['date'] = data['date'].apply(convert_mixed_dates).dt.strftime('%m-%d-%Y')

# Connect to SQLite (or create a new database)
conn = sqlite3.connect('chat.db')
cursor = conn.cursor()

# Define table name and columns based on the Excel file columns
table_name = 'par_table'
columns = [col for col in data.columns if col.lower() != 'id']

# Create table with auto-incrementing primary key
create_table_query = f'''
CREATE TABLE IF NOT EXISTS {table_name} (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    {' TEXT, '.join(columns)} TEXT
);
'''

cursor.execute(create_table_query)

# Insert data from DataFrame into the SQLite table
for _, row in data.iterrows():
    insert_query = f'''
    INSERT INTO {table_name} ({', '.join(columns)})
    VALUES ({', '.join(['?' for _ in columns])})
    '''
    cursor.execute(insert_query, tuple(row[columns]))

# Commit changes
conn.commit()

# Query to view the data in the table
select_query = f'SELECT * FROM {table_name}'
cursor.execute(select_query)

# Fetch all rows and display them
rows = cursor.fetchall()
for row in rows:
    print(row)

# Close the connection
conn.close()

print("Data successfully inserted and retrieved from SQLite database.")
