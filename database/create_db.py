import sqlite3

# Step 1: Connect to SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect('chat.db')

# Step 2: Create a cursor object to execute SQL commands
cursor = conn.cursor()

# Step 3: Create a table (if not exists)
cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        age INTEGER,
        email TEXT,
        available_leave_day INTEGER,
        used_leave_day INTEGER
    )
''')
print("Table created successfully.")

# Step 4: Insert data into the table
cursor.execute("INSERT INTO users (id,name, age, email, available_leave_day, used_leave_day) VALUES (?, ?, ?, ?, ?, ?)", (31300,"Hàn Phi Quang", 52, "khang.tv@ots.vn",100,200))
cursor.execute("INSERT INTO users (id,name, age, email, available_leave_day, used_leave_day) VALUES (?, ?, ?, ?, ?, ?)", (1995,"Nguyễn Xuân Thọ", 25, "tho.nx@ots.vn",99,0))
cursor.execute("INSERT INTO users (id,name, age, email, available_leave_day, used_leave_day) VALUES (?, ?, ?, ?, ?, ?)", (2000,"Bùi Tấn Đạt", 18, "dat.bt@ots.vn",0,99))
print("Data inserted successfully.")

# Commit the changes
conn.commit()

# Step 5: Query data from the table
cursor.execute("SELECT * FROM users")
rows = cursor.fetchall()\

# Display query results
print("Data in users table:")
for row in rows:
    print(row)

# Step 6: Close the connection
conn.close()
