import sqlite3 as sq3

def checkIfTableExists(cursor,tableName):
    try:
        cursor.execute(f"SELECT * FROM {tableName}")
        return True   
            
    except sq3.OperationalError:
        return False

data = [
    ('rishabh',21,'F4'),
    ('paul',21,'F4'),
    ('yash',21,'F5')
]

def print__data(x):
    for row in x.execute("SELECT * FROM student ORDER BY batch"):
        print(row)



db = sq3.connect('attendance__record.db')
db__handler = db.cursor()

if(checkIfTableExists(db__handler,'student')==False):
    db__handler.execute('CREATE TABLE student(name,age,batch)')
else:
    db__handler.execute('DELETE FROM student')

db__handler.executemany("INSERT INTO student VALUES (?,?,?)",data)
db.commit()
print__data(db__handler)