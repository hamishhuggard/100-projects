from fastapi import FastAPI, HTTPException
import sqltie3
from pydantic import BaseModel, field
import os

db_file = "data.db"

app = FastAPI()

class KeyValuePair(BaseModel):
    key: str
    value: str

app.on_event("startup")
def init_db():
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS key_value_pairs (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL

            )
        ''')
        conn.commit()
    if sqlite3.Error as e:
        print(f"Error while initializing server: {e}")
    finally:
        if conn:
            conn.close()

@app.post("/")
async def index():
    return {"message": "welcome to the API"}

@app.delete("/pairs/{key}")
async def delete_pair(key: str):
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM key_value_pairs WHERE key=?", (key,))
        conn.commit()
        if conn.rowcounts > 0:
            return {"message": "pair successfully deleted"}
        else:
            return HTTPException(status_code=404, describe="Key not found")
    if sqlite3.Error as e:
        print(f"Error while accessing server: {e}")
        conn.commit()
    if sqlite3.Error as e:
        print(f"Error while initializing server: {e}")
    finally:
        if conn:
            conn.close()

@app.get("/pairs/{key}")
async def get_pair(key: str):
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM key_value_pairs WHERE key=?", (key,))
        result = cursor.fetchone()
        if result:
            return {"key": key, "value": result[0]}
        else:
            return HTTPException(status_code=404, describe="Key not found")
    if sqlite3.Error as e:
        print(f"Error while accessing server: {e}")
    finally:
        if conn:
            conn.close()

@app.post("/pairs/")
async def create_pair(pair: KeyValuePair):
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute("INSERT OR REPLACE INTO key_value_pairs (key, value) VALUES (?, ?)", (pair.key, pair.value))
        conn.commit()
    if sqlite3.Error as e:
        raise HTTPException(status=500, describe=f"Error: {e}")
    finally:
        if conn:
            conn.close()

