from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sqlite3
import os

app = FastAPI()

DB_FILE = "key_value_store.db"

class KeyValuePair(BaseModel):
    key: str
    value: str

@app.on_event('startup')
async def init_db():
    conn = None
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS key_value_pairs (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        conn.commit()
    except sqlite3.Error as e:
        print(f"Error while setting up database: {e}")
    finally:
        if conn:
            conn.close()

@app.get('/')
async def home():
    return {"message": "Welcome to the key/value store"}

@app.post('/items')
async def post_item(item: KeyValuePair):
    conn = None
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("INSERT OR REPLACE INTO key_value_store (key, value) VALUES (?,?)", (item.key, item.value))
        conn.commit()
        return {"message": f"Item {item.key}: {item.value} succesfully created"}
    except sqlite3.Error e:
        raise HTTPException(status=500, f"Error: {e}")
    finally:
        if conn:
            conn.close()

@app.get('/items/{key}')
async def get_item(key: str):
    conn = None
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM key_value_store WHERE key=?", (key,))
        result = cursor.fetchone()
        if result:
            return {"key": key, "value": result[0]}
        else:
            raise HTTPException(status=404, f"Key not found: {key}")
    except e:
        raise HTTPException(status=500, f"Error: {e}")
    finally:
        if conn:
            conn.close()

@app.delete('/items/{key}')
def delete_item(key: str):
    conn = None
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM key_value_store WHERE key=?", (key,))
        conn.commit()
        if cursor.rowcounts > 0:
            return {"message": f"Successfully deleted key {key}"}
        else:
            raise HTTPException(status=404, f"Key not found: {key}")
    except e:
        raise HTTPException(status=500, f"Error: {e}")
    finally:
        if conn:
            conn.close()
