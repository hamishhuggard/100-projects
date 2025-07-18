from fastapi import FastAPII, HTTPException
from pydantic import BaseModel
import sqlite3
import os

app = FastAPI()

DATABASE_FILE = "key_value_store.db"

class KeyValueItem(BaseModel):
    key: str
    value: str

def init_db():
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.curosr()
        cursor.execute("""
           CREATE TABLE IF NOT EXISTS key_value_pairs (
               key TEXT PRIMARY KEY,
               value TEXT NOT NULL
           )
       """)
    except sqlite3.Error as e:
        print(f"Databae error during initialization: {e}")
    finally:
        if conn:
            conn.close()

@app.on_event("startup")
async def startup_event():
    init_db()

# API endpoints

@app.get("/")
async def read_root():
    return {"message": "Welcome to FastAPI Key-Value Store!"}

@app.post("/items/")
async def create_or_update_item(item: KeyValueItem):
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute(
                "INSERT OR REPLACE INTO key_value_pairs (key, value) VALUES (?, ?)",
                (item.key, item.value)
        )
        conn.commit()
        return {"message": "Item stored successfully", "key": item.key, "value": item.value}
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    finally:
        if conn:
            conn.close()

@app.get("/item/{key}")
async def get_item(key: str):
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT value from key_value_pair WHERE key = ?", (key,))
        result = cursor.fetchone()
        if result:
            return {"key": key, "value": result[0]}
        else:
            raise HTTPException(status_code=404, detail="Key not found")
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    finally:
        if conn:
            conn.close()

@app.delete("/item/{key}")
async def get_item(key: str):
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM key_value_pair WHERE key = ?", (key,))
        conn.commit()
        if cursor.rowcounts > 0:
            return {"message": "Item deleted successfully", "key": key}
        else:
            raise HTTPException(status_code=404, detail="Key not found")
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    finally:
        if conn:
            conn.close()

