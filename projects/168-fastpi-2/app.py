from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sqlite3
import os

app = FastAPI()

DB_FILE = "key_value_store.db"

class KeyValuePair(BaseModel):
    key: str
    value: str


