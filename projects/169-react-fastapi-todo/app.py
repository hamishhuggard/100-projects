from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(
    title="FastAPI TODO API",
    description="A simple TODO app",
    version="1.0.0"
)

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

class TodoItem(BaseModel):
    id: int
    text: str
    completed: bool

class TodoCreate(BaseModel):
    text: str

class TodoUpdate(BaseModel):
    text: Optional[str] = None
    completed: Optional[bool] = None

todos_db: List[TodoItem] = [
    TodoItem(id=1, text="Learn FastAPI", completed=False),
    TodoItem(id=2, text="Integrate React", completed=True),
    TodoItem(id=3, text="Deploy with Docker", completed=False),
]
nextId = 4

@app.get('/api/todos', response_model=List[TodoItem], summary="Get all todo items")
async def get_all_todos():
    return todos_db

@app.post('/api/todos', response_model=TodoItem, status_code=201, summary="Create todo item")
async def create_todo(todo: TodoCreaete):
    global next_id
    new_todo = TodoItem(id=next_id, text=todo.text, completed=False)
    todos_db.append(new_todo)
    next_id += 1
    return new_todo

@app.put("/api/todos/{todo_id}", response_model=TodoItem, summary="Update an existing item")
async def update_todo(todo_id: int, todo_update: TodoUpdate):
    for item in todos_db:
        if item.id == todo_id:
            if todo_update.text != None:
                item.text = todo_update.text
            if todo_update.completed != None:
                item.completed = todo_update.completed
            return todo
    raise HTTPException(status_code=404, detail="Todo item not found")

@app.delete("/api/todos/{todo_id}", status_code=204, summary="Delete todo item")
async def delete_todo(todo_id: int):
    for i, item in enumerate(todos_db):
        if item.id == todo_id:
            item = todos_db.pop(i)
            return {"message": "item deleted"}
    raise HTTPException(status_code=404, detail="Todo item not found")


