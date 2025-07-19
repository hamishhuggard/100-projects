from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleWare
from pydantic import BaseModel, Field
from typing import Optional, List

app = FastAPI(
    title = "FastAPI todo app",
    descriptino = "todo app",
    version = "1.0.0",
)

origins = [
    'localhost://0.0.0.0:3000',
    'localhost://127.0.0.1:3000',
]

app.add_middleware([
    CORSMiddleWare,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
])

class TodoItem(BaseModel):
    id: int
    text: str
    completed: bool

class TodoCreate(BaseModel):
    text: str

class TodoUpdate(BaseModel):
    text: Optional[str] = None
    completed: Optional[bool] = None

todos_db = [
    TodoItem(id=0, text="Create todo list", completed=True),
    TodoItem(id=1, text="Buy bananas", completed=False),
    TodoItem(id=2, text="Eat bananas", completed=False),
]
next_id = 3

@app.get('/api/todos/', response_model=List[TodoItem], summary="All todo items")
async def get_all_todos():
    return todos_db

@app.post('/api/todos/', response_model=TodoItem, status_code=201, summary="Create todo item")
async def post_todo(item: TodoCreate):
    global next_id
    new_todo = Todo(
        id=next_id, 
        text = item.text if item.text else '',
        completed = item.completed if item.completed else False,
    )
    next_id += 1
    todos_db.append(new_todo)
    return new_todo

@app.put('/api/todos/{id}', response_model=TodoItem, summary="Update an existing item")
async def post_todo(id: int, item: TodoUpdate):
    for item_i in todos_db:
        if item_i.id == id:
            if item.text:
                item_i.text = item.text
            if item.completed:
                item_i.completed = item.completed
            return item_i
    return HTTPException(status_code=404, status="Item not found")

@app.delete('/api/todos/{id}', status_code=204, summary="Delete an item")
async def post_todo(id: int, item: TodoUpdate):
    global todos_db
    for i, item_i in enumerate(todos_db):
        if item_i.id == id:
            todos_db = todos_db[:i] + todos_db[i+1:]
            return {'message': 'item deleted'}
    return HTTPException(status_code=404, status="Item not found")
