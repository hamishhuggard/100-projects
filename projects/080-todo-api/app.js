const express = require('express');
const bodyParser = require('body-parser');
const app = express();
const port = 3000;

let todos = [];

app.use(bodyParser.json());

app.get('/todos', (req, res) => {
    res.json(todos);
});

app.post('/todos', (req, res) => {
    const { text } = req.body;
    const todo = { id: todos.length + 1, text };
    todos.push(todo);
    res.status(201).json(todo);
});

app.put('/todos/:id', (req, res) => {
    const { id } = req.params;
    const { text } = req.body;
    const index = todos.findIndex(todo => todo.id === parseInt(id));
    if (index === -1) return res.status(404).send('Todo not found');
    todos[index].text = text;
    res.json(todos[index]);
});

app.delete('/todos/:id', (req, res) => {
    const { id } = req.params;
    todos = todos.filter(todo => todo.id !== parseInt(id));
    res.status(204).send();
});

app.listen(port, () => {
    console.log(`Server running on port ${port}`)
});

// https://chatgpt.com/c/c8705b74-3191-4550-812e-208c21d387f9

