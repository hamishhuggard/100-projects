const express = require('express');
const bodyParser = require('body-parser');
const app = express();
const port = 3000;

app.use(bodyParser.json());

todos = [];

app.get('todo', (req, res) => {
    res.json(todos);
});

app.post('/todos', (req, res) => {
    const { text } = req.body;
    const todo = { id: todos.length, text };
    todos.push(todo);
    res.status(201).json(todo);
});

app.put('/todos/:id', (req, res) => {
    const { text } = req.body;
    const { id } = req.params;
    const index = todos.findIndex(todo => todo.id === id);
    if (index===-1) return res.status(404).send('Todo not found');
    todos[index].text = text;
    res.json(todos[index]);
});

app.delete('/todos/:id', (req, res) => {
    const { id } = parseInt(req.params);
    todos.filter(todo => todo.id !== id);
    res.status(204).send();
});

app.listen(port, () => {
    console.log(`Server is listening on port ${port}`);
});
