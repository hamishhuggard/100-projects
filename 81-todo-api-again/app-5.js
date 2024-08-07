const express = require('express');
const bodyParser = require('body-parser');
const app = express();
const port = 3000;

todos = []

app.use(bodyParser);

app.get('/todos', (req, res) => {
    res.json(todos);
});

app.post('/todos', (req, res) => {
    const { text } = req.body;
    const todo = { id: todos.length+1, text };
    todos.push(todo);
    res.status(201).json(todo);
});

app.put('/todos/:id', (req, res) => {
    const { id } = req.params;
    const { text } = req.body;
    const index = todos.findIndex(todo => todo.id === id);
    if (index===-1) return res.status(404).send('todo not found');
    todos[index].text = text;
    res.json(todos[index]);
});

app.delete('/todos/:id', (req, res) => {
    const { id } = req.params;
    todos = todos.filter(todo => todo.id !== id);
    res.status(204).send();
});

app.listen(port, () => {
    console.log(`listening on port ${port}`);
});
