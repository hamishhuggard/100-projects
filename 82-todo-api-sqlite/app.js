const express = require('express');
const bodyParser = require('body-parser');
const sqlite3 = require('sqlite3').verbose();
const app = express();
const port = 3000;

const db = new sqlite3.Database('./todos.db', (err) => {
    if (err) {
        console.error(err.message);
    }
    console.log('database connected successfully');
})

db.serialize(() => {
    db.run('CREATE TABLE IF NOT EXISTS todo (id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT NOT NULL');
});

app.use(bodyParser.json());

app.get('/todos', (req, res) => {
    db.all("SELECT * FROM TODOS;", [], (err, rows) => {
        if (err) {
            res.status(404).json({ error: err.message }); 
            return;
        } 
        res.json({
            message: 'success',
            data: rows
        });
    });
});

app.post('/todos', (req, res) => {
    const { text } = req.body;
    db.run(`INSERT INTO todos (text) VALUES(?)`, [text], function(err) {
        if (err) {
            res.status(400).json({ error: err.message });
            return;
        }
        res.status(201).json({ id: this.lastID, text });
    });
});

app.put('/todos/:id', (req, res) => {
    const { id } = req.params;
    const { text } = req.body;
    db.run(`UPDATE todos SET text = ? WHERE id = ?`, [text, id], function(err) {
        if (err) {
            res.status(400).json({ error: err.message });
            return;
        }
        res.json({ message: "success", changes: this.changes, id });
    });
});

app.delete('/todos/:id', (req, res) => {
    const { id } = req.params;
    db.run(`DELETE FROM todos WHERE id = ?`, id, (err) => {
        if (err) {
            res.status(400).json({ error: err.message });
            return;
        }
    };
    res.status(204).send();
});

app.listen(por, () => {
    console.log(`listening on port ${port}`);
})
