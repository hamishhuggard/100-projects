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
