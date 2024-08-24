const express = require('express');
const app = express();
const PORT = 3000;

app.get('/', (req, res) => {
    res.send('hello world');
});

app.get('/about', (req, res) => {
    res.send('about page');
});

app.listen(PORT, () => {
    console.log(`server running on http://localhost:${PORT}`)
});
