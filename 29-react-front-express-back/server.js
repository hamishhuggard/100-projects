const express = require('express');
const app = express()
const PORT = 5000;

app.use(express.json());

app.post('/square', (req, res) => {
    const { num } = req.body;
    if (typeof num === 'number') {
        const result = num * num;
        res.json({ result });
    } else {
        res.status(400).json({ error: "Input must be a number" });
    }
});

app.listen(PORT, () => {
    console.log(`server running on http://localhost:${PORT}`);
});
