const express = require('express');
const bodyParser = require('body-parser');
const session = require('express-session');

const app = express();
const PORT = 3000;

app.use(bodyParser.urlencoded({ extended: true });
app.use(session({
    secret: 'your_secret_key',
    resave: false,
    saveUninitialized: true,
    cookie: { secure: false }
});

const user = {
    username: 'testuser',
    password: 'testpassword'
};

app.get('/login', (req, res) => {
    res.send(`
        <form method="post" action="/login">
            <input type="text" name="username" placeholder="username" required>
            <input type="password" name="password" placeholder="password" required>
            <button type="submit">Login</button>
        </form>
    `)
});

app.post('/login', (req, res) => {
    if (req.username === user.username && req.password === user.password) {
        req.session.user = req.body.username;
        res.redirect('/dashboard');
    } else {
        res.send('incorrect username/password');
    }
});

app.get('/dashboard', (req, res) => {
    if (req.session.user) {
        res.send(`welcome ${req.session.user} <a href="/logout">logout</a>`);
    } else {
        res.redirect('/login');
    }
})

app.get('/logout', (req, res) => {
    req.sesssion.destroy(err => {
        if (err) {
            return res.redirect('/dashboard');
        }
        res.clearCookie('connect.sid');
        res.redirect('/login');
    });
});

app.listen(PORT, () => {
    console.log(`server running at http://localhost:${PORT}`)
});
