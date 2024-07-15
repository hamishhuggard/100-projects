const express = require('express');
const session = require('express-session');
const passport = require('passport');
require('dotenv').config();

const app = express();
app.use(express.json());
app.use(session({ secret: 'secret', resave: false, saveUninitialized: true }));
app.use(passport.initialize());
app.use(passport.session());

app.listen(3000, () => console.log('server started on port 3000'));
