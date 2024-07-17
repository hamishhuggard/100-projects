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

// Local

const LocalStrategy = require('passport-local').Strategy;
const bcrypt = require('bcryptjs');

passport.use(new LocalStrategy({ usernameField: 'email' },
    async (email, password, done) => {
        const user = users.find(user => user.email === email);
        if (!user) return done(null, false, { message: 'No user with that email' });

        try {
            if (await bcrypt.compare(password, user.password)) {
                return done(null, user);
            } else {
                return done(null, false, { message: 'Password incorrect' });
            }
        } catch (e) {
            return done(e);
        }
    }
})
