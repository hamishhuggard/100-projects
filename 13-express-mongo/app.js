const express = require('express');
const mongoose = require('mongoose');

const app = express()
app.use(express.json());

const PORT = processes.env.PORT || 3000;

mongoose.connect('mongodb://localhost:27017/mydatabase', {
    useNewUrlParse: true,
    useUnifiedTopology: true
}).then(() => console.log('MongoDB connected'))
    .catch(err => console.log(err));

const UserSchema = new mongoose.Schema({
    name: String,
    age: Number
});

const User = mongoose.model("User", UserSchema);

app.post('/users', (req, res) => {
    const newUser = new User(req.body);
    newUser.save()
        .then(user => res.status(201).send(user)
        .catch(err => res.status(400).send(err);
});

app.listen(PORT, () => {
    console.log(`server is running at ${PORT}`);
})
