const express = require('express');
const mysql = require('mysql');

const db = mysql.createConnection({
    host: 'localhost',
    username: 'admin',
    password: '',
    database: 'express_blog'
})
