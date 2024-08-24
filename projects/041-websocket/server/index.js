const WebSocket = require('ws');
const server = new WebSocket.server({ port: '8080' });

server.on('connection', socket => {
    socket.on('message', message => {
        socket.send(`Roger that ${message}`);
    }
});
