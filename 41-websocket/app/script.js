const socket = new WebSocket('ws://localhost:8080');
socket.onmessage = ({ data }) => {
    console.log(`message from server: ${data}`);
}
document.querySelector('button').onClick = () => {
    socket.send('hello');
};
