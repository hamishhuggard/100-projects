const canvas = document.getElementById('oscillatorCanvas');
const ctx = canvas.getContext('2d');

canvas.width = 800;
canvas.height = 600;

// Axes Labels
const xAxisLabel = 'x';
const yAxisLabel = 'Re(ψ)';
const zAxisLabel = 'Im(ψ)';

// Drawing the parabola (x-5)^2
function drawParabola() {
    ctx.beginPath();
    ctx.moveTo(0, canvas.height);
    for (let x = 0; x <= canvas.width; x++) {
        let y = Math.pow(x / 100 - 5, 2) * 40;  // scale factor for visibility
        ctx.lineTo(x, canvas.height - y);
    }
    ctx.strokeStyle = 'blue';
    ctx.stroke();
}

// Drawing axes
function drawAxes() {
    ctx.beginPath();
    // X-axis
    ctx.moveTo(50, canvas.height - 50);
    ctx.lineTo(canvas.width - 50, canvas.height - 50);
    ctx.strokeStyle = 'black';
    ctx.stroke();
    // Y-axis
    ctx.moveTo(50, canvas.height - 50);
    ctx.lineTo(50, 50);
    ctx.stroke();
}

// Drawing axes labels
function drawLabels() {
    ctx.font = '20px Arial';
    ctx.fillText(xAxisLabel, canvas.width - 40, canvas.height - 30);
    ctx.fillText(yAxisLabel, 10, 60);
    ctx.fillText(zAxisLabel, canvas.width / 2 - 30, 30);
}

// Drawing the real and imaginary parts of ψ
function drawWaveFunction() {
    ctx.beginPath();
    for (let x = 0; x <= canvas.width - 100; x++) {
        let realY = Math.sin(x / 20) * 40;
        let imagY = Math.cos(x / 20) * 40;
        ctx.lineTo(x + 50, canvas.height - 50 - realY);
    }
    ctx.strokeStyle = 'red';
    ctx.stroke();
}

// Main drawing function
function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawAxes();
    drawLabels();
    drawParabola();
    drawWaveFunction();
}

draw();
