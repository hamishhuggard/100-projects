const canvas = document.getElementById("gameCanvas");
const ctx = canvas.getContext("2d");

const headImg = new Image();
headImg.src = "head.png";

const bodyImg = new Image();
bodyImg.src = "square.png";

const foodImg = new Image();
foodImg.src = "food.png";

const tileSize = 40;
const width = canvas.width / tileSize;
const height = canvas.height / tileSize;

const directions = {
    ArrowUp: { x: 0, y: -1 },
    ArrowDown: { x: 0, y: 1 },
    ArrowLeft: { x: -1, y: 0 },
    ArrowRight: { x: 1, y: 0 }
};

let snake = {
    body: [{ x: Math.floor(width / 2), y: Math.floor(height / 2) }],
    direction: null,
    nextDirection: null
};

let food = { x: 0, y: 0 };
let gameOver = false;

document.addEventListener("keydown", startGame);

function startGame(event) {
    if (!snake.direction && directions[event.key]) {
        snake.direction = directions[event.key];
        snake.nextDirection = directions[event.key];
        document.removeEventListener("keydown", startGame);
        document.addEventListener("keydown", changeDirection);
        placeFood();
        gameLoop();
    }
}

function gameLoop() {
    if (gameOver) return;
    moveSnake();
    if (checkCollision()) {
        endGame();
        return;
    }
    if (checkFood()) {
        growSnake();
        placeFood();
    }
    clearCanvas();
    drawFood();
    drawSnake();
    setTimeout(gameLoop, 100);
}

function moveSnake() {
    const head = { ...snake.body[0] };
    head.x += snake.direction.x;
    head.y += snake.direction.y;
    snake.body.unshift(head);
    snake.body.pop();
    if (snake.nextDirection && !isOppositeDirection(snake.direction, snake.nextDirection)) {
        snake.direction = snake.nextDirection;
    }
}

function changeDirection(event) {
    if (directions[event.key] && !isOppositeDirection(snake.direction, directions[event.key])) {
        snake.nextDirection = directions[event.key];
    }
}

function isOppositeDirection(dir1, dir2) {
    return dir1.x + dir2.x === 0 && dir1.y + dir2.y === 0;
}

function checkCollision() {
    const head = snake.body[0];
    if (head.x < 0 || head.y < 0 || head.x >= width || head.y >= height) {
        return true;
    }
    for (let i = 1; i < snake.body.length; i++) {
        if (head.x === snake.body[i].x && head.y === snake.body[i].y) {
            return true;
        }
    }
    return false;
}

function checkFood() {
    const head = snake.body[0];
    return head.x === food.x && head.y === food.y;
}

function growSnake() {
    const tail = { ...snake.body[snake.body.length - 1] };
    snake.body.push(tail);
}

function placeFood() {
    let newFood;
    do {
        newFood = { x: Math.floor(Math.random() * width), y: Math.floor(Math.random() * height) };
    } while (snake.body.some(segment => segment.x === newFood.x && segment.y === newFood.y));
    food = newFood;
}

function clearCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function drawSnake() {
    snake.body.forEach((segment, index) => {
        ctx.save(); // Save the current context state
        if (index === 0) {
            const angle = getRotationAngle(snake.direction);
            ctx.translate(segment.x * tileSize + tileSize / 2, segment.y * tileSize + tileSize / 2);
            ctx.rotate(angle);
            ctx.drawImage(headImg, -tileSize / 2, -tileSize / 2, tileSize, tileSize);
        } else {
            ctx.drawImage(bodyImg, segment.x * tileSize, segment.y * tileSize, tileSize, tileSize);
        }
        ctx.restore(); // Restore the context state
    });
}

function getRotationAngle(direction) {
    if (direction.x === 1) return 0; // Right
    if (direction.x === -1) return Math.PI; // Left
    if (direction.y === 1) return Math.PI / 2; // Down
    if (direction.y === -1) return -Math.PI / 2; // Up
    return 0;
}

function drawFood() {
    ctx.drawImage(foodImg, food.x * tileSize, food.y * tileSize, tileSize, tileSize);
}

function endGame() {
    gameOver = true;
    document.getElementById("gameOver").style.display = "block";
    document.getElementById("playAgain").addEventListener("click", resetGame);
}

function resetGame() {
    snake = {
        body: [{ x: Math.floor(width / 2), y: Math.floor(height / 2) }],
        direction: null,
        nextDirection: null
    };
    gameOver = false;
    document.getElementById("gameOver").style.display = "none";
    document.addEventListener("keydown", startGame);
    clearCanvas();
    drawSnake();
}
