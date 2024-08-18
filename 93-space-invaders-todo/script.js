const rocket = document.getElementById('rocket');
let bullets = [];
let leftKeyPressed = false;
let rightKeyPressed = false;
const ROCKET_WIDTH = 20;

document.addEventListener('keydown', (e) => {
    switch(e.code) {
        case 'ArrowLeft':
            leftKeyPressed = true;
            break;
        case 'ArrowRight':
            rightKeyPressed = true;
            break;
        case 'Space':
            fireBullet();
            break;
    }
});

document.addEventListener('keyup', (e) => {
    switch(e.code) {
        case 'ArrowLeft':
            leftKeyPressed = false;
            break;
        case 'ArrowRight':
            rightKeyPressed = false;
            break;
    }
});

function fireBullet() {
    const bullet = document.createElement('div');
    bullet.className = 'bullet';
    bullet.style.left = `${rocket.offsetLeft + rocket.clientWidth / 2 }px`;
    bullet.style.bottom = '50px';
    document.getElementById('game').appendChild(bullet);
    bullets.push(bullet);
}

function mod(n, m) {
    return ((n % m) + m) % m;
}


function moveRocket() {
    let x = rocket.offsetLeft + (rightKeyPressed-leftKeyPressed)*10
    x = mod(x, window.innerWidth);
    //rocket.style.left = `${Math.max(0, rocket.offsetLeft - 10)}px`;
    //rocket.style.left = `${Math.min(window.innerWidth - rocket.clientWidth, rocket.offsetLeft + 10)}px`;
    rocket.style.left = `${x}px`;
}

function moveBullets() {
    bullets = bullets.filter(bullet => {
        bullet.style.bottom = `${parseInt(bullet.style.bottom) + 10}px`;
        if (parseInt(bullet.style.bottom) > window.innerHeight) {
            bullet.remove();
            return false;
        }
        return true;
    });
}

setInterval(moveBullets, 50);
setInterval(moveRocket, 50);
