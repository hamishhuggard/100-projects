const earth = document.querySelector('.earth');
const sun = document.querySelector('.sun');
let vx = 0; // Initial velocity in x direction
let vy = 0; // Initial velocity in y direction
const G = 0.1; // Gravitational constant, adjusted for visualization
const massSun = 10000; // Mass of the sun, adjusted for visualization

function getDistance(x1, y1, x2, y2) {
    return Math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2);
}

function updateEarthPosition() {
    const sunRect = sun.getBoundingClientRect();
    const earthRect = earth.getBoundingClientRect();

    const sunX = sunRect.left + sunRect.width / 2;
    const sunY = sunRect.top + sunRect.height / 2;
    const earthX = earthRect.left + earthRect.width / 2;
    const earthY = earthRect.top + earthRect.height / 2;

    const distance = getDistance(sunX, sunY, earthX, earthY);
    const force = (G * massSun) / (distance * distance);

    const ax = (force * (sunX - earthX)) / distance;
    const ay = (force * (sunY - earthY)) / distance;

    vx += ax;
    vy += ay;

    const newEarthX = earthX + vx;
    const newEarthY = earthY + vy;

    earth.style.left = `${newEarthX - earthRect.width / 2}px`;
    earth.style.top = `${newEarthY - earthRect.height / 2}px`;

    requestAnimationFrame(updateEarthPosition);
}

// Set initial position of Earth
earth.style.left = `calc(50% + 100px)`; // Offset from the Sun
earth.style.top = `50%`;

updateEarthPosition();
