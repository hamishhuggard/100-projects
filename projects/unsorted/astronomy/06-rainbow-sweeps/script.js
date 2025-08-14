const earth = document.querySelector('.earth');
const orbit = document.querySelector('.orbit');
const eccentricitySlider = document.getElementById('eccentricity-slider');
const eccentricityValueDisplay = document.getElementById('eccentricity-value');
const line = document.getElementById('line');
const solarSystem = document.querySelector('.solar-system');

const G = 6.67430e-11; // Gravitational constant in m^3 kg^-1 s^-2
const M = 1.989e30;    // Mass of the Sun in kilograms
const scale = 1e9;     // Scale down to make it suitable for visualization

let angle = 0;
const a = 200; // Semi-major axis (scaled units)
let eccentricity = parseFloat(eccentricitySlider.value);

let b, c, h;
let locations = []; // Array to store the Earth's positions
let timeStepCounter = 0; // Counter for time steps
let polygonCounter = 1; // Counter for polygons
let currentPolygon = null; // Currently active polygon

const dt = 2.01; // Time step in seconds

function updateOrbit() {
    b = a * Math.sqrt(1 - eccentricity * eccentricity); // Semi-minor axis
    c = a * eccentricity; // Distance from the center to the focus

    orbit.style.width = `${2 * a}px`;
    orbit.style.height = `${2 * b}px`;

    // Adjust the position of the orbit to keep the Sun at the focus
    orbit.style.setProperty('--yOffset', `${-c}px`);

    // Specific angular momentum (constant for a given orbit)
    h = Math.sqrt(G * M * a * (1 - eccentricity * eccentricity)) / scale; // Scaled h for visualization
}

// Event listener to update eccentricity based on slider value
eccentricitySlider.addEventListener('input', function() {
    eccentricity = parseFloat(this.value);
    eccentricityValueDisplay.textContent = eccentricity.toFixed(2);
    updateOrbit();
});

// Initialize orbit with current eccentricity value
updateOrbit();

function colorSchemeFunction(n) {
    // Simple color scheme using HSL (Hue, Saturation, Lightness)
    // n is the polygon index starting from 1
    const hue = (n * 60) % 360; // Change hue every polygon
    return `hsl(${hue}, 100%, 50%)`;
}

function startNewPolygon() {
    const svgNS = "http://www.w3.org/2000/svg";
    currentPolygon = document.createElementNS(svgNS, "polygon");

    currentPolygon.setAttribute("fill", colorSchemeFunction(polygonCounter));
    currentPolygon.setAttribute("opacity", "0.5");

    const connector = document.querySelector('.connector');
    connector.appendChild(currentPolygon);

    polygonCounter++;
}

function updateCurrentPolygon() {
    if (!currentPolygon) return;

    const centerX = solarSystem.clientWidth / 2;
    const centerY = solarSystem.clientHeight / 2;

    // Start the points string with the center of the solar system (Sun's position)
    let points = `${centerX},${centerY} `;

    // Add all the saved locations (Earth's positions)
    points += locations.map(loc => {
        const x = loc.x + centerX;
        const y = loc.y + centerY;
        return `${x},${y}`;
    }).join(" ");

    // Close the polygon by looping back to the Sun
    points += ` ${centerX},${centerY}`;

    currentPolygon.setAttribute("points", points);
}

function rotateEarth() {
    const x = a * Math.cos(angle);
    const y = b * Math.sin(angle);
    const r = Math.sqrt((x - c) ** 2 + y ** 2); // Distance to the sun

    // Orbital speed based on vis-viva equation (scaled for visualization)
    const v = Math.sqrt(G * M * (2 / r - 1 / a)) / scale;
    const angularSpeed = h / r ** 2; // Adjust angle based on angular speed

    angle += angularSpeed * dt;
    if (angle >= 360) angle -= 360;

    // Store current position in the locations array
    locations.push({ x: x - c, y: y });
    timeStepCounter++;

    // Update the current polygon with the new positions
    updateCurrentPolygon();

    // Every 100 time steps, start a new polygon and reset the locations array
    if (timeStepCounter >= 100) {
        startNewPolygon();
        locations = [];
        timeStepCounter = 0; // Reset the counter
    }

    // Move the Earth, taking into account the offset from the center
    earth.style.setProperty('--dx', `${x - c}px`);
    earth.style.setProperty('--dy', `${y}px`);

    // Update the line connecting the Sun to the Earth
    const sunRect = document.querySelector('.sun').getBoundingClientRect();
    const earthRect = earth.getBoundingClientRect();
    const solarSystemRect = document.querySelector('.solar-system').getBoundingClientRect();

    line.setAttribute('x1', `${sunRect.left + sunRect.width / 2 - solarSystemRect.left}`);
    line.setAttribute('y1', `${sunRect.top + sunRect.height / 2 - solarSystemRect.top}`);
    line.setAttribute('x2', `${earthRect.left + earthRect.width / 2 - solarSystemRect.left}`);
    line.setAttribute('y2', `${earthRect.top + earthRect.height / 2 - solarSystemRect.top}`);

    requestAnimationFrame(rotateEarth);
}

// Initialize the first polygon
startNewPolygon();

// Initialize the rotation
rotateEarth();
