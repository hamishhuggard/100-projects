class Sweep {
    constructor() {
        // Create the HTML element for this sweep
        this.element = this.createHTMLElement();

        // Initialize an empty array of points (excluding the focus)
        this.points = [];

        // Add this instance to the static array of sweeps
        Sweep.instances.push(this);

        // Refresh the HTML element with the initial state
        this.refreshElement();
    }

    // Static array to store all Sweep instances
    static instances = [];

    // Static method to get the color of the next Sweep
    static getNextColor() {
        const hue = (Sweep.instances.length * 60) % 360; // Same color scheme as before
        return `hsl(${hue}, 100%, 50%)`;
    }

    // Method to create the HTML (SVG) element for the sweep
    createHTMLElement() {
        const svgNS = "http://www.w3.org/2000/svg";
        const polygon = document.createElementNS(svgNS, "polygon");

        polygon.setAttribute("fill", Sweep.getNextColor());
        polygon.setAttribute("opacity", "0.5");

        const connector = document.querySelector('.connector');
        connector.appendChild(polygon);

        return polygon;
    }

    // Method to refresh the HTML element based on the current points
    refreshElement() {
        const centerX = solarSystem.clientWidth / 2;
        const centerY = solarSystem.clientHeight / 2;

        // Create the points string starting with the focus (center of the solar system)
        let pointsString = `${centerX},${centerY} `;

        // Add all the points in the array
        pointsString += this.points.map(point => {
            const x = point.x + centerX;
            const y = point.y + centerY;
            return `${x},${y}`;
        }).join(" ");

        // Close the polygon by returning to the focus
        pointsString += ` ${centerX},${centerY}`;

        // Update the element's points attribute
        this.element.setAttribute("points", pointsString);
    }

    // Method to add a new point to the sweep (and refresh the HTML element)
    addPoint(point) {
        this.points.push(point);
        this.refreshElement();
    }

    // Method to check if the sweep is full (more than 99 points)
    isFull() {
        return this.points.length > 99;
    }

    // Method to remove a point from the start of the array (and refresh the element)
    removeFirstPoint() {
        this.points.shift();
        if (this.points.length === 0) {
            this.delete();
        } else {
            this.refreshElement();
        }
    }

    // Method to delete the current sweep from the static array and the DOM
    delete() {
        // Remove the element from the DOM
        this.element.remove();

        // Remove this instance from the static array
        const index = Sweep.instances.indexOf(this);
        if (index > -1) {
            Sweep.instances.splice(index, 1);
        }
    }

    // Static method to delete all sweeps and create a new empty sweep
    static resetAll() {
        // Delete all current sweeps
        while (Sweep.instances.length > 0) {
            Sweep.instances[0].delete();
        }

        // Create a new initial sweep
        new Sweep();
    }

    // Static method to manage the gradual erasing of the first polygon as a new one is drawn
    static updateSweeps() {
        if (Sweep.instances.length > 1) {
            const firstSweep = Sweep.instances[0];
            firstSweep.removeFirstPoint();
        }
    }
}
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
let timeStepCounter = 0; // Counter for time steps

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

function rotateEarth() {
    const x = a * Math.cos(angle);
    const y = b * Math.sin(angle);
    const r = Math.sqrt((x - c) ** 2 + y ** 2); // Distance to the sun

    // Orbital speed based on vis-viva equation (scaled for visualization)
    const v = Math.sqrt(G * M * (2 / r - 1 / a)) / scale;
    const angularSpeed = h / r ** 2; // Adjust angle based on angular speed

    angle += angularSpeed * dt;
    if (angle >= 360) angle -= 360;

    const point = { x: x - c, y: y };
    const currentSweep = Sweep.instances[Sweep.instances.length - 1];

    // Add the current point to the active sweep
    currentSweep.addPoint(point);

    // If the sweep is full, create a new one
    if (currentSweep.isFull()) {
        new Sweep();
    }

    // Update sweeps to gradually erase the first polygon
    Sweep.updateSweeps();

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

// Event listener to update eccentricity based on slider value
eccentricitySlider.addEventListener('input', function () {
    eccentricity = parseFloat(this.value);
    eccentricityValueDisplay.textContent = eccentricity.toFixed(2);
    updateOrbit();
    Sweep.resetAll(); // Clear all sweeps and start fresh when eccentricity changes
});

// Initialize orbit with current eccentricity value
updateOrbit();

// Initialize the first sweep
new Sweep();

// Initialize the rotation
rotateEarth();
