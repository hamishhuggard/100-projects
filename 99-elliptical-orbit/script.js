const earth = document.querySelector('.earth');
const orbit = document.querySelector('.orbit');
const eccentricitySlider = document.getElementById('eccentricity-slider');
const eccentricityValueDisplay = document.getElementById('eccentricity-value');
let angle = 0;

const a = 200; // Semi-major axis
let eccentricity = parseFloat(eccentricitySlider.value);

let b, c;

function updateOrbit() {
    b = a * Math.sqrt(1 - eccentricity * eccentricity); // Semi-minor axis
    c = a * eccentricity; // Distance from the center to the focus

    orbit.style.width = `${2 * a}px`;
    orbit.style.height = `${2 * b}px`;

    // Adjust the position of the orbit to keep the Sun at the focus
    orbit.style.setProperty('--yOffset', `${-c}px`);
}

// Event listener to update eccentricity based on slider value
eccentricitySlider.addEventListener('input', function() {
    eccentricity = parseFloat(this.value);
    eccentricityValueDisplay.textContent = eccentricity.toFixed(2);
    updateOrbit();
});

// Initialize orbit with current eccentricity value
updateOrbit();

function rotateEarth() {
    angle = (angle + 1) % 360;
    const radians = angle * (Math.PI / 180);
    const x = a * Math.cos(radians);
    const y = b * Math.sin(radians);

    // Move the Earth, taking into account the offset from the center
    earth.style.setProperty('--dx', `${x - c}px`);
    earth.style.setProperty('--dy', `${y}px`);

    requestAnimationFrame(rotateEarth);
}

rotateEarth();
