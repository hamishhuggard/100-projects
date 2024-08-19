const earth = document.querySelector('.earth');
const orbit = document.querySelector('.orbit');
let angle = 0;
const eccentricity = 0.5; // Adjust the eccentricity here (0 <= eccentricity < 1)

const a = 200; // Semi-major axis
const b = a * Math.sqrt(1 - eccentricity * eccentricity); // Semi-minor axis

// Set the orbit dimensions based on the calculated a and b
orbit.style.width = `${2 * a}px`;
orbit.style.height = `${2 * b}px`;
/*
orbit.style.marginLeft = `-${a*0.7}px`;
orbit.style.marginTop = `-${b*0.75}px`;
*/

function rotateEarth() {
    angle = (angle + 1) % 360;
    const radians = angle * (Math.PI / 180);
    const x = a * Math.cos(radians);
    const y = b * Math.sin(radians);

    earth.style.setProperty('--dx', `${x}px`);
    earth.style.setProperty('--dy', `${y}px`);

    requestAnimationFrame(rotateEarth);
}

rotateEarth();
