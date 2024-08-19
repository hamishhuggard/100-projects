const earth = document.querySelector('.earth');
const sun = document.querySelector('.sun');

let angle = 0;

function rotateEarth() {
    angle = (angle + 1) % 360;
    const radians = angle * (Math.PI / 180);
    const x = 150 * Math.cos(radians) - 150;
    const y = 150 * Math.sin(radians);

    earth.style.transform = `translate(${x}px, ${y}px)`;
    requestAnimationFrame(rotateEarth);
}

rotateEarth();
