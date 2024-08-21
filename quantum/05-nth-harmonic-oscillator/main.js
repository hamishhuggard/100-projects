import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

// Scene setup
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xffffff);

const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(0, 10, 20);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);

// Axes
const axesHelper = new THREE.AxesHelper(10);
scene.add(axesHelper);

// Parabola on the x/y plane
const parabolaPoints = [];
for (let x = -5; x <= 5; x += 0.1) {
    parabolaPoints.push(new THREE.Vector3(x, Math.pow(x, 2) / 5, 0));
}
const parabolaGeometry = new THREE.BufferGeometry().setFromPoints(parabolaPoints);
const parabolaMaterial = new THREE.LineBasicMaterial({ color: 0x0000ff });
const parabola = new THREE.Line(parabolaGeometry, parabolaMaterial);
scene.add(parabola);

// Function to generate the nth Hermite polynomial
function hermitePolynomial(n) {
    if (n === 0) return (x) => 1;
    if (n === 1) return (x) => 2 * x;

    const H_n_minus_2 = hermitePolynomial(n - 2);
    const H_n_minus_1 = hermitePolynomial(n - 1);

    return (x) => 2 * x * H_n_minus_1(x) - 2 * (n - 1) * H_n_minus_2(x);
}

// Factorial function
function factorial(num) {
    if (num === 0 || num === 1) return 1;
    return num * factorial(num - 1);
}

// Parameters
let n = 3;  // Starting value of n
const m = 1;
const omega = 1;
const hbar = 1;

function calculateNormFactor(n) {
    return Math.pow(m * omega / (Math.PI * hbar), 1 / 4) * (1 / Math.sqrt(Math.pow(2, n) * factorial(n)));
}

// Compute the nth state wavefunction
function calculatePsi(x, t, n) {
    const normFactor = calculateNormFactor(n);
    const hermite = hermitePolynomial(n);
    const psi_n_x = normFactor * hermite(Math.sqrt(m * omega / hbar) * x) * Math.exp(-m * omega * x * x / (2 * hbar));
    const realPart = psi_n_x * Math.cos((n + 0.5) * omega * t);
    const imagPart = psi_n_x * Math.sin((n + 0.5) * omega * t);
    return { realPart, imagPart };
}

let time = 0;
const timeStep = 0.01;

// Psi (Wavefunction) - rotating in space
const psiPoints = [];

function updatePsiPoints() {
    psiPoints.length = 0; // Clear previous points
    for (let x = -5; x <= 5; x += 0.1) {
        const { realPart, imagPart } = calculatePsi(x, time, n);
        psiPoints.push(new THREE.Vector3(x, realPart, imagPart));
    }
    const psiGeometry = new THREE.BufferGeometry().setFromPoints(psiPoints);
    psi.geometry.dispose();
    psi.geometry = psiGeometry;
}

const psiMaterial = new THREE.LineBasicMaterial({ color: 0xff0000 });
const psiGeometry = new THREE.BufferGeometry();
const psi = new THREE.Line(psiGeometry, psiMaterial);
scene.add(psi);

// Function to update the scene with the new n
function updateScene() {
    updatePsiPoints();
}

// Add event listeners to buttons
document.getElementById('increase-n').addEventListener('click', () => {
    n++;
    updateScene();
});

document.getElementById('decrease-n').addEventListener('click', () => {
    if (n > 0) {
        n--;
        updateScene();
    }
});

// Animation loop
function animate() {
    requestAnimationFrame(animate);

    // Update time and wavefunction
    time += timeStep;
    updatePsiPoints();

    controls.update();
    renderer.render(scene, camera);
}

animate();

// Handle window resize
window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});
