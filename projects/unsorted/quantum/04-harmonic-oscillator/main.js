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

// Psi (Wavefunction) - rotating in space
const psiPoints = [];
const m = 1;
const omega = 1;
const hbar = 1;
const normFactor = Math.pow(m * omega / (Math.PI * hbar), 1/4);

function calculatePsi(x, t) {
    const realPart = normFactor * Math.exp(-m * omega * x * x / (2 * hbar)) * Math.cos(omega * t / 2);
    const imagPart = normFactor * Math.exp(-m * omega * x * x / (2 * hbar)) * Math.sin(omega * t / 2);
    return { realPart, imagPart };
}

let time = 0;
const timeStep = 0.01;

function updatePsiPoints() {
    psiPoints.length = 0; // Clear previous points
    for (let x = -5; x <= 5; x += 0.1) {
        const { realPart, imagPart } = calculatePsi(x, time);
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
