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


const loader = new THREE.TextureLoader();
loader.load('arrow.png', function(texture) {
    const scale = 0.01;
    const planeGeometry = new THREE.PlaneGeometry(223*scale, 1314*scale);
    const planeMaterial = new THREE.MeshBasicMaterial({
        map: texture,
        transparent: true,
        side: THREE.DoubleSide,  // Make the plane double-sided
        alphaTest: 0.5           // Enable alpha transparency
    });

    // Create and position the plane for the x-axis
    xAxisArrow = new THREE.Mesh(planeGeometry, planeMaterial);
    xAxisArrow.position.set(0, 0, 0); // Position it at the end of the x-axis
    xAxisArrow.rotation.z = -Math.PI / 2; // Rotate to align with the x-axis
    scene.add(xAxisArrow);
    
    // Create and position the plane for the y-axis
    yAxisArrow = new THREE.Mesh(planeGeometry, planeMaterial);
    yAxisArrow.position.set(0, 0, 0); // Position it at the end of the y-axis
    scene.add(yAxisArrow);

    // Create and position the plane for the x-axis
    zAxisArrow = new THREE.Mesh(planeGeometry, planeMaterial);
    zAxisArrow.position.set(0, 0, 0); // Position it at the end of the x-axis
    zAxisArrow.rotation.x = -Math.PI / 2; // Rotate to align with the x-axis
    scene.add(zAxisArrow);

});


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

let xAxisArrow;
let yAxisArrow;
let zAxisArrow;


// Handle window resize
window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});
function animate() {
    requestAnimationFrame(animate);

    // Update time and wavefunction
    time += timeStep;
    updatePsiPoints();

    // Calculate the angle to rotate on the x-axis so that the arrow faces the camera
    if (xAxisArrow) {
        const cameraDistanceZ = camera.position.z - xAxisArrow.position.z;
        const cameraDistanceY = camera.position.y - xAxisArrow.position.y;
        xAxisArrow.rotation.x = -Math.atan2(cameraDistanceY, cameraDistanceZ);
    }
    if (yAxisArrow) {
        const cameraDistanceZ = camera.position.z - yAxisArrow.position.z;
        const cameraDistanceX = camera.position.x - yAxisArrow.position.x;
        yAxisArrow.rotation.y = Math.atan2(cameraDistanceX, cameraDistanceZ);
    }
    if (zAxisArrow) {
        const cameraDistanceX = camera.position.x - zAxisArrow.position.x;
        const cameraDistanceY = camera.position.y - zAxisArrow.position.y;
        zAxisArrow.rotation.y = Math.atan2(cameraDistanceX, cameraDistanceY);
    }

    controls.update();
    renderer.render(scene, camera);
}

animate();
