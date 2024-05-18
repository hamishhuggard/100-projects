import * as THREE from 'three';
import { CSS2DRenderer, CSS2DObject } from 'three/examples/jsm/renderers/CSS2DRenderer.js';
import * as d3 from 'd3';

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
//camera.position.z = 5;

const renderer = new THREE.WebGLRenderer();
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

const labelRenderer = new CSS2DRenderer();
labelRenderer.setSize(window.innerWidth, window.innerHeight);
labelRenderer.domElement.style.position = 'absolute';
labelRenderer.domElement.style.top = '0';
document.body.appendChild(labelRenderer.domElement);

async function loadData() {
    let data = await d3.csv('models.csv');
    data = data.filter(row => row.System.startsWith('GPT'));
    console.table(data);
    console.log(Object.keys(data[0]));
    data.forEach(d => {
        console.log(d.System);
        console.log(d.Parameters);
    })
}

function createLabel(text, position) {
    const div = document.createElement('div');
    div.className = 'label';
    div.textContent = text;
    //div.style.marginTop = '-1em'; // Adjust based on your specific needs
    const label = new CSS2DObject(div);
    label.position.set(position.x, position.y + 1.5, position.z);
    return label;
}

const cubes = [];

let x = 0;
[
    { size: 1e3, x: -0.1, name: 'perceptron', year: 1957},
    //{ size: 8e3, x: -0.1, name: 'deep blue', year: 1997},
    { size: 10.5e3, x: -0.1, name: 'LSTM', year: 1997},
    { size: 11.9e6, x: -0.1, name: 'NPLM', year: 2003},
    //{ size: 60e6, x: -0.1, name: 'word2vec', year: 2013},
    { size: 60e6, x: -0.1, name: 'alexnet', year: 2012},
    //{ size: 117e6, x: 0, name: 'gpt-1', year: 2018},
    { size: 1.5e9, x: 1, name: 'gpt-2', year: 2019},
    { size: 175e9, x: 1100, name: 'gpt-3', year: 2020},
    { size: 1.76e12, x: 1100, name: 'gpt-4', year: 2023},
    { size: 150e12, x: 1100, name: 'human neocortex', year: 2023}
].forEach((d, i) => {
    const length = Math.cbrt(d.size) / 500;
    x += length * 1.0;
    const geometry = new THREE.BoxGeometry(length, length, length);
    const material = new THREE.MeshBasicMaterial({ color: 0x00ff00, wireframe: true });
    const cube = new THREE.Mesh(geometry, material);
    cube.position.x = x;
    x += length * 1.0;
    cube.position.y = length/2;
    cubes.push(cube)
    scene.add(cube);

    const label = createLabel(`Cube ${i + 1}`, cube.position);
    cube.add(label);
});

console.log(cubes.length);
document.body.style.height = `${window.innerHeight * cubes.length}px`;

const grid = new THREE.GridHelper(100, 100);
//scene.add(grid);

window.addEventListener('resize', onWindowResize, false);

function animate() {
    requestAnimationFrame(animate);
    renderer.render(scene, camera);
    labelRenderer.render(scene, camera);
}
animate();

function onWindowResize() {
    renderer.setSize(window.innerWidth, window.innerHeight);
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    
}

function onDocumentScroll() {
    const maxScroll = document.body.scrollHeight - window.innerHeight;
    const scrollFraction = window.scrollY / maxScroll;
    const linearZoom = Math.exp(scrollFraction * cubes.length - 2.1);
    camera.position.z = linearZoom;
    camera.position.x = linearZoom/2;
    camera.position.y = linearZoom/2;
}
onDocumentScroll();


document.body.onscroll = onDocumentScroll;
