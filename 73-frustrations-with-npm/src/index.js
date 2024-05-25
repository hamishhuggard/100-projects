console.log('hello')

/*
import * as THREE from 'three';
import { PointerLockControls } from 'three/examples/jsm/controls/PointerLockControls.js';

let camera, scene, renderer;
let gridHelper;

let controls;

function init() {
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0xcccccc);

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    document.body.appendChild(renderer.domElement);

    camera = new THREE.PerspectiveCamera(70, window.innerWidth / window.innerHeight, 0.01, 100);
    camera.position.set(0, 5, 0);

    controls = new THREE.PointerLockControls(camera, renderer.domElement);

    // Add event listener to lock pointer on click
    document.addEventListener('click', () => {
        controls.lock();
    }, false);

    gridHelper = new THREE.GridHelper(100, 100);
    scene.add(gridHelper);

    animate();
}

function animate() {
    requestAnimationFrame(animate);
    renderer.render(scene, camera);
}

init();
*/
