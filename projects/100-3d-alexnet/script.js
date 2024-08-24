// Create the scene, camera, and renderer
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer();
renderer.setSize(window.innerWidth, window.innerHeight);
document.getElementById('container').appendChild(renderer.domElement);

// AlexNet layers with dimensions according to the provided hyperparameters
const layers = [
    { width: 224, height: 224, depth: 3, color: 0x156289, name: 'Input', label: '224x224x3' },
    { width: 55, height: 55, depth: 96, color: 0x1b7ea4, name: 'Conv1', label: '55x55x96\n11x11 Filter, Stride 4' },
    /*
    { width: 27, height: 27, depth: 256, color: 0x1d89bd, name: 'Conv2', label: '27x27x256\n5x5 Filter' },
    { width: 13, height: 13, depth: 384, color: 0x2395d3, name: 'Conv3', label: '13x13x384\n3x3 Filter' },
    { width: 13, height: 13, depth: 384, color: 0x289fe8, name: 'Conv4', label: '13x13x384\n3x3 Filter' },
    { width: 13, height: 13, depth: 256, color: 0x2ba4f0, name: 'Conv5', label: '13x13x256\n3x3 Filter' },
    { width: 1, height: 1, depth: 4096, color: 0x32b9ff, name: 'FC6', label: '4096' },
    { width: 1, height: 1, depth: 4096, color: 0x35c4ff, name: 'FC7', label: '4096' },
    { width: 1, height: 1, depth: 1000, color: 0x38d0ff, name: 'FC8', label: '1000' },
    */
];

// Position the layers sequentially
const PADDING = 25
const SCALE = 0.1;
const totalDepth = layers.reduce((sum, layer) => sum + layer.depth*SCALE + PADDING, 0) - PADDING;

let positionZ = -totalDepth/2;

console.log({ totalDepth, positionZ })

layers.forEach(layer => {
    const geometry = new THREE.BoxGeometry(layer.width*SCALE, layer.height*SCALE, layer.depth*SCALE);
    const material = new THREE.MeshBasicMaterial({ color: layer.color, wireframe: false });
    const cube = new THREE.Mesh(geometry, material);
    cube.position.z = positionZ;
    scene.add(cube);

    console.log({ positionZ, depth: layer.depth*SCALE })

    // Add labels
    const loader = new THREE.FontLoader();
    loader.load('https://threejs.org/examples/fonts/helvetiker_regular.typeface.json', function (font) {
        return;
        const textGeometry = new THREE.TextGeometry(layer.label, {
            font: font,
            size: 2,
            height: 0.1,
        });
        const textMaterial = new THREE.MeshBasicMaterial({ color: 0xffffff });
        const mesh = new THREE.Mesh(textGeometry, textMaterial);
        mesh.position.set(-layer.width*SCALE / 40, -layer.height*SCALE / 40, positionZ);
        scene.add(mesh);
    });

    positionZ += layer.depth*SCALE + PADDING;
});

// Camera position
camera.position.z = -1*totalDepth;
//camera.position.z = -100;
scene.rotation.y = 0.11;
renderer.render(scene, camera);

// Animation loop
function animate() {
    requestAnimationFrame(animate);
    scene.rotation.y += 0.005;
    renderer.render(scene, camera);
}

animate();

// Handle window resize
window.addEventListener('resize', () => {
    const width = window.innerWidth;
    const height = window.innerHeight;
    renderer.setSize(width, height);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
});
