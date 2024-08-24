document.addEventListener('keydown', function(event) {
    const square = document.getElementById('square');
    const stepSize = 10; // Number of pixels to move per key press
    const maxLeft = 0; // Left boundary
    const maxRight = window.innerWidth - square.offsetWidth; // Right boundary

    let currentLeft = square.offsetLeft;

    if (event.key === 'ArrowLeft') {
        // Move left
        if (currentLeft - stepSize >= maxLeft) {
            square.style.left = `${currentLeft - stepSize}px`;
        } else {
            square.style.left = `${maxLeft}px`;
        }
    } else if (event.key === 'ArrowRight') {
        // Move right
        if (currentLeft + stepSize <= maxRight) {
            square.style.left = `${currentLeft + stepSize}px`;
        } else {
            square.style.left = `${maxRight}px`;
        }
    }
});
