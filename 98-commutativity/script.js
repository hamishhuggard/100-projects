const triangle = document.querySelector('.triangle');
const square = document.querySelector('.square');
const commuteBtn = document.getElementById('commuteBtn');

let swapped = false;

commuteBtn.addEventListener('click', () => {
    if (swapped) {
        square.style.transform = 'translateX(0)';
        triangle.style.transform = 'translateX(0)';
    } else {
        square.style.transform = 'translateX(150px)';
        triangle.style.transform = 'translateX(-150px)';
    }
    swapped = !swapped;
});
