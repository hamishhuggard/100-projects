// NEXT TIME:
// 

document.getElementById("walk1");




let imageCounter = 0;

function nextFrame() {
    
    imageCounter = imageCounter + 1;
    if (imageCounter > 2) {
        imageCounter = 0;
    } 
    console.log('walk' + imageCounter + '.png');
    document.getElementById("walk1").src = 'walk' + imageCounter + '.png';
}

setInterval(nextFrame, 1000);

function startGame(event) {
    console.log(event.key);
}
    
document.addEventListener("keydown", startGame);