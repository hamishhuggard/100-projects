let countdown;
let idleTimer;
let totalTime = 60; // Default 1 minute
let timeRemaining = totalTime;
let timerStarted = false;
let done = false;

const timerDisplay = document.getElementById('timer');
const textInput = document.getElementById('textInput');
const content = document.getElementById('content');

function startCountdown() {
    if (!timerStarted) {
        timerStarted = true;
        countdown = setInterval(() => {
            timeRemaining--;
            updateTimerDisplay();

            if (timeRemaining <= 0) {
                resetTimer();
                done = true;
            }
        }, 1000);
    }
}

function updateTimerDisplay() {
    let minutes = Math.floor(timeRemaining / 60);
    let seconds = timeRemaining % 60;
    timerDisplay.textContent = `${minutes < 10 ? '0' : ''}${minutes}:${seconds < 10 ? '0' : ''}${seconds}`;
}

function resetTimer() {
    clearInterval(countdown);
    timerStarted = false;
    timeRemaining = totalTime;
    //updateTimerDisplay();
    //textInput.value = '';
    content.classList.remove('smooth-transition');
    content.style.backgroundColor = 'white'; // Reset background color
}

function handleTyping() {
    if (done) {
        return;
    }
    if (!timerStarted) {
        startCountdown();
    }
    
    clearTimeout(idleTimer);
    content.classList.remove('smooth-transition');

    if (timeRemaining < 3) {
        return;
    }
    idleTimer = setTimeout(() => {
        content.classList.add('smooth-transition');
        idleTimer = setTimeout(() => {
            resetTimer();
        }, 3000);
    }, 1000);
}

textInput.addEventListener('input', handleTyping);
updateTimerDisplay();
