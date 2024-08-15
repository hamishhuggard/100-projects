let countdown;
let idleTimer;
let totalTime = 60; // Default 1 minute
let timeRemaining = totalTime;
let timerStarted = false;

const timerDisplay = document.getElementById('timer');
const textInput = document.getElementById('textInput');

function startCountdown() {
    if (!timerStarted) {
        timerStarted = true;
        countdown = setInterval(() => {
            timeRemaining--;
            updateTimerDisplay();

            if (timeRemaining <= 0) {
                resetTimer();
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
    updateTimerDisplay();
    textInput.value = '';
    textInput.classList.remove('smooth-transition');
    textInput.style.backgroundColor = 'white'; // Reset background color
}

function handleTyping() {
    if (!timerStarted) {
        startCountdown();
    }
    
    clearTimeout(idleTimer);
    textInput.classList.remove('smooth-transition');

    idleTimer = setTimeout(() => {
        textInput.classList.add('smooth-transition');
        idleTimer = setTimeout(() => {
            resetTimer();
        }, 3000);
    }, 200);
}

textInput.addEventListener('input', handleTyping);
updateTimerDisplay();
