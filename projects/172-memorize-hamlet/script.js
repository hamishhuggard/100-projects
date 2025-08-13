// Constants
const CASE_SENSITIVE = false; // Initially set to false (case insensitive)
const CARE_ABOUT_PUNCTUATION = false; // Initially set to false (don't care about punctuation)

// Hamlet's "To be or not to be" speech
const TARGET_TEXT = `To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles,
And by opposing end them. To die, to sleep—
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep: perchance to dream: ay, there's the rub;
For in that sleep of death what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause: there's the respect
That makes calamity of so long life;
For who would bear the whips and scorns of time,
The oppressor's wrong, the proud man's contumely,
The pangs of despised love, the law's delay,
The insolence of office and the spurns
That patient merit of the unworthy takes,
When he himself might his quietus make
With a bare bodkin? who would fardels bear,
To grunt and sweat under a weary life,
But that the dread of something after death,
The undiscover'd country from whose bourn
No traveller returns, puzzles the will
And makes us rather bear those ills we have
Than fly to others that we know not of?
Thus conscience does make cowards of us all;
And thus the native hue of resolution
Is sicklied o'er with the pale cast of thought,
And enterprises of great pith and moment
With this regard their currents turn awry,
And lose the name of action.`;

// DOM elements
const targetText = document.getElementById('targetText');
const typedText = document.getElementById('typedText');
const timer = document.getElementById('timer');
const progress = document.getElementById('progress');
const textDisplay = document.getElementById('textDisplay');
const incorrectSound = document.getElementById('incorrectSound');
const caseSensitiveCheckbox = document.getElementById('caseSensitive');
const punctuationCheckbox = document.getElementById('punctuation');
const resetBtn = document.getElementById('resetBtn');

// State variables
let startTime = null;
let isRunning = false;
let currentText = '';
let currentIndex = 0;
let isEnterHeld = false;

// Initialize the app
function init() {
    // Set initial checkbox states
    caseSensitiveCheckbox.checked = CASE_SENSITIVE;
    punctuationCheckbox.checked = CARE_ABOUT_PUNCTUATION;
    
    // Display the target text
    targetText.textContent = TARGET_TEXT;
    
    // Set up event listeners
    setupEventListeners();
    
    // Make the page focusable for key events
    document.body.tabIndex = 0;
    document.body.focus();
    
    // Start by showing the typed text area (empty initially)
    showTypedText();
}

// Set up event listeners
function setupEventListeners() {
    // Key handling
    document.addEventListener('keydown', handleKeyDown);
    document.addEventListener('keyup', handleKeyUp);
    
    // Checkbox changes
    caseSensitiveCheckbox.addEventListener('change', handleSettingsChange);
    punctuationCheckbox.addEventListener('change', handleSettingsChange);
    
    // Reset button
    resetBtn.addEventListener('click', resetTest);
    
    // Click anywhere to focus for key events
    document.addEventListener('click', () => {
        document.body.focus();
    });
}

// Handle key down events
function handleKeyDown(e) {
    // Handle Enter key for showing full text
    if (e.key === 'Enter') {
        e.preventDefault();
        isEnterHeld = true;
        showFullText();
        return;
    }
    
    // Handle other keys for typing
    if (e.key.length === 1) {
        handleTyping(e.key);
    }
}

// Handle key up events
function handleKeyUp(e) {
    // Handle Enter key release
    if (e.key === 'Enter') {
        isEnterHeld = false;
        showTypedText();
    }
}

// Show the full target text
function showFullText() {
    targetText.style.display = 'block';
    typedText.style.display = 'none';
}

// Show only the typed text
function showTypedText() {
    targetText.style.display = 'none';
    typedText.style.display = 'block';
}

// Handle typing input
function handleTyping(key) {
    if (!isRunning) {
        startTest();
    }
    
    // Get the current character that should be typed
    const expectedChar = TARGET_TEXT[currentIndex];
    
    if (expectedChar === undefined) {
        return; // Text is complete
    }
    
    // Check if the input matches the expected character
    if (isCharacterMatch(key, expectedChar)) {
        // Correct character
        currentText += expectedChar;
        currentIndex++;
        updateDisplay();
        updateProgress();
        
        // Check if text is complete
        if (currentIndex >= TARGET_TEXT.length) {
            completeTest();
        }
    } else {
        // Incorrect character
        playIncorrectSound();
        shakeTextDisplay();
    }
}

// Check if characters match based on settings
function isCharacterMatch(input, expected) {
    let inputChar = input;
    let expectedChar = expected;
    
    // Handle case sensitivity
    if (!caseSensitiveCheckbox.checked) {
        inputChar = inputChar.toLowerCase();
        expectedChar = expectedChar.toLowerCase();
    }
    
    // Handle punctuation
    if (!punctuationCheckbox.checked) {
        // If expected char is punctuation, skip it
        if (/[^\w\s]/.test(expected)) {
            return true;
        }
        
        // If input char is punctuation, skip it
        if (/[^\w\s]/.test(input)) {
            return true;
        }
    }
    
    return inputChar === expectedChar;
}

// Start the test
function startTest() {
    if (!isRunning) {
        isRunning = true;
        startTime = Date.now();
        startTimer();
    }
}

// Start the timer
function startTimer() {
    const timerInterval = setInterval(() => {
        if (!isRunning) {
            clearInterval(timerInterval);
            return;
        }
        
        const elapsed = Date.now() - startTime;
        const minutes = Math.floor(elapsed / 60000);
        const seconds = Math.floor((elapsed % 60000) / 1000);
        const milliseconds = Math.floor((elapsed % 1000) / 10);
        
        timer.textContent = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}.${milliseconds.toString().padStart(2, '0')}`;
    }, 10);
}

// Update the display
function updateDisplay() {
    typedText.textContent = currentText;
}

// Update progress
function updateProgress() {
    const percentage = Math.round((currentIndex / TARGET_TEXT.length) * 100);
    progress.textContent = `${percentage}%`;
}

// Complete the test
function completeTest() {
    isRunning = false;
    const totalTime = Date.now() - startTime;
    const minutes = Math.floor(totalTime / 60000);
    const seconds = Math.floor((totalTime % 60000) / 1000);
    const milliseconds = Math.floor((totalTime % 1000) / 10);
    
    // Show completion message
    setTimeout(() => {
        alert(`Congratulations! You completed the text in ${minutes}:${seconds.toString().padStart(2, '0')}.${milliseconds.toString().padStart(2, '0')}`);
    }, 100);
}

// Play incorrect sound
function playIncorrectSound() {
    incorrectSound.currentTime = 0;
    incorrectSound.play().catch(e => {
        // Ignore errors if audio can't play
        console.log('Audio playback failed:', e);
    });
}

// Shake the text display
function shakeTextDisplay() {
    textDisplay.classList.add('shake');
    setTimeout(() => {
        textDisplay.classList.remove('shake');
    }, 500);
}

// Handle settings changes
function handleSettingsChange() {
    // Reset the test when settings change
    resetTest();
}

// Reset the test
function resetTest() {
    isRunning = false;
    startTime = null;
    currentText = '';
    currentIndex = 0;
    isEnterHeld = false;
    
    // Reset display
    typedText.textContent = '';
    showTypedText(); // Show typed text area by default
    
    // Reset timer and progress
    timer.textContent = '00:00.00';
    progress.textContent = '0%';
    
    // Remove shake class
    textDisplay.classList.remove('shake');
    
    // Focus on body for key events
    document.body.focus();
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', init); 