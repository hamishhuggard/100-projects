// Constants
// Removed case sensitivity and punctuation sensitivity features

// Sample German-English word pairs for learning
const WORD_PAIRS = [
    'Haus house',
    'Katze cat', 
    'Hund dog',
    'Buch book',
    'Tisch table',
    'Stuhl chair',
    'Fenster window',
    'Tür door',
    'Auto car',
    'Brot bread',
    'Wasser water',
    'Milch milk',
    'Apfel apple',
    'Käse cheese',
    'Fleisch meat',
    'Gemüse vegetables',
    'Obst fruit',
    'Kaffee coffee',
    'Tee tea'
];

// Shuffle function for word pairs
function shuffleArray(array) {
    const shuffled = [...array];
    for (let i = shuffled.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    return shuffled;
}

// Create shuffled target text
const TARGET_TEXT = shuffleArray(WORD_PAIRS).join(' ');

// DOM elements
const targetText = document.getElementById('targetText');
const typedText = document.getElementById('typedText');
const timer = document.getElementById('timer');
const progress = document.getElementById('progress');
const textDisplay = document.getElementById('textDisplay');
const incorrectSound = document.getElementById('incorrectSound');
const resetBtn = document.getElementById('resetBtn');

// State variables
let startTime = null;
let isRunning = false;
let currentText = '';
let currentIndex = 0;

// Initialize the app
function init() {
    // Set initial checkbox states
    // Removed caseSensitiveCheckbox and punctuationCheckbox
    
    // Display the target text with HTML formatting
    targetText.innerHTML = TARGET_TEXT;
    
    // Set up event listeners
    setupEventListeners();
    
    // Make the page focusable for key events
    document.body.tabIndex = 0;
    document.body.focus();
    
    // Start by hiding the target text (memory mode)
    showTypedText();
}

// Set up event listeners
function setupEventListeners() {
    // Key handling
    document.addEventListener('keydown', handleKeyDown);
    document.addEventListener('keyup', handleKeyUp);
    
    // Removed caseSensitiveCheckbox and punctuationCheckbox listeners
    
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

// Check if characters match
function isCharacterMatch(input, expected) {
    // Handle umlaut substitutions for easier typing
    const umlautMap = {
        'a': ['a', 'ä'],
        'ä': ['a', 'ä'],
        'o': ['o', 'ö'],
        'ö': ['o', 'ö'],
        'u': ['u', 'ü'],
        'ü': ['u', 'ü'],
        's': ['s', 'ß'],
        'ß': ['s', 'ß']
    };
    
    // Check if either character is a umlaut variant
    const inputVariants = umlautMap[input] || [input];
    const expectedVariants = umlautMap[expected] || [expected];
    
    // Check if there's any overlap between the variants
    return inputVariants.some(inputVariant => 
        expectedVariants.some(expectedVariant => inputVariant === expectedVariant)
    );
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



// Reset the test
function resetTest() {
    isRunning = false;
    startTime = null;
    currentText = '';
    currentIndex = 0;
    
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