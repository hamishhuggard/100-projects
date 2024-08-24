const words = ["apple", "banana", "cherry", "date", "elderberry", "fig", "grape", "honeydew", "kiwi", "lemon"];

const word1Button = document.getElementById('word1');
const word2Button = document.getElementById('word2');

function getRandomWords() {
    let word1Index = Math.floor(Math.random() * words.length);
    let word2Index;
    
    do {
        word2Index = Math.floor(Math.random() * words.length);
    } while (word1Index === word2Index);
    
    return [words[word1Index], words[word2Index]];
}

function updateWords() {
    const [word1, word2] = getRandomWords();
    word1Button.textContent = word1;
    word2Button.textContent = word2;
}

word1Button.addEventListener('click', function() {
    console.log('You chose:', word1Button.textContent);
    updateWords();
});

word2Button.addEventListener('click', function() {
    console.log('You chose:', word2Button.textContent);
    updateWords();
});

// Initialize the first set of words
updateWords();
