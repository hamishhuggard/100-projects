document.addEventListener('DOMContentLoaded', function () {
    const textContent = document.getElementById('textContent');
    const audioPlayer = document.getElementById('audioPlayer');

    // Function to fetch the speech marks file and process it line by line
    fetch('speech_marks.json')
        .then(response => response.text())
        .then(text => {
            // Split the file into lines and parse each line as a JSON object
            const speechMarks = text.split('\n').map(line => {
                return line ? JSON.parse(line) : null;
            }).filter(mark => mark !== null);

            console.log(speechMarks);

            // Populate the text content area with words
            speechMarks.forEach(mark => {
                const span = document.createElement('span');
                span.textContent = mark.value + ' ';
                span.dataset.time = mark.time;
                textContent.appendChild(span);
            });

            // Highlight words as they are spoken
            audioPlayer.addEventListener('timeupdate', function () {
                const currentTime = audioPlayer.currentTime * 1000; // Convert to milliseconds
                speechMarks.forEach((mark, index) => {
                    const span = textContent.children[index];
                    if (currentTime >= mark.time) {
                        span.classList.add('highlight');
                    } else {
                        span.classList.remove('highlight');
                    }
                });
            });
        })
        .catch(error => console.error('Error loading speech marks:', error));
});
