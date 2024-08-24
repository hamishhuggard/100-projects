/*
document.getElementById('customForm').addEventListener('submit', function(e) {
    e.preventDefault();

    const formData = new FormData(e.target);

    fetch('https://docs.google.com/forms/d/e/1FAIpQLScotYOPHo8PogU9jfWyJoQLQ-xGu38JyAXxTVa5ufmhxq2lTw/formResponse', {
        method: 'POST',
        body: formData,
        mode: 'no-cors'
    }).then(response => {
        alert('Form submitted successfully!');
    }).catch(error => {
        console.error('Error!', error.message);
    });
});
*/
async function fetchCSV(url) {
    const response = await fetch(url);
    const text = await response.text();
    const lines = text.trim().split('\n');
    return lines.map(line => line.trim().split(','));
}

async function loadTweets() {
    const tweetsUrl = 'https://docs.google.com/spreadsheets/d/1vEOyjh8bhBfDLe-6BHCQxG3B7NqAmqICtZXA4Cewovc/export?format=csv&gid=1029905748&single=true&output=csv;' // Replace with the actual URL
    
    const [rawTweets] = await Promise.all([fetchCSV(tweetsUrl)]);
    console.log({rawTweets})
    
    const tweets = rawTweets.slice(1); // Skip header row
    
    const tweetsContainer = document.getElementById('tweets');
    tweets.forEach(([datetime, handle, tweet]) => {
        console.log({handle, tweet, datetime})
        const tweetElement = document.createElement('div');
        tweetElement.className = 'tweet';
        tweetElement.innerHTML = `
            <div><span class="username">${handle}</span></div>
            <div>${tweet}</div>
            <div class="datetime">${datetime}</div>
        `;
        tweetsContainer.appendChild(tweetElement);
    });
}

loadTweets();
