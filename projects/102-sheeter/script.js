async function fetchCSV(url) {
    const response = await fetch(url);
    const text = await response.text();
    const lines = text.trim().split('\n');
    return lines.map(line => line.trim().split(','));
}

async function loadTweets() {
    const tweetsUrl = 'https://docs.google.com/spreadsheets/d/1vEOyjh8bhBfDLe-6BHCQxG3B7NqAmqICtZXA4Cewovc/export?format=csv&gid=1029905748&single=true&output=csv;'
    
    const rawTweets = await fetchCSV(tweetsUrl);
    
    const tweets = rawTweets.slice(1); // Skip header row
    
    const tweetsContainer = document.getElementById('tweets');
    tweets.forEach(([timestamp, username, tweetContent]) => {
        const tweetElement = document.createElement('div');
        tweetElement.className = 'tweet';
        tweetElement.innerHTML = `
            <div class="username">@${username}</div>
            <div class="tweet-content">${tweetContent}</div>
            <div class="datetime">${timestamp}</div>
        `;
        tweetsContainer.appendChild(tweetElement);
    });
}

loadTweets();
