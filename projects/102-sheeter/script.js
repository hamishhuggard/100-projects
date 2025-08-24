async function fetchCSV(url) {
    const response = await fetch(url);
    const text = await response.text();
    const lines = text.trim().split('\n');
    return lines.map(line => line.trim().split(','));
}

async function loadTweets() {
    const usersUrl  = 'https://docs.google.com/spreadsheets/d/1vEOyjh8bhBfDLe-6BHCQxG3B7NqAmqICtZXA4Cewovc/export?format=csv&gid=0&single=true&output=csv;' // Replace with the actual URL
    const tweetsUrl = 'https://docs.google.com/spreadsheets/d/1vEOyjh8bhBfDLe-6BHCQxG3B7NqAmqICtZXA4Cewovc/export?format=csv&gid=1029905748&single=true&output=csv;'
    
    const [rawTweets, rawUsers] = await Promise.all([fetchCSV(tweetsUrl), fetchCSV(usersUrl)]);
    console.log({rawTweets, rawUsers})
    
    const tweets = rawTweets.slice(1); // Skip header row
    const users = rawUsers.slice(1);   // Skip header row
    
    const userMap = {};
    users.forEach(([handle, username]) => userMap[handle] = username);

    const tweetsContainer = document.getElementById('tweets');
    tweets.forEach(([handle, tweet, datetime, slug]) => {
        console.log({handle, tweet, datetime, slug})
        const tweetElement = document.createElement('div');
        tweetElement.className = 'tweet';
        tweetElement.innerHTML = `
            <div>@${handle}<span class="username"> ${userMap[handle]}</span></div>
            <div>${tweet}</div>
            <div class="datetime">${datetime}</div>
        `;
        tweetsContainer.appendChild(tweetElement);
    });
}

loadTweets();
