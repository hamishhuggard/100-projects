const docId = '10Wc1ZwlwrwIzrs70DdTDRRsvsGjIE-3wQN9ubhG4QLI';
document.addEventListener("DOMContentLoaded", function() {
    const docContentElement = document.getElementById('doc-content');

    // Function to fetch and display Google Doc content as HTML
    async function loadGoogleDoc() {
        try {
            const response = await fetch(`https://docs.google.com/document/d/${docId}/export?format=html`);
            // you can also do format=txt
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            const docHtml = await response.text();
            console.log(docHtml);
            docContentElement.innerHTML = docHtml;
        } catch (error) {
            console.error('Error fetching the Google Doc:', error);
            docContentElement.innerHTML = '<p>Failed to load the document.</p>';
        }
    }

    // Load the Google Doc content when the page loads
    loadGoogleDoc();
});
