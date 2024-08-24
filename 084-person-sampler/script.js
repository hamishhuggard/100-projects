// Load the CSV data from the Google Sheets URL
const url = 'https://docs.google.com/spreadsheets/d/1A9mAwXHEJJ9RfICkUTvIhtm7jlET06wDwCjZTOh_-40/export?format=csv&gid=0&output=csv;' 
d3.csv(url).then(function(data) {

    // Convert population values to integers, removing commas
    data.forEach(function(d) {
        d.Population = +d.Population.replace(/,/g, '');
    });

    // Event listener for the button
    document.getElementById("select-country").addEventListener("click", function() {
        let selectedCountry = selectRandomCountry(data);
        document.getElementById("result").innerText = selectedCountry.Location;
    });
});

// Function to select a random country weighted by population
function selectRandomCountry(data) {
    let totalPopulation = d3.sum(data, d => d.Population);
    let randomValue = Math.random() * totalPopulation;
    let cumulativeSum = 0;

    for (let i = 0; i < data.length; i++) {
        cumulativeSum += data[i].Population;
        if (randomValue <= cumulativeSum) {
            return data[i];
        }
    }
}
