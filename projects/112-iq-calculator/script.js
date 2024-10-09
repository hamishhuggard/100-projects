const population = 8_000_000_000; // 8 billion people
const meanIQ = 100;
const stdDev = 15;

function updateIQ(iq) {
    const iqValue = document.getElementById("iqValue");
    const populationCount = document.getElementById("populationCount");
    const oneInN = document.getElementById("oneInN");

    // Display the selected IQ value
    iqValue.textContent = iq;

    // Calculate the z-score
    const zScore = (iq - meanIQ) / stdDev;

    // Calculate the cumulative probability for the z-score (normal distribution)
    const cumulativeProbability = 0.5 * (1 + erf(zScore / Math.SQRT2));

    // Number of people with this IQ or higher
    const numberOfPeople = Math.round(population * (1 - cumulativeProbability));

    // Calculate "one in n"
    const oneIn = Math.round(1 / (1 - cumulativeProbability));

    // Update the webpage with results
    populationCount.textContent = numberOfPeople.toLocaleString();
    oneInN.textContent = oneIn.toLocaleString();
}

// Error function approximation for the normal distribution
function erf(x) {
    const sign = x >= 0 ? 1 : -1;
    x = Math.abs(x);
    const a1 =  0.254829592;
    const a2 = -0.284496736;
    const a3 =  1.421413741;
    const a4 = -1.453152027;
    const a5 =  1.061405429;
    const p  =  0.3275911;

    const t = 1.0 / (1.0 + p * x);
    const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
    
    return sign * y;
}

// Initialize with default IQ of 100
updateIQ(100);
