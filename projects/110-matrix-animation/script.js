function generateMatrix(containerId, matrixData) {
    const matrix = d3.select(containerId).append("div").attr("class", "matrix");

    matrixData.forEach(row => {
        const rowDiv = matrix.append("div").attr("class", "row");
        row.forEach(value => {
            rowDiv.append("div").attr("class", "cell").text(value);
        });
    });
    return matrix;
}

const randomMatrixData = Array.from({ length: 4 }, () => 
    Array.from({ length: 3 }, () => Math.floor(Math.random() * 19) - 9)
);
const randomMatrix = generateMatrix("#random-matrix", randomMatrixData);


// Sparse Matrix (4x3)
const sparseMatrixData = [
    [1, 0, -1, 0],
    [0, 1, 0, 1],
    [-1, 0, 1, 1],
];
const sparseMatrix = generateMatrix("#sparse-matrix", sparseMatrixData);

function setPositionAndOpacity(matrix, x, y, opacity) {
    matrix.style("transform", `translate(${x}px, ${y}px)`)
        .style("opacity", opacity);
}

setPositionAndOpacity(randomMatrix, 0, -15, 1);
setPositionAndOpacity(sparseMatrix, 0, 0, 1);

setTimeout(() => {
    setPositionAndOpacity(sparseMatrix, 0, -120, 1);
}, 1000);

setTimeout(() => {
    setPositionAndOpacity(sparseMatrix, 0, -100, 0);
}, 2000);
