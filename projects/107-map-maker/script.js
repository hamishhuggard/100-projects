const viewportWidth = window.innerWidth;
const viewportHeight = window.innerHeight;
const viewportRatio = viewportHeight / viewportWidth;
const viewportWidthInGridWidths = viewportRatio > 1 ? 21 : 59;

const CSV_URL = 'https://docs.google.com/spreadsheets/d/15Azis63nHoKSljVMy7lliSF1XXsVCp2PyOV_7XmScLs/export?format=csv&gid=0&single=true&output=csv'

/*
const zoom = d3.zoom()
    .scaleExtent([0.8, 5])
    .on('zoom', zoomed);

const svg = d3.select('#grid')
    .append('svg')
    .attr('width', viewportWidth)
    .attr('height', viewportHeight)
    .call(zoom)
    .append('g');

rangeX.forEach(x => {
  gridGroup.append('line')
    .attr('x1', x)
    .attr('y1', 0)
    .attr('x2', x)
    .attr('y2', gridHeight)
    .attr('class', 'grid-line');
});

rangeY.forEach(y => {
  gridGroup.append('line')
    .attr('x1', 0)
    .attr('y1', y)
    .attr('x2', gridWidth)
    .attr('y2', y)
    .attr('class', 'grid-line');
});

rangeX.forEach((x, i) => {
  rangeY.forEach((y, j) => {
    gridGroup.append('text')
      .attr('x', x + 2)
      .attr('y', y + 12)
      .attr('class', 'grid-text')
      .text(`${i},${j}`);
  });
});

function zoomed(event) {
    svg.attr('transform', event.transform);
}
*/


d3.text(CSV_URL).then(function(data) {
    console.log('hello')
    console.log({data});
    let grid = data.split("\n").map(line => line.split(","));
    console.log(grid);
});

/*
d3.csv(CSV_URL)
    .then(data => {
        console.log(data);
        //updateInfoBox(data); // Add this line
        //data.forEach(drawSquare);
        //setMapItemOrigins();
    });
*/

/*
// Add the tooltip
const tooltip = d3.select('#tooltip');

function drawSquare(row) {
// Discard rows with 'x' in the 'hide' column
if (row.hide === 'x') return;

const x = +row.x;
const y = +row.y;
const scale = (+row.scale || 1) * 0.55;
let logo = row.logo;
const label = row.Label;
const link = row.Link;
const longLabel = row.LongLabel;
const description = row.Description;

// Calculate the position in pixels
const xPos = x * gridSize + xOffset;
const yPos = y * gridSize;

// Create a group for the logo, label, and link
const itemGroup = svg.append('g')
  .attr('transform', `translate(${xPos}, ${yPos})`);

// Create a link
const itemLink = itemGroup.append('a')
  .attr('href', link)
  .attr('target', '_blank');

// Create a group for the logo and label, and apply hover effect to it
const contentGroup = itemLink.append('g')
  .attr('class', 'mapItem')
  .on('mouseover', function (event, row) {
    tooltip
      .style('left', (event.pageX + 10) + 'px')
      .style('top', (event.pageY + 10) + 'px')
      .html(`<strong>${longLabel}</strong><br>${description}`)
      .classed('hidden', false);
  })
  .on('mouseout', function () {
    tooltip.classed('hidden', true);
  });

// Replace logo URL with processed version
const PREFIX = '/logos/'
const extension = '.' + logo.split('.').pop()
if (avifIsSupported && logo.startsWith(PREFIX) && processedImageFormats.includes(extension)) {
  const name = logo.substring(PREFIX.length, logo.length - extension.length)
  logo = `${PREFIX}avif/${name}.avif`
}

// Add the label
const labelText = label;
const labelMaxWidth = gridSize * scale * 2;
const fontsize = gridSize * scale * 0.3;

const itemContainer = contentGroup.append('foreignObject')
  .attr('x', gridSize * scale / 2 - labelMaxWidth / 2)
  .attr('y', 0)
  .attr('width', labelMaxWidth)
  .attr('font-size', fontsize)
  .attr('height', gridSize * scale + 5 + 30);

const outerDiv = itemContainer.append('xhtml:div')
  .attr('class', 'labelContainer')
  .style('align-items', 'center');

outerDiv.append('xhtml:img')
  .attr('alt', 'Image description')
  .attr('src', logo)
  .attr('style', `max-width: ${labelMaxWidth*0.9}px; max-height: ${labelMaxWidth*0.5}px`);

outerDiv.append('xhtml:div')
  .text(labelText);

}

function setMapItemOrigins() {
// Set transform-origin dynamically for each square
d3.selectAll('.mapItem')
  .each(function () {
    const xPos = d3.select(this).attr('data-x');
    const yPos = d3.select(this).attr('data-y');
    d3.select(this).style('transform-origin', `${xPos}px ${yPos}px`);
  });
}
*/


window.addEventListener('resize', () => {
  d3.select('#grid svg')
    .attr('width', window.innerWidth)
    .attr('height', window.innerHeight);
});
