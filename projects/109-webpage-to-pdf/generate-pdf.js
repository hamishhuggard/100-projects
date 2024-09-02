const puppeteer = require('puppeteer');

(async () => {
  const browser = await puppeteer.launch();
  const page = await browser.newPage();
  //await page.goto('file:///path/to/your/cv.html', {waitUntil: 'networkidle0'});
  await page.goto('../083-cv/index.html', {waitUntil: 'networkidle0'});
  await page.pdf({
    path: 'cv.pdf',
    format: 'A4',
    printBackground: true
  });

  await browser.close();
})();
