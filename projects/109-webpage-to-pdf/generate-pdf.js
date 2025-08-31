const puppeteer = require('puppeteer');

(async () => {
  const browser = await puppeteer.launch();
  const page = await browser.newPage();
  
  // Navigate to the localhost URL
  await page.goto('http://localhost:8001', { waitUntil: 'networkidle0' });
  await page.pdf({
    path: 'cv.pdf',
    format: 'A4',
    printBackground: true
  });

  await browser.close();
})();
