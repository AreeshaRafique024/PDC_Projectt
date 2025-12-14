
import fetch from 'node-fetch';
import fs from 'fs';
import path from 'path';

async function testLogging() {
  console.log('Sending first request...');
  await sendRequest('test-session-1', 'Hello 1');
  await checkRowCount();

  console.log('Sending second request...');
  await sendRequest('test-session-1', 'Hello 2');
  await checkRowCount();
}

async function sendRequest(sessionId, message) {
  try {
    const response = await fetch('http://localhost:5000/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
        modelId: 'mistral-7b-instruct', 
        message: message,
        conversationId: sessionId
        })
    });
    const data = await response.json();
    console.log('Response status:', response.status);
  } catch (e) {
    console.error("Error sending request:", e);
  }
}

async function checkRowCount() {
  await new Promise(resolve => setTimeout(resolve, 1000)); // Wait 1s
  const filePath = path.join(process.cwd(), 'data', 'metrics.xlsx');
  if (!fs.existsSync(filePath)) {
    console.log('File does not exist yet.');
    return;
  }
  const stats = fs.statSync(filePath);
  console.log(`File size: ${stats.size}, Modified: ${stats.mtime.toISOString()}`);

  const workbook = new ExcelJS.Workbook();
  await workbook.xlsx.readFile(filePath);
  const worksheet = workbook.getWorksheet("Metrics");
  if (worksheet) {
    console.log(`Current Row Count: ${worksheet.rowCount}`);
    const lastRow = worksheet.getRow(worksheet.rowCount);
    console.log(`Last Row ID: ${lastRow.getCell(1).value}`);
  } else {
    console.log('Worksheet not found.');
  }
}

// Import ExcelJS dynamically or assume it's available since we are in module
import ExcelJS from 'exceljs';

testLogging();
