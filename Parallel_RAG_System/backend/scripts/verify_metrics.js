
import ExcelJS from 'exceljs';
import path from 'path';

const METRICS_FILE = '/home/faizancyber/Pictures/LLM_Comparison/backend/data/metrics.xlsx';

async function verifyFile() {
  const workbook = new ExcelJS.Workbook();
  await workbook.xlsx.readFile(METRICS_FILE);
  const worksheet = workbook.getWorksheet("Metrics");
  
  if (!worksheet) {
    console.log("Worksheet 'Metrics' not found!");
    return;
  }

  const rowCount = worksheet.rowCount;
  console.log(`Total rows in file: ${rowCount}`);
  
  // Print last 3 rows to see if they match our test messages
  console.log("Last 3 rows prompts:");
  for (let i = rowCount - 2; i <= rowCount; i++) {
     if (i > 1) { // Skip header
        const row = worksheet.getRow(i);
        // Column 5 is Prompt based on my code
        console.log(`Row ${i}: ${row.getCell(5).value}`);
     }
  }
}

verifyFile();
