
import fetch from 'node-fetch';
import fs from 'fs';

async function testAppend() {
  const messages = [
    "Hello, this is message 1",
    "This is message 2",
    "And message 3"
  ];

  console.log("Sending 3 messages...");

  for (const msg of messages) {
    const response = await fetch('http://localhost:5000/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        modelId: 'mistral-7b-instruct',
        message: msg,
        conversationId: 'test-session-append'
      })
    });
    const data = await response.json();
    console.log(`Sent: "${msg}", Status: ${response.status}`);
    // Wait a bit to ensure file write completes (though await should handle it, file system might lag slightly)
    await new Promise(r => setTimeout(r, 1000));
  }

  console.log("Finished sending.");
}

testAppend();
