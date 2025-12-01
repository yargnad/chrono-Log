// We access the global window object instead of importing
// In Tauri v2, the global object structure is window.__TAURI__.core or specific plugins
// Let's try the standard v2 access pattern:

const statusEl = document.getElementById('status');
const imgEl = document.getElementById('monitor-view');
let frameCount = 0;

async function startListener() {
  // Access the 'listen' function from the global object
  // Note: If this fails, we will try window.__TAURI__.event
  const { listen } = window.__TAURI__.event; 

  console.log("Starting listener..."); // Debug log to browser console

  await listen('new-screenshot', (event) => {
    // event.payload contains the base64 data URL string
    imgEl.src = event.payload;
    
    frameCount++;
    statusEl.textContent = `Receiving Stream | Frames Processed: ${frameCount}`;
  });
}

// Wait for the window to load completely before running
window.addEventListener('DOMContentLoaded', () => {
    startListener().catch(console.error);
});