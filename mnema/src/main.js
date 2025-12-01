const statusEl = document.getElementById('status');
const imgEl = document.getElementById('monitor-view');
const searchInput = document.getElementById('search-input');
const resultsContainer = document.getElementById('results-container');
let frameCount = 0;

// --- CORE STREAMING LOGIC (Keep existing) ---
async function startListener() {
    console.log("Starting listener...");

    const { listen } = window.__TAURI__.event;

    await listen('new-screenshot', (event) => {
        imgEl.src = event.payload;
        frameCount++;
        statusEl.textContent = `Mnemosyne Stream | Frames Processed: ${frameCount}`;
    });
}

// --- NEW: SEARCH LOGIC ---
async function search(query) {
    if (!query) return;

    // Reset UI
    resultsContainer.innerHTML = ''; 
    statusEl.textContent = `Mnemosyne is searching for "${query}"...`;
    
    // Call the Rust function
    try {
        const results = await window.__TAURI__.core.invoke('search_memories', { query }); 

        statusEl.textContent = `Search complete. Found ${results.length} memories.`;
        
        // Render Results
        if (results.length > 0) {
            
            // Note: We need a utility function later to load the image file path directly, 
            // but for now, we'll display the metadata.
            const list = document.createElement('ul');
            list.innerHTML = results.map(r => `
                <li>
                    <strong>Memory:</strong> ${r.id.substring(0, 8)}...<br>
                    <strong>Timestamp:</strong> ${r.timestamp}<br>
                    <strong>Relevance Score:</strong> ${r.score.toFixed(4)}
                </li>
            `).join('');
            resultsContainer.appendChild(list);

        } else {
            resultsContainer.textContent = "No relevant memories found.";
        }

        // Auto-close search area after 5 seconds of display
        setTimeout(() => {
            statusEl.textContent = `Mnemosyne Stream | Frames Processed: ${frameCount}`;
            resultsContainer.innerHTML = '';
        }, 5000);

    } catch (error) {
        console.error("Search failed:", error);
        statusEl.textContent = `Search Error! Check console.`;
    }
}

// --- INITIALIZATION ---
window.addEventListener('DOMContentLoaded', () => {
    startListener().catch(console.error);
    
    // Attach search event listener to the button
    const searchButton = document.getElementById('search-button');
    if (searchButton) {
        searchButton.addEventListener('click', () => {
            const query = searchInput.value;
            search(query);
        });
    }

    // Example of a helpful utility function (Needs new Tauri permissions)
    // getCurrent().setTitle('Mnema · Ready to Serve the Rebellion');
});