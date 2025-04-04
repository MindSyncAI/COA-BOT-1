// Theme Toggle Functionality
const themeToggleBtn = document.getElementById('themeToggle');
const themeIcon = themeToggleBtn.querySelector('i');
const themeText = themeToggleBtn.querySelector('span');

themeToggleBtn.addEventListener('click', () => {
    document.body.classList.toggle('light-theme');
    
    if (document.body.classList.contains('light-theme')) {
        themeIcon.className = 'fas fa-sun';
        themeText.textContent = 'Light';
    } else {
        themeIcon.className = 'fas fa-moon';
        themeText.textContent = 'Dark';
    }
});

// Create circuit board background elements
function createCircuitElements() {
    const backgroundAnimation = document.getElementById('backgroundAnimation');
    
    // Create circuit nodes and traces
    const nodes = [];
    
    // Create nodes (connection points)
    for (let i = 0; i < 30; i++) {
        const node = document.createElement('div');
        node.className = 'circuit-node';
        
        // Position nodes in a grid-like pattern to resemble circuit board
        // Use modulo to create rows and columns
        const row = Math.floor(i / 6);
        const col = i % 6;
        
        // Add some randomness to the grid for more organic feel
        const x = (col * 18) + Math.random() * 5 + 5;
        const y = (row * 18) + Math.random() * 5 + 5;
        
        node.style.left = `${x}vw`;
        node.style.top = `${y}vh`;
        node.style.animationDelay = `${Math.random() * 4}s`;
        backgroundAnimation.appendChild(node);
        nodes.push({ element: node, x, y });
    }
    
    // Create circuit traces (connections between nodes)
    // For a circuit board look, we'll create more horizontal and vertical lines
    for (let i = 0; i < nodes.length; i++) {
        const sourceNode = nodes[i];
        
        // Connect to 2-3 nearby nodes to create circuit pattern
        const connectionsCount = 2 + Math.floor(Math.random() * 2);
        
        for (let j = 0; j < connectionsCount; j++) {
            // Find a node that's relatively close (preferably in same row or column)
            // to create more grid-like circuit traces
            let targetIndex;
            const sameRowCol = Math.random() > 0.5;
            
            if (sameRowCol) {
                // Same row (horizontal trace)
                const row = Math.floor(i / 6);
                const possibleTargets = nodes.filter((node, idx) => 
                    Math.floor(idx / 6) === row && idx !== i
                );
                if (possibleTargets.length > 0) {
                    targetIndex = nodes.indexOf(possibleTargets[Math.floor(Math.random() * possibleTargets.length)]);
                } else {
                    targetIndex = Math.floor(Math.random() * nodes.length);
                }
            } else {
                // Same column (vertical trace)
                const col = i % 6;
                const possibleTargets = nodes.filter((node, idx) => 
                    idx % 6 === col && idx !== i
                );
                if (possibleTargets.length > 0) {
                    targetIndex = nodes.indexOf(possibleTargets[Math.floor(Math.random() * possibleTargets.length)]);
                } else {
                    targetIndex = Math.floor(Math.random() * nodes.length);
                }
            }
            
            if (targetIndex !== i) {
                const targetNode = nodes[targetIndex];
                
                // Calculate distance and angle
                const dx = targetNode.x - sourceNode.x;
                const dy = targetNode.y - sourceNode.y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                const angle = Math.atan2(dy, dx) * 180 / Math.PI;
                
                // Create circuit trace line
                const trace = document.createElement('div');
                trace.className = 'circuit-trace';
                trace.style.left = `${sourceNode.x}vw`;
                trace.style.top = `${sourceNode.y}vh`;
                trace.style.width = `${distance}vw`;
                trace.style.transform = `rotate(${angle}deg)`;
                trace.style.animationDelay = `${Math.random() * 5}s`;
                backgroundAnimation.appendChild(trace);

                // Add data pulses traveling along this trace
                if (Math.random() > 0.3) {
                    for (let p = 0; p < 2; p++) {
                        const pulse = document.createElement('div');
                        pulse.className = 'data-pulse';
                        pulse.style.left = `${sourceNode.x}vw`;
                        pulse.style.top = `${sourceNode.y}vh`;
                        pulse.style.setProperty('--travel-distance', `${dx}vw`);
                        pulse.style.setProperty('--travel-height', `${dy}vh`);
                        pulse.style.animationDelay = `${p * 2.5 + Math.random()}s`;
                        backgroundAnimation.appendChild(pulse);
                    }
                }
            }
        }
    }
    
    // Create digital waves (representing binary signals)
    for (let i = 0; i < 5; i++) {
        const digitalWave = document.createElement('div');
        digitalWave.className = 'digital-wave';
        
        // Position at a node
        const nodeIndex = Math.floor(Math.random() * nodes.length);
        const node = nodes[nodeIndex];
        digitalWave.style.left = `${node.x}vw`;
        digitalWave.style.top = `${node.y}vh`;
        
        // Random width between 50px and 150px
        const width = 50 + Math.random() * 100;
        digitalWave.style.width = `${width}px`;
        digitalWave.style.height = `3px`;
        
        // Animation timing
        digitalWave.style.animationDelay = `${Math.random() * 3}s`;
        
        backgroundAnimation.appendChild(digitalWave);
    }
    
    // Create processor glowing elements
    for (let i = 0; i < 4; i++) {
        const processorGlow = document.createElement('div');
        processorGlow.className = 'processor-glow';
        
        // Size between 50px and 100px
        const size = 50 + Math.random() * 50;
        processorGlow.style.width = `${size}px`;
        processorGlow.style.height = `${size}px`;
        
        // Position near a random node
        const nodeIndex = Math.floor(Math.random() * nodes.length);
        const node = nodes[nodeIndex];
        const offsetX = -25 + Math.random() * 50;
        const offsetY = -25 + Math.random() * 50;
        processorGlow.style.left = `calc(${node.x}vw + ${offsetX}px)`;
        processorGlow.style.top = `calc(${node.y}vh + ${offsetY}px)`;
        
        // Animation timing
        processorGlow.style.animationDuration = `${8 + Math.random() * 7}s`;
        processorGlow.style.animationDelay = `${Math.random() * 5}s`;
        
        backgroundAnimation.appendChild(processorGlow);
    }
}

// Initialize the background animations
createCircuitElements();

// Chat functionality
const messageInput = document.getElementById('messageInput');
const sendButton = document.getElementById('sendButton');
const chatMessages = document.getElementById('chatMessages');
const chips = document.querySelectorAll('.chip');

function sendMessage() {
    const message = messageInput.value.trim();
    if (message === '') return;
    
    // Add user message
    const userMessageDiv = document.createElement('div');
    userMessageDiv.className = 'message user-message';
    userMessageDiv.textContent = message;
    chatMessages.appendChild(userMessageDiv);
    
    // Clear input
    messageInput.value = '';
    
    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    // Show typing indicator
    const typingIndicator = document.createElement('div');
    typingIndicator.className = 'typing-indicator';
    for (let i = 0; i < 3; i++) {
        const dot = document.createElement('div');
        dot.className = 'typing-dot';
        typingIndicator.appendChild(dot);
    }
    chatMessages.appendChild(typingIndicator);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    // Send message to backend
    fetch('/ask', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: message })
    })
    .then(response => response.json())
    .then(data => {
        // Remove typing indicator
        chatMessages.removeChild(typingIndicator);
        
        // Add bot response with markdown rendering
        const botResponseDiv = document.createElement('div');
        botResponseDiv.className = 'message bot-message markdown-content';
        botResponseDiv.innerHTML = marked.parse(data.answer);
        chatMessages.appendChild(botResponseDiv);
        
        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
    })
    .catch(error => {
        console.error('Error:', error);
        // Remove typing indicator
        chatMessages.removeChild(typingIndicator);
        
        // Add error message
        const errorDiv = document.createElement('div');
        errorDiv.className = 'message bot-message error';
        errorDiv.textContent = 'Sorry, there was an error processing your request. Please try again.';
        chatMessages.appendChild(errorDiv);
        
        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
    });
}

// Event listeners
sendButton.addEventListener('click', sendMessage);

messageInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        sendMessage();
    }
});

chips.forEach(chip => {
    chip.addEventListener('click', () => {
        messageInput.value = chip.textContent;
        sendMessage();
    });
});