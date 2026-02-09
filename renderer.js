// renderer.js

// --- UI Elements ---
const powerButton = document.getElementById('power-button');
const statusText = document.getElementById('status-text');
const videoElement = document.getElementById('localVideo');
const videoSourceSelect = document.getElementById('video-source');
const currentWordDisplay = document.getElementById('current-word');
const currentSentenceDisplay = document.getElementById('current-sentence');

// --- State Variables ---
let localStream = null;
let ws = null;
let captureInterval = null;
let statusInterval = null;
let isServiceRunning = false;
let isPaused = false; // --- ADDED: Pause state ---
let tempStreamForPermissions = null;

// Local tracking of current word and sentence
let localWord = "";
let localSentence = "";

const FRAME_RATE = 30;

// --- Event Listeners ---
powerButton.addEventListener('click', () => {
  if (isServiceRunning) {
    stopService();
  } else {
    startService();
  }
});

// --- KEYBOARD CONTROLS (Always active when service running) ---
document.addEventListener('keydown', (event) => {
  if (!isServiceRunning || !ws || ws.readyState !== WebSocket.OPEN) {
    return;
  }
  
  // 'P' or 'p' key - Pause/Resume video sending
  if ((event.key === 'p' || event.key === 'P') && !event.repeat) {
    event.preventDefault();
    isPaused = !isPaused; // Toggle the pause state
    if (isPaused) {
      console.log('⏸️ Paused video capture');
      statusText.textContent = "Paused";
      videoElement.classList.add('paused'); // Dim the video for feedback
    } else {
      console.log('▶️ Resumed video capture');
      statusText.textContent = "Connected";
      videoElement.classList.remove('paused');
    }
  }

  // If paused, ignore other gameplay keys
  if (isPaused) return;

  // 'A' or 'a' key - AUTOCORRECT with LLM
  if ((event.key === 'a' || event.key === 'A') && !event.repeat) {
    event.preventDefault();
    console.log('🔤 A key pressed - Autocorrect with LLM');
    ws.send(JSON.stringify({ type: 'AUTOCORRECT', timestamp: Date.now() }));
    flashStatus("Autocorrecting...");
    currentWordDisplay.classList.add('updating');
    setTimeout(() => currentWordDisplay.classList.remove('updating'), 300);
  }
  
  // SPACE key - Add word to sentence
  if (event.code === 'Space' && !event.repeat) {
    event.preventDefault();
    console.log('⌨️  SPACE key pressed - Add word to sentence');
    ws.send(JSON.stringify({ type: 'SPACE', timestamp: Date.now() }));
    flashStatus("Word Added");
    currentSentenceDisplay.classList.add('updating');
    setTimeout(() => currentSentenceDisplay.classList.remove('updating'), 300);
  }

  // ENTER key - Submit sentence for TTS
  if (event.code === 'Enter' && !event.repeat) {
    event.preventDefault();
    console.log('⌨️  ENTER key pressed - Submit sentence');
    ws.send(JSON.stringify({ type: 'SUBMIT_SENTENCE', timestamp: Date.now() }));
    flashStatus("Submitting Sentence...");
    currentSentenceDisplay.classList.add('updating');
    setTimeout(() => currentSentenceDisplay.classList.remove('updating'), 300);
  }

  // Backspace key to delete last character
  if (event.code === 'Backspace' && !event.repeat) {
    event.preventDefault();
    console.log('⌨️  BACKSPACE key pressed - Delete character');
    ws.send(JSON.stringify({ type: 'BACKSPACE', timestamp: Date.now() }));
    flashStatus("Backspace");
    currentWordDisplay.classList.add('updating');
    setTimeout(() => currentWordDisplay.classList.remove('updating'), 300);
  }
});

// --- UI Update Functions ---
function setUIState(isRunning, message = "Stopped") {
  isServiceRunning = isRunning;
  isPaused = false; // Reset pause state on any major state change
  videoElement.classList.remove('paused');

  if (isRunning) {
    statusText.textContent = message;
    statusText.classList.add('active');
    powerButton.classList.add('active');
    videoSourceSelect.disabled = true;
  } else {
    statusText.textContent = message;
    statusText.classList.remove('active');
    powerButton.classList.remove('active');
    videoSourceSelect.disabled = false;
    currentWordDisplay.textContent = "---";
    currentSentenceDisplay.textContent = "---";
    localWord = "";
    localSentence = "";
  }
}

function flashStatus(message) {
  const originalMessage = statusText.textContent;
  // Don't overwrite the "Paused" status with a temporary flash
  if (isPaused) return; 

  statusText.textContent = message;
  setTimeout(() => {
    if (isServiceRunning && !isPaused) {
      statusText.textContent = "Connected";
    }
  }, 800);
}

function updateWordDisplay(word) {
  localWord = word;
  currentWordDisplay.textContent = word || "---";
  console.log(`📝 Word updated: "${word}"`);
}

function updateSentenceDisplay(sentence) {
  localSentence = sentence;
  currentSentenceDisplay.textContent = sentence || "---";
  console.log(`📄 Sentence updated: "${sentence}"`);
  currentSentenceDisplay.scrollTop = currentSentenceDisplay.scrollHeight;
}

// --- Core Logic ---
async function initializeApp() {
  try {
    try {
      tempStreamForPermissions = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      tempStreamForPermissions.getTracks().forEach(track => track.stop());
    } catch (permissionError) {
      console.warn("Could not get initial stream for permissions (this is okay if another app is using the camera).");
    }

    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoDevices = devices.filter(device => device.kind === 'videoinput');
    
    videoSourceSelect.innerHTML = '';
    if (videoDevices.length === 0) {
      throw new Error("No cameras found on this system.");
    }

    videoDevices.forEach(device => {
      const option = document.createElement('option');
      option.value = device.deviceId;
      option.text = device.label || `Camera ${videoSourceSelect.length + 1}`;
      videoSourceSelect.appendChild(option);
    });
    
    const obsOption = Array.from(videoSourceSelect.options).find(o => o.text.toLowerCase().includes('obs virtual camera'));
    if (obsOption) {
      videoSourceSelect.value = obsOption.value;
    } else {
      statusText.textContent = "⚠️ OBS Cam not found";
    }
    
  } catch (err) {
    console.error("CRITICAL ERROR during initialization:", err);
    statusText.textContent = "Error: Camera access denied.";
    powerButton.disabled = true;
  }
}

async function startService() {
  if (isServiceRunning) return;
  setUIState(true, "Starting...");
  
  try {
    const selectedDeviceId = videoSourceSelect.value;
    if (!selectedDeviceId) throw new Error("No camera selected.");
    
    localStream = await navigator.mediaDevices.getUserMedia({
      video: { deviceId: { exact: selectedDeviceId } }
    });
    videoElement.srcObject = localStream;
    
    const canvasElement = document.createElement('canvas');
    canvasElement.width = 640;
    canvasElement.height = 480;
    const canvasContext = canvasElement.getContext('2d');
    
    const devices = await navigator.mediaDevices.enumerateDevices();
    const virtualMicOutput = devices.find(d => d.kind === 'audiooutput' && d.label.includes('CABLE Input'));
    if (!virtualMicOutput) throw new Error("VB-CABLE Input device not found!");

    ws = new WebSocket('ws://localhost:8080/ws/ml');
    
    ws.onopen = () => {
      setUIState(true, "Connected");
      console.log('✅ WebSocket connected');
      console.log('💡 Press P to pause, Backspace to delete');
      
      ws.send(JSON.stringify({ type: 'GET_STATE', timestamp: Date.now() }));
      
      captureInterval = setInterval(() => {
        // Only send frames if not paused
        if (ws?.readyState === WebSocket.OPEN && canvasContext && !isPaused) {
          canvasContext.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);
          ws.send(canvasElement.toDataURL('image/jpeg', 0.5));
        }
      }, 1000 / FRAME_RATE);
      
      statusInterval = setInterval(() => {
        // Only poll for state if not paused
        if (ws?.readyState === WebSocket.OPEN && !isPaused) {
          ws.send(JSON.stringify({ type: 'GET_STATE', timestamp: Date.now() }));
        }
      }, 500);
    };

    ws.onmessage = async (event) => {
        if (typeof event.data === 'string') {
          try {
            const stateUpdate = JSON.parse(event.data);
            
            if (stateUpdate.type === 'STATE_UPDATE') {
              updateWordDisplay(stateUpdate.word);
              updateSentenceDisplay(stateUpdate.sentence);
            } else if (stateUpdate.error) {
              console.error("Server Error:", stateUpdate.error);
            }
          } catch {
            console.error("Server Message:", event.data);
          }
          return;
        }
        
        const audio = new Audio(URL.createObjectURL(event.data));
        try {
          await audio.setSinkId(virtualMicOutput.deviceId);
          audio.play();
          console.log('🔊 Playing audio through VB-CABLE');
        } catch (err) {
          console.error("Failed to route audio:", err);
        }
    };

    ws.onclose = () => { stopService(); setUIState(false, "Server Disconnected"); };
    ws.onerror = (err) => { console.error(err); stopService(); setUIState(false, "Connection Error"); };

  } catch (err) {
    console.error("Error in startService:", err);
    setUIState(false, `Error: ${err.message}`);
    stopService();
  }
}

function stopService() {
  clearInterval(captureInterval);
  clearInterval(statusInterval);
  captureInterval = null;
  statusInterval = null;
  
  if (ws) { ws.close(); ws = null; }
  if (localStream) { 
    localStream.getTracks().forEach(track => track.stop()); 
    localStream = null; 
  }
  if (videoElement) { 
    videoElement.srcObject = null; 
  }
  
  setUIState(false, "Stopped");
}

// --- INITIALIZATION ---
initializeApp();