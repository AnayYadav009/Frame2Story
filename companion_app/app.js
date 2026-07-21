// State Management
let videoFile = null;
let subtitleTimeline = [];
let currentSubtitleIndex = -1;
let chatHistory = [];
let isGenerating = false;
let serverHasApiKey = false;

// HTML sanitizer to prevent XSS injection via subtitles, errors, or API responses
function escapeHtml(str) {
    const div = document.createElement('div');
    div.appendChild(document.createTextNode(str));
    return div.innerHTML;
}

// Truncate filename with ellipsis only when needed
function truncateFilename(name, maxLen = 15) {
    return name.length > maxLen ? name.substring(0, maxLen) + '...' : name;
}

// DOM Elements
const videoInput = document.getElementById('video-input');
const subtitleInput = document.getElementById('subtitle-input');
const videoPlayer = document.getElementById('video-player');
const videoPlaceholder = document.getElementById('video-placeholder');
const subtitleDisplay = document.getElementById('subtitle-display');
const subtitleStatus = document.getElementById('subtitle-status');
const chatHistoryContainer = document.getElementById('chat-history');
const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');
const clearChatBtn = document.getElementById('clear-chat-btn');
const contextInfo = document.getElementById('context-info');
const openSettingsBtn = document.getElementById('open-settings-btn');
const closeSettingsBtn = document.getElementById('close-settings-btn');
const settingsModal = document.getElementById('settings-modal');
const saveSettingsBtn = document.getElementById('save-settings-btn');
const apiKeyInput = document.getElementById('api-key-input');
const promptStyleSelect = document.getElementById('prompt-style-select');
const visionFramesSelect = document.getElementById('vision-frames-select');
const videoLabel = document.getElementById('video-label');
const subtitleLabel = document.getElementById('subtitle-label');
const volumeToggleBtn = document.getElementById('volume-toggle-btn');
const volumeIcon = document.getElementById('volume-icon');
const volumeText = document.getElementById('volume-text');

// Load Settings & Check Server API Key on Init
document.addEventListener('DOMContentLoaded', async () => {
    const savedApiKey = localStorage.getItem('GEMINI_API_KEY');
    const savedPersona = localStorage.getItem('CINEBUDDY_PERSONA');
    const savedFrames = localStorage.getItem('CINEBUDDY_VISION_FRAMES');
    
    if (savedApiKey) {
        apiKeyInput.value = savedApiKey;
    }
    if (savedPersona) {
        promptStyleSelect.value = savedPersona;
    }
    if (savedFrames && visionFramesSelect) {
        visionFramesSelect.value = savedFrames;
    }

    // Check if server loaded API key from .env file
    try {
        const res = await fetch('/api/config');
        const config = await res.json();
        if (config.hasApiKey) {
            serverHasApiKey = true;
            if (!apiKeyInput.value) {
                apiKeyInput.placeholder = "API Key detected in server .env file";
            }
        }
    } catch (e) {
        console.warn("Could not check server API config:", e);
    }
    
    // Auto-scroll chat to bottom
    scrollToBottom();
});

// Event Listeners for File Picking
videoInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (file) {
        videoFile = file;
        const fileUrl = URL.createObjectURL(file);
        videoPlayer.src = fileUrl;
        videoPlayer.muted = false;
        videoPlayer.volume = 1.0;
        videoPlayer.style.display = 'block';
        videoPlaceholder.style.display = 'none';
        videoLabel.innerHTML = `<i class="fa-solid fa-video"></i> ${truncateFilename(file.name)}`;
        contextInfo.textContent = `Video: ${escapeHtml(file.name)}`;
        
        // Show volume toggle button
        if (volumeToggleBtn) {
            volumeToggleBtn.style.display = 'inline-flex';
            volumeIcon.className = 'fa-solid fa-volume-high';
            volumeText.textContent = 'Mute';
        }
        
        // Reset chat history when loading a new video
        resetChat();
        enableChatInputIfReady();

        // Register video file with server for high-speed FFmpeg audio extraction
        if (subtitleTimeline.length === 0) {
            subtitleDisplay.innerHTML = `<p class="empty-subtitle-msg"><i class="fa-solid fa-spinner fa-spin"></i> Linking video with AI Audio Engine...</p>`;
        }

        try {
            // Register video path with server
            const regRes = await fetch('/api/register-video', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filename: file.name })
            });
            let regData = await regRes.json();

            if (!regData.registeredPath) {
                // Upload video file to server if not found in local directories
                subtitleDisplay.innerHTML = `<p class="empty-subtitle-msg"><i class="fa-solid fa-spinner fa-spin"></i> Processing video file for AI Audio Listening...</p>`;
                const uploadRes = await fetch('/api/register-video', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/octet-stream' },
                    body: file
                });
                regData = await uploadRes.json();
            }

            if (subtitleTimeline.length === 0) {
                subtitleStatus.textContent = 'AI Audio Listening Active';
                subtitleStatus.className = 'status-badge active';
                const shortPath = regData.registeredPath ? regData.registeredPath.split(/[\\/]/).pop() : file.name;
                subtitleDisplay.innerHTML = `<p class="empty-subtitle-msg">✨ Video audio linked: <strong>${escapeHtml(shortPath)}</strong>. Ask questions anytime without needing an .srt file!</p>`;
            }
        } catch (err) {
            console.warn("Video server registration warning:", err);
            if (subtitleTimeline.length === 0) {
                subtitleDisplay.innerHTML = `<p class="empty-subtitle-msg">Video loaded (${escapeHtml(file.name)}). Ask questions anytime!</p>`;
            }
        }
    }
});

subtitleInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(event) {
            const text = event.target.result;
            subtitleTimeline = parseSubtitles(text);
            
            subtitleStatus.textContent = 'Subtitles Loaded';
            subtitleStatus.className = 'status-badge active';
            subtitleLabel.innerHTML = `<i class="fa-solid fa-closed-captioning"></i> ${truncateFilename(file.name)}`;
            
            // Show initial status message in subtitle pane
            subtitleDisplay.innerHTML = `<p class="empty-subtitle-msg">Synced: 0 of ${subtitleTimeline.length} lines loaded.</p>`;
            
            enableChatInputIfReady();
        };
        reader.readAsText(file);
    }
});

// Settings Modal Events
openSettingsBtn.addEventListener('click', () => settingsModal.classList.add('open'));
closeSettingsBtn.addEventListener('click', () => settingsModal.classList.remove('open'));
saveSettingsBtn.addEventListener('click', () => {
    localStorage.setItem('GEMINI_API_KEY', apiKeyInput.value.trim());
    localStorage.setItem('CINEBUDDY_PERSONA', promptStyleSelect.value);
    if (visionFramesSelect) {
        localStorage.setItem('CINEBUDDY_VISION_FRAMES', visionFramesSelect.value);
    }
    settingsModal.classList.remove('open');
    addSystemMessage("Settings saved successfully!");
});

// Clear Chat Event
clearChatBtn.addEventListener('click', () => {
    resetChat();
});

// Subtitle Parser (SRT & VTT support)
function parseSubtitles(data) {
    // Normalize newlines
    data = data.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
    
    // Split by double newline to isolate blocks
    const blocks = data.split('\n\n');
    const parsed = [];
    
    for (let block of blocks) {
        block = block.trim();
        if (!block) continue;
        
        const lines = block.split('\n');
        if (lines.length < 2) continue;
        
        let timeIndex = -1;
        for (let i = 0; i < lines.length; i++) {
            if (lines[i].includes('-->')) {
                timeIndex = i;
                break;
            }
        }
        
        if (timeIndex === -1) continue;
        
        const timePart = lines[timeIndex];
        const times = timePart.split('-->');
        if (times.length !== 2) continue;
        
        const start = timeToSeconds(times[0].trim());
        const end = timeToSeconds(times[1].trim());
        
        // Extract text content (omit HTML tags like <i>)
        const text = lines.slice(timeIndex + 1).join(' ').replace(/<[^>]*>/g, '').trim();
        if (text) {
            parsed.push({ start, end, text });
        }
    }
    
    return parsed.sort((a, b) => a.start - b.start);
}

// Convert HH:MM:SS,mmm or MM:SS.mmm to seconds
function timeToSeconds(timeStr) {
    timeStr = timeStr.replace(',', '.'); // Normalize decimals
    const parts = timeStr.split(':');
    let hrs = 0, mins = 0, secs = 0;
    
    if (parts.length === 3) {
        hrs = parseFloat(parts[0]);
        mins = parseFloat(parts[1]);
        secs = parseFloat(parts[2]);
    } else if (parts.length === 2) {
        mins = parseFloat(parts[0]);
        secs = parseFloat(parts[1]);
    } else {
        secs = parseFloat(parts[0]);
    }
    
    return hrs * 3600 + mins * 60 + secs;
}

// Track Video Playback for Syncing Subtitles
videoPlayer.addEventListener('timeupdate', () => {
    const currentTime = videoPlayer.currentTime;
    
    // Find active subtitle
    let activeIndex = -1;
    for (let i = 0; i < subtitleTimeline.length; i++) {
        if (currentTime >= subtitleTimeline[i].start && currentTime <= subtitleTimeline[i].end) {
            activeIndex = i;
            break;
        }
    }
    
    if (activeIndex !== currentSubtitleIndex) {
        currentSubtitleIndex = activeIndex;
        if (activeIndex !== -1) {
            subtitleDisplay.innerHTML = `<p class="current-subtitle">${escapeHtml(subtitleTimeline[activeIndex].text)}</p>`;
        } else {
            // Find the most recent past subtitle
            let pastIndex = -1;
            for (let i = subtitleTimeline.length - 1; i >= 0; i--) {
                if (currentTime >= subtitleTimeline[i].end) {
                    pastIndex = i;
                    break;
                }
            }
            if (pastIndex !== -1) {
                subtitleDisplay.innerHTML = `<p class="empty-subtitle-msg">(Dialogue ended: "${escapeHtml(subtitleTimeline[pastIndex].text)}")</p>`;
            } else {
                subtitleDisplay.innerHTML = `<p class="empty-subtitle-msg">Playback running... (No dialogue yet)</p>`;
            }
        }
    }
    
    // Update sidebar context text
    const formattedTime = formatTime(currentTime);
    contextInfo.textContent = `Active | ${formattedTime}`;
});

// Format seconds into MM:SS
function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

// Enable chat controls if inputs are ready
function enableChatInputIfReady() {
    if (videoFile) {
        chatInput.disabled = false;
        sendBtn.disabled = false;
        chatInput.placeholder = "Ask CineBuddy about the movie...";
    }
}

// Resets/Clears Chat
function resetChat() {
    chatHistory = [];
    chatHistoryContainer.innerHTML = `
        <div class="message system">
            <div class="msg-content">
                Chat cleared. Pause the video at any time to ask context-specific questions!
            </div>
        </div>
    `;
}

// Append System Message
function addSystemMessage(text) {
    const msg = document.createElement('div');
    msg.className = 'message system';
    msg.innerHTML = `<div class="msg-content">${text}</div>`;
    chatHistoryContainer.appendChild(msg);
    scrollToBottom();
}

// Chat Input Event Handlers
chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

sendBtn.addEventListener('click', sendMessage);

// Send Message Flow
async function sendMessage() {
    const question = chatInput.value.trim();
    if (!question || isGenerating) return;
    
    // Check for API Key (either in localStorage or in server .env)
    const apiKey = localStorage.getItem('GEMINI_API_KEY') || '';
    if (!apiKey && !serverHasApiKey) {
        addSystemMessage("❌ Error: Please configure your Gemini API Key in your .env file or in the settings (top-right gear icon) first!");
        return;
    }
    
    // Pause video when user interacts with chat
    if (!videoPlayer.paused) {
        videoPlayer.pause();
        addSystemMessage(`⏸️ Video paused automatically at ${formatTime(videoPlayer.currentTime)} to analyze dialogue.`);
    }
    
    // Add user message to UI
    appendMessage('user', question);
    chatInput.value = '';
    
    // Set loading state
    isGenerating = true;
    sendBtn.disabled = true;
    chatInput.disabled = true;
    
    const currentTime = videoPlayer.currentTime;
    const spokenTranscript = getTranscriptUpTo(currentTime);
    
    // Prepare API prompt
    const persona = localStorage.getItem('CINEBUDDY_PERSONA') || 'helpful';
    const systemInstruction = getSystemInstruction(persona, currentTime, spokenTranscript);
    
    // Construct user parts
    const currentUserParts = [{ text: question }];

    // Format payload
    const formattedHistory = chatHistory.map(msg => ({
        role: msg.role === 'assistant' ? 'model' : 'user',
        parts: msg.parts
    }));

    const contents = [...formattedHistory, { role: 'user', parts: currentUserParts }];
    
    // Append temporary loading model message
    const loadingMsg = appendMessage('assistant', '<i class="fa-solid fa-spinner fa-spin"></i> CineBuddy is analyzing audio context & thinking...');
    
    const numFrames = parseInt(localStorage.getItem('CINEBUDDY_VISION_FRAMES') || (visionFramesSelect ? visionFramesSelect.value : '5'), 10);
    const enableVision = numFrames > 0;

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                ...(serverHasApiKey ? {} : { apiKey: apiKey }),
                contents: contents,
                systemInstruction: systemInstruction,
                timestamp: currentTime,
                hasSubtitles: subtitleTimeline.length > 0,
                enableVision: enableVision,
                numFrames: numFrames
            })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            const reply = data.candidates[0].content.parts[0].text;
            // Update loading message with response
            loadingMsg.innerHTML = `<div class="msg-content">${formatResponse(reply)}</div><span class="timestamp">${new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}</span>`;
            
            // Record in state
            chatHistory.push({ role: 'user', parts: [{ text: question }] });
            chatHistory.push({ role: 'assistant', parts: [{ text: reply }] });
        } else {
            throw new Error(data.error?.message || data.error || 'Failed to generate response.');
        }
    } catch (err) {
        loadingMsg.innerHTML = `<div class="msg-content" style="color: #ef4444;"><i class="fa-solid fa-triangle-exclamation"></i> Error: ${escapeHtml(err.message)}</div>`;
    } finally {
        isGenerating = false;
        enableChatInputIfReady();
        scrollToBottom();
    }
}

// Generate transcript subset up to timestamp
function getTranscriptUpTo(timestamp) {
    const lines = subtitleTimeline.filter(item => item.start <= timestamp);
    return lines.map(line => `[${formatTime(line.start)}] ${line.text}`).join('\n');
}

// Prompt Engineering template
function getSystemInstruction(persona, timestamp, transcript) {
    let personaStr = 'helpful and friendly film companion';
    if (persona === 'analytical') {
        personaStr = 'deep cinematic analyzer who focuses on themes, visual metaphors, and narrative structure';
    } else if (persona === 'funny') {
        personaStr = 'sarcastic, funny movie buddy who cracks jokes about the characters and plot';
    }
    
    const contextStr = transcript 
        ? `Dialogue Transcript so far:\n${transcript}` 
        : `(Video Audio recording slice and Visual video keyframe images attached up to timestamp [${formatTime(timestamp)}])`;

    return `You are CineBuddy, a ${personaStr}. The user is watching a video and has paused at playback time [${formatTime(timestamp)}].

CRITICAL SAFETY & REASONING RULES:
1. You are provided with BOTH spoken audio recording slices AND visual video keyframe images leading up to paused timestamp [${formatTime(timestamp)}].
2. Use both the visual information (on-screen action, character expressions, props, visual text, fight/silent scenes) and spoken audio dialogue to answer the user's questions accurately.
3. STRICT RULE: Do NOT reveal any characters, plot details, twists, deaths, or events that happen AFTER timestamp (${formatTime(timestamp)}) in the story. Keep it completely spoiler-free!
4. If a user asks about something that hasn't happened yet, say: "That hasn't happened yet in the playback! Keep watching to find out!" or something similar.
5. Be natural, conversational, and direct.

Current Playback Time: ${formatTime(timestamp)}

${contextStr}`;
}

// Append message to UI
function appendMessage(role, text) {
    const msg = document.createElement('div');
    msg.className = `message ${role}`;
    
    const formattedTime = new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
    
    msg.innerHTML = `
        <div class="msg-content">${text}</div>
        <span class="timestamp">${formattedTime}</span>
    `;
    
    chatHistoryContainer.appendChild(msg);
    scrollToBottom();
    return msg;
}

// Scroll chat panel to bottom
function scrollToBottom() {
    chatHistoryContainer.scrollTop = chatHistoryContainer.scrollHeight;
}

// Formats response markdown tags for rich display
function formatResponse(text) {
    // Escape HTML entities to prevent XSS, then apply formatting
    let html = escapeHtml(text);
    
    // Code blocks (triple backtick)
    html = html.replace(/```([\s\S]*?)```/g, '<pre style="background:rgba(255,255,255,0.05);padding:10px;border-radius:8px;overflow-x:auto;font-size:0.85rem;"><code>$1</code></pre>');
    // Inline code
    html = html.replace(/`([^`]+)`/g, '<code style="background:rgba(255,255,255,0.08);padding:2px 5px;border-radius:4px;font-size:0.85em;">$1</code>');
    // Headers (### h3, ## h2, # h1)
    html = html.replace(/^### (.+)$/gm, '<strong style="font-size:1.05em;">$1</strong>');
    html = html.replace(/^## (.+)$/gm, '<strong style="font-size:1.1em;">$1</strong>');
    html = html.replace(/^# (.+)$/gm, '<strong style="font-size:1.15em;">$1</strong>');
    // Bold and italic
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');
    // Bullet lists
    html = html.replace(/^[\-\*] (.+)$/gm, '• $1');
    // Line breaks
    html = html.replace(/\n/g, '<br>');
    
    return html;
}

// Custom Volume Controls to override browser limits or bugs
if (volumeToggleBtn) {
    volumeToggleBtn.addEventListener('click', () => {
        if (videoPlayer.muted) {
            videoPlayer.muted = false;
            videoPlayer.volume = 1.0;
            volumeIcon.className = 'fa-solid fa-volume-high';
            volumeText.textContent = 'Mute';
        } else {
            videoPlayer.muted = true;
            volumeIcon.className = 'fa-solid fa-volume-xmark';
            volumeText.textContent = 'Unmute';
        }
    });
}

// Sync volume toggle icon when native controls are used
videoPlayer.addEventListener('volumechange', () => {
    if (videoPlayer.muted || videoPlayer.volume === 0) {
        volumeIcon.className = 'fa-solid fa-volume-xmark';
        volumeText.textContent = 'Unmute';
    } else {
        volumeIcon.className = 'fa-solid fa-volume-high';
        volumeText.textContent = 'Mute';
    }
});
