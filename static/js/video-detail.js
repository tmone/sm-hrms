// Video Detail Page JavaScript

// Global variables
let socket = null;
let videoId = null;
let videoStatus = null;

// Initialize the page
function initializeVideoDetail(config) {
    videoId = config.videoId;
    videoStatus = config.videoStatus;
    
    // Set up event listeners
    setupEventListeners();
    
    // Initialize progress tracking based on video status
    if (videoStatus === 'converting') {
        console.log('🔄 Video is converting, setting up real-time progress tracking...');
        if (typeof io !== 'undefined') {
            initializeWebSocket();
        } else {
            console.log('📡 Socket.IO not available, using AJAX polling...');
            setupAjaxPolling();
        }
    } else if (videoStatus === 'processing') {
        console.log('🔄 Video is processing (person extraction), setting up progress tracking...');
        setupProcessingPolling();
    } else {
        // Check if video might have just started processing
        console.log('🔍 Checking if video started processing...');
        setTimeout(function() {
            checkInitialStatus();
        }, 1000);
    }
}

// Video handler functions
function handleVideoLoad() {
    console.log('Video loading started...');
    const loading = document.getElementById('videoLoading');
    const error = document.getElementById('videoError');
    if (loading) loading.classList.add('hidden');
    if (error) error.classList.add('hidden');
}

function handleVideoError() {
    console.log('❌ Video error occurred');
    const video = document.getElementById('videoPlayer');
    const error = document.getElementById('videoError');
    
    if (video && video.error) {
        console.error('Video error details:', {
            code: video.error.code,
            message: video.error.message,
            MEDIA_ERR_ABORTED: video.error.code === 1,
            MEDIA_ERR_NETWORK: video.error.code === 2,
            MEDIA_ERR_DECODE: video.error.code === 3,
            MEDIA_ERR_SRC_NOT_SUPPORTED: video.error.code === 4
        });
        
        // Try to get more info about the video
        const sources = video.querySelectorAll('source');
        sources.forEach((source, index) => {
            console.log(`Source ${index + 1}: ${source.src}`);
            console.log(`Type: ${source.type}`);
        });
    }
    
    if (error) {
        error.classList.remove('hidden');
    }
}

// Toggle person detections in grouped view
function togglePersonDetections(personId) {
    const detectionsDiv = document.getElementById(`detections-${personId}`);
    const toggleSpan = document.getElementById(`toggle-${personId}`);
    
    if (detectionsDiv) {
        if (detectionsDiv.classList.contains('hidden')) {
            detectionsDiv.classList.remove('hidden');
            if (toggleSpan) toggleSpan.textContent = '▲';
        } else {
            detectionsDiv.classList.add('hidden');
            if (toggleSpan) toggleSpan.textContent = '▼';
        }
    }
}

// Navigation function for person detection
function navigateToDetection(timestamp, detectionId, bboxX, bboxY, bboxWidth, bboxHeight) {
    console.log(`🎯 Navigating to detection ${detectionId} at ${timestamp}s`);
    
    const video = document.getElementById('videoPlayer');
    if (!video) {
        console.error('❌ Video player not found');
        return;
    }
    
    // Validate timestamp
    if (isNaN(timestamp) || timestamp < 0) {
        console.error('❌ Invalid timestamp:', timestamp);
        return;
    }
    
    // Wait for video metadata to be loaded
    if (video.readyState < 1) {
        console.log('⏳ Waiting for video metadata to load...');
        video.addEventListener('loadedmetadata', function() {
            performNavigation();
        }, { once: true });
    } else {
        performNavigation();
    }
    
    function performNavigation() {
        console.log(`🎬 Video duration: ${video.duration}s, seeking to: ${timestamp}s`);
        
        // Validate timestamp against video duration
        if (timestamp > video.duration) {
            console.warn('⚠️ Timestamp exceeds video duration, adjusting...');
            timestamp = Math.min(timestamp, video.duration - 0.1);
        }
        
        // Seek to timestamp and PAUSE (don't auto-play)
        video.currentTime = timestamp;
        if (!video.paused) {
            video.pause();
        }
        
        // Highlight the detection row in the navigator panel
        highlightDetectionInNavigator(detectionId);
        
        // Remove any existing bounding box (cleanup)
        const existingBox = document.getElementById('boundingBox');
        if (existingBox) {
            existingBox.remove();
        }
        
        // Update current time display
        updateTimeDisplay();
        
        // Scroll detection item into view if needed
        const detectionItem = document.querySelector(`[data-detection-id="${detectionId}"]`);
        if (detectionItem) {
            detectionItem.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
        
        console.log('✅ Navigation to detection completed (no box drawn)');
    }
}

// Progress update functions
function updateProgressUI(data) {
    console.log('🎨 Updating progress UI with data:', data);
    
    const progressBar = document.getElementById('conversionProgressBar');
    const progressPercent = document.getElementById('conversionPercent');
    const progressMessage = document.getElementById('conversionMessage');
    const liveStatus = document.getElementById('liveStatus');
    
    if (data.status === 'completed') {
        console.log('✅ Conversion completed! Progress:', data.progress);
        if (progressBar) progressBar.style.width = '100%';
        if (progressPercent) progressPercent.textContent = '100%';
        if (progressMessage) progressMessage.textContent = 'Conversion completed successfully!';
        if (liveStatus) liveStatus.textContent = '✅ Conversion completed';
        
        // Reload page after a short delay to show completed state
        setTimeout(() => {
            console.log('🔄 Reloading page to show converted video...');
            location.reload();
        }, 2000);
        
    } else if (data.status === 'failed') {
        console.error('❌ Conversion failed:', data.error_message);
        if (progressBar) progressBar.style.width = '0%';
        if (progressPercent) progressPercent.textContent = 'Failed';
        if (progressMessage) progressMessage.textContent = data.error_message || 'Conversion failed';
        if (liveStatus) liveStatus.textContent = '❌ Conversion failed';
        
        // Clear any progress intervals
        if (window.conversionProgressInterval) {
            clearInterval(window.conversionProgressInterval);
        }
        
    } else if (data.status === 'converting') {
        const progress = data.progress || 0;
        const message = data.progress_message || 'Converting...';
        
        console.log(`🔄 Conversion progress: ${progress.toFixed(1)}% - ${message}`);
        
        if (progressBar) {
            progressBar.style.width = `${progress}%`;
            console.log(`📊 Updated progress bar to ${progress}%`);
        }
        if (progressPercent) progressPercent.textContent = `${progress.toFixed(1)}%`;
        if (progressMessage) progressMessage.textContent = message;
        
        // Update live status with elapsed time
        const startTime = data.started_at ? new Date(data.started_at).getTime() : Date.now();
        const elapsed = Math.floor((Date.now() - startTime) / 1000);
        const elapsedMinutes = Math.floor(elapsed / 60);
        const elapsedSeconds = elapsed % 60;
        const elapsedText = elapsedMinutes > 0 ? 
            `${elapsedMinutes}m ${elapsedSeconds}s` : `${elapsedSeconds}s`;
        
        if (liveStatus) {
            liveStatus.innerHTML = `🔄 ${message} (${elapsedText} elapsed)`;
        }
        
        // Update page title to show progress
        document.title = `Converting ${progress.toFixed(1)}% - ${message}`;
        
        // Visual feedback: pulse effect on high progress
        if (progress > 80 && progressBar) {
            progressBar.style.animation = 'pulse 1s infinite';
        } else if (progressBar) {
            progressBar.style.animation = 'none';
        }
    }
}

function updateProcessingUI(data) {
    console.log('🎨 Updating person extraction UI with data:', data);
    
    const progressBar = document.getElementById('processingProgressBar');
    const progressPercent = document.getElementById('processingPercent');
    const progressMessage = document.getElementById('processingMessage');
    
    // Reset consecutive failures on successful response
    window.consecutiveFailures = 0;
    
    if (data.status === 'completed') {
        console.log('✅ Person extraction completed!');
        if (progressBar) progressBar.style.width = '100%';
        if (progressPercent) progressPercent.textContent = '100%';
        if (progressMessage) progressMessage.textContent = 'Person extraction completed successfully!';
        
        // Clear the progress interval
        if (window.processingProgressInterval) {
            clearInterval(window.processingProgressInterval);
            window.processingProgressInterval = null;
        }
        
        setTimeout(() => location.reload(), 2000);
        
    } else if (data.status === 'failed') {
        console.log('❌ Person extraction failed:', data.error_message);
        showProcessingError(data.error_message || 'Unknown error', true);
        
    } else if (data.status === 'processing') {
        const progress = data.progress || 0;
        const message = data.progress_message || 'Processing...';
        
        console.log(`🔄 Person extraction: ${progress.toFixed(1)}% - ${message}`);
        
        // Track progress updates for stuck detection
        const currentProgress = Math.floor(progress);
        if (currentProgress > (window.lastProgress || 0)) {
            window.lastProgress = currentProgress;
            window.lastProgressUpdate = Date.now();
        }
        
        // Update progress bar with animation
        if (progressBar) {
            progressBar.style.width = `${progress}%`;
            progressBar.style.transition = 'width 0.3s ease-in-out';
            
            // Add visual feedback when progress updates
            if (progress > (window.lastVisualProgress || 0)) {
                progressBar.style.backgroundColor = '#22c55e'; // Green flash
                setTimeout(() => {
                    progressBar.style.backgroundColor = '#2563eb'; // Back to blue
                }, 300);
                window.lastVisualProgress = progress;
            }
        }
        
        if (progressPercent) progressPercent.textContent = `${progress.toFixed(1)}%`;
        if (progressMessage) progressMessage.textContent = message;
    }
}

// WebSocket functions
function initializeWebSocket() {
    console.log('📡 Initializing WebSocket connection...');
    
    socket = io();
    
    socket.on('connect', function() {
        console.log('📡 WebSocket connected');
        socket.emit('join_video_room', { video_id: videoId });
        socket.emit('request_video_status', { video_id: videoId });
    });
    
    socket.on('disconnect', function() {
        console.log('📡 WebSocket disconnected');
    });
    
    socket.on('conversion_progress', function(data) {
        console.log('🔄 WebSocket progress update:', data);
        if (data.video_id === videoId) {
            updateProgressUI(data);
        }
    });
    
    socket.on('video_status', function(data) {
        console.log('📊 WebSocket video status:', data);
        if (data.video_id === videoId) {
            updateProgressUI(data);
        }
    });
    
    socket.on('connect_error', function(error) {
        console.error('📡 WebSocket connection error:', error);
        console.log('📡 Falling back to AJAX polling...');
        
        if (videoStatus === 'converting') {
            setupAjaxPolling();
        }
    });
}

// AJAX polling functions
function setupAjaxPolling() {
    console.log('🔄 Setting up AJAX polling fallback...');
    
    checkConversionStatus();
    
    const progressInterval = setInterval(function() {
        checkConversionStatus();
    }, 3000);
    
    window.conversionProgressInterval = progressInterval;
    
    document.addEventListener('visibilitychange', function() {
        if (document.hidden && window.conversionProgressInterval) {
            clearInterval(window.conversionProgressInterval);
            console.log('⏸️ Paused progress tracking (page hidden)');
        } else if (!document.hidden && videoStatus === 'converting') {
            window.conversionProgressInterval = setInterval(checkConversionStatus, 3000);
            console.log('▶️ Resumed progress tracking (page visible)');
        }
    });
}

function setupProcessingPolling() {
    console.log('🔄 Setting up person extraction progress polling...');
    
    window.processingPollCount = 0;
    window.processingStartTime = Date.now();
    window.lastProgressUpdate = Date.now();
    window.maxPollAttempts = 300;
    window.stuckTimeout = 60000;
    
    checkProcessingStatus();
    
    const processingInterval = setInterval(function() {
        checkProcessingStatus();
    }, 2000);
    
    window.processingProgressInterval = processingInterval;
    
    document.addEventListener('visibilitychange', function() {
        if (document.hidden && window.processingProgressInterval) {
            clearInterval(window.processingProgressInterval);
            console.log('⏸️ Paused person extraction tracking (page hidden)');
        } else if (!document.hidden && videoStatus === 'processing') {
            window.processingProgressInterval = setInterval(checkProcessingStatus, 2000);
            console.log('▶️ Resumed person extraction tracking (page visible)');
        }
    });
}

// Status check functions
function checkConversionStatus() {
    console.log(`🔍 Checking conversion status for video ${videoId}`);
    
    fetch(`/videos/api/${videoId}/conversion-status`)
        .then(response => response.json())
        .then(data => {
            console.log('📊 Conversion status data:', data);
            updateProgressUI(data);
            
            if (data.status === 'completed' || data.status === 'failed') {
                if (window.conversionProgressInterval) {
                    clearInterval(window.conversionProgressInterval);
                }
            }
        })
        .catch(error => {
            console.error('❌ Error checking conversion status:', error);
        });
}

function checkProcessingStatus() {
    window.processingPollCount++;
    const elapsedTime = Date.now() - window.processingStartTime;
    const timeSinceLastUpdate = Date.now() - window.lastProgressUpdate;
    
    console.log(`🔍 Checking person extraction status for video ${videoId} (attempt ${window.processingPollCount}, elapsed: ${Math.round(elapsedTime/1000)}s)`);
    
    if (window.processingPollCount > window.maxPollAttempts) {
        console.error('⏰ Processing timeout: exceeded maximum poll attempts');
        showProcessingError('Processing timed out after 10 minutes', true);
        return;
    }
    
    if (timeSinceLastUpdate > window.stuckTimeout) {
        console.error('⏰ Processing appears stuck: no progress for over 60 seconds');
        showProcessingError('Processing appears stuck. No progress detected for over 1 minute.', true);
        return;
    }
    
    fetch(`/videos/api/${videoId}/processing-status`)
        .then(response => {
            console.log(`📡 Processing API Response status: ${response.status}`);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            return response.json();
        })
        .then(data => {
            console.log('📊 Person extraction status data received:', data);
            updateProcessingUI(data);
        })
        .catch(error => {
            console.error('❌ Error checking person extraction status:', error);
            
            window.consecutiveFailures = (window.consecutiveFailures || 0) + 1;
            
            if (window.consecutiveFailures >= 3) {
                showProcessingError('Network error: Unable to check processing status', true);
            } else {
                const message = document.getElementById('processingMessage');
                if (message) {
                    message.textContent = `⚠️ Connection error (${window.consecutiveFailures}/3). Retrying...`;
                }
            }
        });
}

function checkInitialStatus() {
    fetch(`/videos/api/${videoId}/processing-status`)
        .then(response => response.json())
        .then(data => {
            if (data.status === 'processing') {
                console.log('🔄 Video is now processing, setting up progress tracking...');
                setupProcessingPolling();
            }
        })
        .catch(error => console.error('Error checking initial status:', error));
}

// Error handling
function showProcessingError(message, clearInterval) {
    const progressBar = document.getElementById('processingProgressBar');
    const progressPercent = document.getElementById('processingPercent');
    const progressMessage = document.getElementById('processingMessage');
    
    if (progressBar) {
        progressBar.style.width = '0%';
        progressBar.style.backgroundColor = '#ef4444';
    }
    if (progressPercent) progressPercent.textContent = 'Error';
    if (progressMessage) {
        progressMessage.textContent = message;
        progressMessage.style.color = '#ef4444';
    }
    
    if (clearInterval && window.processingProgressInterval) {
        clearInterval(window.processingProgressInterval);
        window.processingProgressInterval = null;
    }
}

// Utility functions
function toggleProcessOptions() {
    const options = document.getElementById('processOptions');
    options.classList.toggle('hidden');
}

function showConversionInstructions() {
    const instructions = document.getElementById('conversionInstructions');
    if (instructions) {
        instructions.innerHTML = `
            <h4 class="font-semibold text-blue-900 mb-2">FFmpeg Conversion Instructions:</h4>
            <div class="space-y-2 text-sm">
                <p>1. Install FFmpeg if not already installed:</p>
                <code class="block bg-gray-800 text-white p-2 rounded">sudo apt install ffmpeg</code>
                
                <p class="mt-3">2. Convert the video to MP4 format:</p>
                <code class="block bg-gray-800 text-white p-2 rounded">ffmpeg -i "input.mkh" -c:v libx264 -c:a aac output.mp4</code>
                
                <p class="mt-3">3. For web optimization, use these settings:</p>
                <code class="block bg-gray-800 text-white p-2 rounded">ffmpeg -i "input.mkh" -c:v libx264 -preset fast -crf 22 -c:a aac -b:a 128k -movflags +faststart output.mp4</code>
            </div>
        `;
        instructions.classList.remove('hidden');
    }
}

function openDiagnosticTool() {
    const modal = document.getElementById('diagnosticModal');
    if (modal) {
        modal.classList.remove('hidden');
    }
}

function closeDiagnosticModal() {
    const modal = document.getElementById('diagnosticModal');
    if (modal) {
        modal.classList.add('hidden');
    }
}

function confirmDeleteVideo(videoId) {
    if (confirm('Are you sure you want to delete this video? This action cannot be undone.')) {
        const form = document.createElement('form');
        form.method = 'POST';
        form.action = `/videos/${videoId}/delete`;
        
        const csrfToken = document.querySelector('meta[name="csrf-token"]');
        if (csrfToken) {
            const input = document.createElement('input');
            input.type = 'hidden';
            input.name = 'csrf_token';
            input.value = csrfToken.content;
            form.appendChild(input);
        }
        
        document.body.appendChild(form);
        form.submit();
    }
}

// Detection navigator functions
function highlightDetectionInNavigator(detectionId) {
    // Remove previous highlights
    document.querySelectorAll('.detection-item').forEach(item => {
        item.classList.remove('active');
    });
    
    // Add highlight to current detection
    const detectionItem = document.querySelector(`[data-detection-id="${detectionId}"]`);
    if (detectionItem) {
        detectionItem.classList.add('active');
    }
}

function updateTimeDisplay() {
    const video = document.getElementById('videoPlayer');
    if (video) {
        const currentTime = video.currentTime;
        const duration = video.duration;
        
        // Update any time display elements
        const timeDisplay = document.getElementById('currentTimeDisplay');
        if (timeDisplay) {
            timeDisplay.textContent = formatTime(currentTime) + ' / ' + formatTime(duration);
        }
    }
}

function formatTime(seconds) {
    const minutes = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${minutes}:${secs.toString().padStart(2, '0')}`;
}

// Event listeners setup
function setupEventListeners() {
    // Video player events
    const video = document.getElementById('videoPlayer');
    if (video) {
        video.addEventListener('timeupdate', updateTimeDisplay);
        video.addEventListener('loadedmetadata', () => {
            console.log('Video metadata loaded');
        });
    }
    
    // Window events
    window.addEventListener('beforeunload', () => {
        if (socket) {
            socket.disconnect();
        }
        if (window.conversionProgressInterval) {
            clearInterval(window.conversionProgressInterval);
        }
        if (window.processingProgressInterval) {
            clearInterval(window.processingProgressInterval);
        }
    });
}

// Export functions for global access
window.handleVideoLoad = handleVideoLoad;
window.handleVideoError = handleVideoError;
window.togglePersonDetections = togglePersonDetections;
window.navigateToDetection = navigateToDetection;
window.toggleProcessOptions = toggleProcessOptions;
window.showConversionInstructions = showConversionInstructions;
window.openDiagnosticTool = openDiagnosticTool;
window.closeDiagnosticModal = closeDiagnosticModal;
window.confirmDeleteVideo = confirmDeleteVideo;
window.initializeVideoDetail = initializeVideoDetail;