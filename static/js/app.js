// StepMedia HRM JavaScript Application

// Theme Management
const ThemeManager = {
    init() {
        const savedTheme = localStorage.getItem('theme') || 'light';
        this.setTheme(savedTheme);
        
        // Theme toggle button
        const themeToggle = document.getElementById('theme-toggle');
        if (themeToggle) {
            themeToggle.addEventListener('click', () => {
                const currentTheme = document.documentElement.getAttribute('data-theme') || 'light';
                const newTheme = currentTheme === 'light' ? 'dark' : 'light';
                this.setTheme(newTheme);
            });
        }
    },
    
    setTheme(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        if (theme === 'dark') {
            document.documentElement.classList.add('dark');
        } else {
            document.documentElement.classList.remove('dark');
        }
        localStorage.setItem('theme', theme);
    }
};

// Socket.IO Connection Management
const SocketManager = {
    socket: null,
    
    init() {
        if (typeof io !== 'undefined') {
            this.socket = io();
            this.setupEventListeners();
        }
    },
    
    setupEventListeners() {
        if (!this.socket) return;
        
        this.socket.on('connect', () => {
            console.log('Connected to server');
        });
        
        this.socket.on('disconnect', () => {
            console.log('Disconnected from server');
        });
        
        this.socket.on('processing_progress', (data) => {
            this.handleProcessingProgress(data);
        });
        
        this.socket.on('notification', (data) => {
            this.showNotification(data.message, data.type);
        });
    },
    
    handleProcessingProgress(data) {
        const progressBar = document.querySelector(`[data-video-id="${data.video_id}"] .progress-bar`);
        if (progressBar) {
            progressBar.style.width = `${data.progress}%`;
        }
        
        const statusElement = document.querySelector(`[data-video-id="${data.video_id}"] .status-badge`);
        if (statusElement) {
            statusElement.textContent = data.status;
            statusElement.className = `status-badge ${data.status}`;
        }
    },
    
    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `notification notification-enter fixed top-4 right-4 p-4 rounded-lg shadow-lg z-50 max-w-sm
                                ${type === 'success' ? 'bg-green-500 text-white' : ''}
                                ${type === 'error' ? 'bg-red-500 text-white' : ''}
                                ${type === 'info' ? 'bg-blue-500 text-white' : ''}
                                ${type === 'warning' ? 'bg-yellow-500 text-black' : ''}`;
        
        notification.innerHTML = `
            <div class="flex items-center justify-between">
                <span>${message}</span>
                <button onclick="this.parentElement.parentElement.remove()" class="ml-4 text-lg">&times;</button>
            </div>
        `;
        
        document.body.appendChild(notification);
        
        // Auto remove after 5 seconds
        setTimeout(() => {
            if (notification.parentElement) {
                notification.classList.add('notification-exit');
                setTimeout(() => notification.remove(), 300);
            }
        }, 5000);
    }
};

// File Upload Management
const FileUploadManager = {
    init() {
        this.setupDropZones();
        this.setupFileInputs();
    },
    
    setupDropZones() {
        const dropZones = document.querySelectorAll('.upload-zone');
        
        dropZones.forEach(zone => {
            zone.addEventListener('dragover', (e) => {
                e.preventDefault();
                zone.classList.add('dragover');
            });
            
            zone.addEventListener('dragleave', (e) => {
                e.preventDefault();
                zone.classList.remove('dragover');
            });
            
            zone.addEventListener('drop', (e) => {
                e.preventDefault();
                zone.classList.remove('dragover');
                
                const files = Array.from(e.dataTransfer.files);
                this.handleFiles(files, zone);
            });
        });
    },
    
    setupFileInputs() {
        const fileInputs = document.querySelectorAll('input[type="file"]');
        
        fileInputs.forEach(input => {
            input.addEventListener('change', (e) => {
                const files = Array.from(e.target.files);
                this.handleFiles(files, input.closest('.upload-zone'));
            });
        });
    },
    
    handleFiles(files, zone) {
        const allowedTypes = zone.dataset.allowedTypes ? zone.dataset.allowedTypes.split(',') : [];
        const maxSize = zone.dataset.maxSize ? parseInt(zone.dataset.maxSize) : 500 * 1024 * 1024; // 500MB default
        
        files.forEach(file => {
            if (allowedTypes.length > 0 && !allowedTypes.some(type => file.type.includes(type))) {
                SocketManager.showNotification(`File type not allowed: ${file.name}`, 'error');
                return;
            }
            
            if (file.size > maxSize) {
                SocketManager.showNotification(`File too large: ${file.name}`, 'error');
                return;
            }
            
            this.uploadFile(file, zone);
        });
    },
    
    uploadFile(file, zone) {
        const formData = new FormData();
        formData.append('file', file);
        
        const uploadUrl = zone.dataset.uploadUrl || '/videos/upload';
        
        // Create progress indicator
        const progressContainer = this.createProgressIndicator(file.name);
        zone.appendChild(progressContainer);
        
        fetch(uploadUrl, {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                SocketManager.showNotification(`Upload successful: ${file.name}`, 'success');
                // Refresh page or update UI
                setTimeout(() => location.reload(), 2000);
            } else {
                SocketManager.showNotification(`Upload failed: ${data.error}`, 'error');
            }
        })
        .catch(error => {
            SocketManager.showNotification(`Upload error: ${error.message}`, 'error');
        })
        .finally(() => {
            progressContainer.remove();
        });
    },
    
    createProgressIndicator(filename) {
        const container = document.createElement('div');
        container.className = 'mt-4 p-3 bg-gray-100 dark:bg-gray-700 rounded-lg';
        container.innerHTML = `
            <div class="flex items-center justify-between mb-2">
                <span class="text-sm font-medium text-gray-700 dark:text-gray-300">${filename}</span>
                <span class="text-sm text-gray-500">Uploading...</span>
            </div>
            <div class="w-full bg-gray-200 dark:bg-gray-600 rounded-full h-2">
                <div class="bg-primary h-2 rounded-full progress-bar" style="width: 0%"></div>
            </div>
        `;
        return container;
    }
};

// Video Player Management
const VideoPlayerManager = {
    players: new Map(),
    
    init() {
        this.setupVideoPlayers();
    },
    
    setupVideoPlayers() {
        const videoElements = document.querySelectorAll('video[data-video-id]');
        
        videoElements.forEach(video => {
            const videoId = video.dataset.videoId;
            this.players.set(videoId, {
                element: video,
                detections: JSON.parse(video.dataset.detections || '[]'),
                currentDetection: null
            });
            
            this.setupVideoControls(video, videoId);
        });
    },
    
    setupVideoControls(video, videoId) {
        const player = this.players.get(videoId);
        
        // Add time update listener
        video.addEventListener('timeupdate', () => {
            this.updateDetectionOverlay(videoId, video.currentTime);
        });
        
        // Setup person timeline clicks
        const timeline = document.querySelector(`[data-video-id="${videoId}"] .person-timeline`);
        if (timeline) {
            timeline.addEventListener('click', (e) => {
                const rect = timeline.getBoundingClientRect();
                const clickX = e.clientX - rect.left;
                const percentage = clickX / rect.width;
                const time = percentage * video.duration;
                video.currentTime = time;
            });
        }
    },
    
    updateDetectionOverlay(videoId, currentTime) {
        const player = this.players.get(videoId);
        if (!player) return;
        
        // Find current detection
        const currentDetection = player.detections.find(detection => 
            currentTime >= detection.start_time && currentTime <= detection.end_time
        );
        
        if (currentDetection !== player.currentDetection) {
            player.currentDetection = currentDetection;
            this.renderDetectionBoxes(videoId, currentDetection);
        }
    },
    
    renderDetectionBoxes(videoId, detection) {
        const container = document.querySelector(`[data-video-id="${videoId}"] .video-container`);
        if (!container) return;
        
        // Remove existing boxes
        container.querySelectorAll('.face-box').forEach(box => box.remove());
        
        if (detection && detection.bbox_data) {
            detection.bbox_data.forEach(bbox => {
                const box = document.createElement('div');
                box.className = 'face-box';
                box.style.left = `${bbox.x}%`;
                box.style.top = `${bbox.y}%`;
                box.style.width = `${bbox.width}%`;
                box.style.height = `${bbox.height}%`;
                
                if (detection.employee_name) {
                    box.innerHTML = `<span class="bg-accent text-white px-2 py-1 text-xs rounded">${detection.employee_name}</span>`;
                }
                
                container.appendChild(box);
            });
        }
    },
    
    jumpToPerson(videoId, personCode) {
        const player = this.players.get(videoId);
        if (!player) return;
        
        const detection = player.detections.find(d => d.person_code === personCode);
        if (detection) {
            player.element.currentTime = detection.start_time;
            player.element.play();
        }
    }
};

// Data Table Management
const DataTableManager = {
    init() {
        this.setupSorting();
        this.setupFiltering();
        this.setupPagination();
    },
    
    setupSorting() {
        const sortableHeaders = document.querySelectorAll('[data-sort]');
        
        sortableHeaders.forEach(header => {
            header.addEventListener('click', () => {
                const column = header.dataset.sort;
                const currentOrder = header.dataset.order || 'asc';
                const newOrder = currentOrder === 'asc' ? 'desc' : 'asc';
                
                this.sortTable(column, newOrder);
                header.dataset.order = newOrder;
                
                // Update UI indicators
                sortableHeaders.forEach(h => h.classList.remove('sort-asc', 'sort-desc'));
                header.classList.add(`sort-${newOrder}`);
            });
        });
    },
    
    setupFiltering() {
        const filterInputs = document.querySelectorAll('[data-filter]');
        
        filterInputs.forEach(input => {
            input.addEventListener('input', debounce(() => {
                this.applyFilters();
            }, 300));
        });
    },
    
    setupPagination() {
        const paginationButtons = document.querySelectorAll('[data-page]');
        
        paginationButtons.forEach(button => {
            button.addEventListener('click', (e) => {
                e.preventDefault();
                const page = parseInt(button.dataset.page);
                this.loadPage(page);
            });
        });
    },
    
    sortTable(column, order) {
        const url = new URL(window.location);
        url.searchParams.set('sort', column);
        url.searchParams.set('order', order);
        window.location.href = url.toString();
    },
    
    applyFilters() {
        const filters = {};
        document.querySelectorAll('[data-filter]').forEach(input => {
            if (input.value) {
                filters[input.dataset.filter] = input.value;
            }
        });
        
        const url = new URL(window.location);
        Object.keys(filters).forEach(key => {
            url.searchParams.set(key, filters[key]);
        });
        
        window.location.href = url.toString();
    },
    
    loadPage(page) {
        const url = new URL(window.location);
        url.searchParams.set('page', page);
        window.location.href = url.toString();
    }
};

// Mobile Menu Management
const MobileMenuManager = {
    init() {
        this.setupMobileToggle();
        this.setupOverlay();
    },
    
    setupMobileToggle() {
        const mobileToggle = document.getElementById('mobile-menu-toggle');
        const sidebar = document.querySelector('nav.fixed.left-0');
        
        if (mobileToggle && sidebar) {
            mobileToggle.addEventListener('click', () => {
                sidebar.classList.toggle('mobile-open');
                this.toggleOverlay();
            });
        }
    },
    
    setupOverlay() {
        const overlay = document.getElementById('mobile-overlay');
        if (overlay) {
            overlay.addEventListener('click', () => {
                this.closeMobileMenu();
            });
        }
    },
    
    toggleOverlay() {
        let overlay = document.getElementById('mobile-overlay');
        if (!overlay) {
            overlay = document.createElement('div');
            overlay.id = 'mobile-overlay';
            overlay.className = 'mobile-overlay';
            document.body.appendChild(overlay);
        }
        overlay.style.display = overlay.style.display === 'block' ? 'none' : 'block';
    },
    
    closeMobileMenu() {
        const sidebar = document.querySelector('nav.fixed.left-0');
        const overlay = document.getElementById('mobile-overlay');
        
        if (sidebar) sidebar.classList.remove('mobile-open');
        if (overlay) overlay.style.display = 'none';
    }
};

// Utility Functions
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function formatDuration(seconds) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    
    if (hours > 0) {
        return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }
    return `${minutes}:${secs.toString().padStart(2, '0')}`;
}

// Initialize everything when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    ThemeManager.init();
    SocketManager.init();
    FileUploadManager.init();
    VideoPlayerManager.init();
    DataTableManager.init();
    MobileMenuManager.init();
    
    console.log('StepMedia HRM initialized');
});

// Global object for external access
window.StepMediaHRM = {
    ThemeManager,
    SocketManager,
    FileUploadManager,
    VideoPlayerManager,
    DataTableManager,
    MobileMenuManager
};