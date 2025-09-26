class VideoEditor {
    constructor() {
        this.images = [];
        this.imageOrder = [];
        this.currentSlideIndex = 0;
        this.slideshowInterval = null;
        this.audio = null;
        this.isPlaying = false;
        this.orderFileName = 'image_order.txt';
        
        this.initializeElements();
        this.loadImages();
        this.setupEventListeners();
        this.loadOrderFromFile();
    }

    initializeElements() {
        this.imageGrid = document.getElementById('imageGrid');
        this.playBtn = document.getElementById('playBtn');
        this.slideshowContainer = document.getElementById('slideshowContainer');
        this.slideshowImage = document.getElementById('slideshowImage');
        this.stopBtn = document.getElementById('stopBtn');
        this.currentImageInfo = document.getElementById('currentImageInfo');
        this.debugAudio = document.getElementById('debugAudio');
    }

    async loadImages() {
        try {
            // In a real implementation, you'd need a backend to list files
            // For now, we'll use the known images from the directory
            const imageFiles = [
                'apollo 11.png',
                'einstein.png',
                'moon landing.png',
                'newton.png',
                'tycho brahe.png'
            ];

            this.images = imageFiles.map((filename, index) => ({
                filename,
                src: `images/${filename}`,
                order: index
            }));

            this.imageOrder = [...this.images];
            this.renderImages();
        } catch (error) {
            this.showError('Failed to load images: ' + error.message);
        }
    }

    renderImages() {
        this.imageGrid.innerHTML = '';
        
        this.imageOrder.forEach((image, index) => {
            const tile = this.createImageTile(image, index);
            this.imageGrid.appendChild(tile);
        });
    }

    createImageTile(image, index) {
        const tile = document.createElement('div');
        tile.className = 'image-tile';
        tile.draggable = true;
        tile.dataset.index = index;

        tile.innerHTML = `
            <div class="order-number">${index + 1}</div>
            <img src="${image.src}" alt="${image.filename}" onerror="this.src='data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjE1MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZGRkIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIxNCIgZmlsbD0iIzk5OSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPkltYWdlIG5vdCBmb3VuZDwvdGV4dD48L3N2Zz4='">
            <div class="filename">${image.filename}</div>
        `;

        this.setupDragAndDrop(tile, index);
        return tile;
    }

    setupDragAndDrop(tile, index) {
        tile.addEventListener('dragstart', (e) => {
            tile.classList.add('dragging');
            e.dataTransfer.effectAllowed = 'move';
            e.dataTransfer.setData('text/html', tile.outerHTML);
            e.dataTransfer.setData('text/plain', index);
        });

        tile.addEventListener('dragend', () => {
            tile.classList.remove('dragging');
        });

        tile.addEventListener('dragover', (e) => {
            e.preventDefault();
            e.dataTransfer.dropEffect = 'move';
        });

        tile.addEventListener('drop', (e) => {
            e.preventDefault();
            const draggedIndex = parseInt(e.dataTransfer.getData('text/plain'));
            const targetIndex = parseInt(tile.dataset.index);

            if (draggedIndex !== targetIndex) {
                this.reorderImages(draggedIndex, targetIndex);
            }
        });
    }

    reorderImages(fromIndex, toIndex) {
        const draggedImage = this.imageOrder.splice(fromIndex, 1)[0];
        this.imageOrder.splice(toIndex, 0, draggedImage);
        this.renderImages();
        this.autoSaveOrder();
    }

    setupEventListeners() {
        this.playBtn.addEventListener('click', () => this.startSlideshow());
        this.stopBtn.addEventListener('click', () => this.stopSlideshow());

        // Keyboard controls
        document.addEventListener('keydown', (e) => {
            if (this.isPlaying) {
                if (e.key === 'Escape') {
                    this.stopSlideshow();
                }
            }
        });
    }

    async startSlideshow() {
        if (this.imageOrder.length === 0) {
            alert('No images to display!');
            return;
        }

        this.isPlaying = true;
        this.currentSlideIndex = 0;
        this.slideshowContainer.classList.add('active');
        this.playBtn.disabled = true;

        // Load and play audio
        try {
            this.audio = new Audio('./hung-up.mp3');
            this.audio.preload = 'auto';
            this.audio.volume = 0.7;
            
            // Try to play audio, but don't fail if it doesn't work
            this.audio.play().catch(e => {
                console.log('Audio autoplay blocked or failed:', e);
                // Show a message to user that they need to interact first
                this.showAudioMessage();
            });
        } catch (error) {
            console.log('Audio not available:', error);
        }

        this.showCurrentSlide();
        this.slideshowInterval = setInterval(() => {
            this.nextSlide();
        }, 1000); // 0.5 seconds per image
    }

    showCurrentSlide() {
        const currentImage = this.imageOrder[this.currentSlideIndex];
        this.slideshowImage.src = currentImage.src;
        this.currentImageInfo.textContent = `${this.currentSlideIndex + 1} / ${this.imageOrder.length} - ${currentImage.filename}`;
    }

    nextSlide() {
        this.currentSlideIndex++;
        if (this.currentSlideIndex >= this.imageOrder.length) {
            this.stopSlideshow();
            return;
        }
        this.showCurrentSlide();
    }

    stopSlideshow() {
        this.isPlaying = false;
        this.slideshowContainer.classList.remove('active');
        this.playBtn.disabled = false;

        if (this.slideshowInterval) {
            clearInterval(this.slideshowInterval);
            this.slideshowInterval = null;
        }

        if (this.audio) {
            this.audio.pause();
            this.audio.currentTime = 0;
        }
    }

    loadOrderFromLines(lines) {
        const newOrder = [];
        const remainingImages = [...this.images];

        // First, add images in the order specified in the file
        lines.forEach(filename => {
            const image = remainingImages.find(img => img.filename === filename.trim());
            if (image) {
                newOrder.push(image);
                const index = remainingImages.indexOf(image);
                remainingImages.splice(index, 1);
            }
        });

        // Add any remaining images that weren't in the file
        newOrder.push(...remainingImages);

        this.imageOrder = newOrder;
        this.renderImages();
    }

    async autoSaveOrder() {
        try {
            const orderData = this.imageOrder.map(img => img.filename).join('\n');
            // Save to localStorage for auto-loading
            localStorage.setItem('imageOrder', orderData);
            console.log('Order auto-saved to localStorage');
        } catch (error) {
            console.log('Auto-save failed:', error);
        }
    }

    async loadOrderFromFile() {
        try {
            // First try to load from localStorage
            const savedOrder = localStorage.getItem('imageOrder');
            if (savedOrder) {
                const lines = savedOrder.split('\n').filter(line => line.trim());
                this.loadOrderFromLines(lines);
                console.log('Order loaded from localStorage');
                return;
            }

            // If no localStorage, try to fetch the order file
            const response = await fetch(this.orderFileName);
            if (response.ok) {
                const text = await response.text();
                const lines = text.split('\n').filter(line => line.trim());
                this.loadOrderFromLines(lines);
                console.log('Order loaded from file');
            }
        } catch (error) {
            console.log('No saved order found, using default order');
        }
    }

    showAudioMessage() {
        // Create a temporary message about audio
        const message = document.createElement('div');
        message.style.cssText = `
            position: fixed;
            top: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 15px 25px;
            border-radius: 10px;
            z-index: 1001;
            font-size: 14px;
            text-align: center;
        `;
        message.innerHTML = '🎵 Click anywhere to enable audio playback';
        document.body.appendChild(message);
        
        // Remove message after 3 seconds
        setTimeout(() => {
            if (message.parentNode) {
                message.parentNode.removeChild(message);
            }
        }, 3000);

        // Enable audio on first user interaction
        const enableAudio = () => {
            if (this.audio && this.isPlaying) {
                this.audio.play().catch(e => console.log('Audio still failed:', e));
            }
            document.removeEventListener('click', enableAudio);
            document.removeEventListener('keydown', enableAudio);
        };
        
        document.addEventListener('click', enableAudio);
        document.addEventListener('keydown', enableAudio);
    }

    showError(message) {
        this.imageGrid.innerHTML = `<div class="error">${message}</div>`;
    }
}

// Initialize the app when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new VideoEditor();
});
