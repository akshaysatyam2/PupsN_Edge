const canvas = document.getElementById('videoCanvas');
const ctx = canvas.getContext('2d');
const statusBanner = document.getElementById('status-banner');

// Connect to the WebSocket Server
const socket = io();

socket.on('connect', () => {
    console.log('Connected to WebSocket server');
});

socket.on('stream_update', (data) => {
    // Check Status
    if (data.status === 'inactive') {
        statusBanner.textContent = "Camera Feed Inactive or Reconnecting...";
        statusBanner.className = "status-inactive";
        return;
    } else {
        statusBanner.textContent = "Camera Feed Active";
        statusBanner.className = "status-active";
    }

    // Initialize an in-memory image
    const img = new Image();
    
    img.onload = () => {
        // Clear previous frame
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Dynamically adjust canvas to incoming frame dimensions if needed
        if (canvas.width !== img.width || canvas.height !== img.height) {
            canvas.width = img.width;
            canvas.height = img.height;
        }

        // Draw fresh background frame
        ctx.drawImage(img, 0, 0);

        // Render AI Detections
        if (data.detections && data.detections.length > 0) {
            data.detections.forEach(det => {
                const [x_min, y_min, x_max, y_max] = det.bbox;
                const width = x_max - x_min;
                const height = y_max - y_min;

                // Set styles for bounding box
                ctx.strokeStyle = '#39ff14'; // Neon Green
                ctx.lineWidth = 3;
                
                // Draw Box
                ctx.beginPath();
                ctx.rect(x_min, y_min, width, height);
                ctx.stroke();

                // Draw Background for Text
                const label = `${det.name} (${Math.round(det.confidence * 100)}%)`;
                ctx.font = '16px Arial';
                const textMetrics = ctx.measureText(label);
                
                ctx.fillStyle = '#39ff14';
                ctx.fillRect(x_min, y_min - 25, textMetrics.width + 10, 25);
                
                // Draw Text Element
                ctx.fillStyle = '#000000';
                ctx.fillText(label, x_min + 5, y_min - 7);
            });
        }
    };
    
    // Trigger image load with the incoming Base64 ASCII string
    img.src = data.image;
});
