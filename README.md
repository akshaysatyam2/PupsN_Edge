# PupsN_Edge

PupsN Vision System - Day 1 MVP Pipeline

This project implements a lightweight, dual-threaded Core Flask Backend Architecture designed for processing video streams, running AI models (YOLO Nano / OSNet), and streaming processed frames to a web client via WebSockets and HTML5 Canvas.

## Features
- Real-time video processing with robust frame buffer management.
- Dual-threaded backend (Web Server + AI Inference Engine).
- Frame skipping for AI processing (1 FPS) while streaming at native framerate.
- SQLite-based database with startup vector caching.
- Base64 Frame Streaming Protocol over WebSockets.
