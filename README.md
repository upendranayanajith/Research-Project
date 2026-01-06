## Project Overview

"Multi Model Ensemble Architecture For Analog Clock Reading" is a sophisticated deep learning system designed to analyze analog clock images through a multi-stage pipeline. The project combines state-of-the-art computer vision models with physics-based reasoning to accurately determine the time shown on a clock face.

### Key Features

- Multi-Stage Architecture: Four interconnected components (C1-C4) working together
- Real-time Processing: Support for live video stream analysis via Streamlit
- Expert Mode: Hybrid approach combining deep learning and physics validation
- Visual Analytics: Comprehensive metrics tracking and performance monitoring
- Explainable AI: Gradient-CAM visualizations for model interpretability
- REST API: FastAPI backend for integration with external systems
- Interactive UI: Streamlit-based frontend for visualization and testing

---

## Architecture

### System Architecture Diagram


### Component Breakdown

| Component | Model | Purpose |
|-----------|-------|---------|
| C1: Localization | YOLOv8 | Detects and localizes the clock face in the image |
| C2: Hands Skeleton | YOLOv8 | Identifies hour and minute hand keypoints |
| C3: Angle Regression | ResNet-18 | Predicts precise hour and minute hand angles |
| C4: Physics Engine | Rule-based | Validates predictions using physics constraints |


## Project Dependencies

### Core AI & Computer Vision
- torch  - Deep learning framework
- torchvision  - Computer vision utilities
- ultralytics  - YOLOv8 implementation
- opencv-python  - Image processing
- numpy  - Numerical computing
- Pillow  - Image handling
- grad-cam  - Explainable AI visualizations

### Backend Framework
- fastapi  - Modern Python web framework
- uvicorn  - ASGI server
- python-multipart  - Form data handling

### Frontend Framework
- streamlit  - Interactive UI framework
- streamlit-webrtc  - Real-time video streaming
- av  - Audio/video processing
- requests  - HTTP client library

### Data Analytics & Visualization
- pandas - Data manipulation
- plotly  - Interactive visualizations

---

## Getting Started

### Prerequisites
- Python 3.8+
- GPU (NVIDIA CUDA-enabled) recommended for optimal performance
- 4GB+ RAM

### Installation

1. Clone the repository
   ```bash
   cd Research-Project

2. Create virtual environment 
python -m venv venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac

3. Install dependencies
pip install -r requirements.txt

4. Verify GPU setup 
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"


Option 1: Interactive Streamlit App
streamlit run app/frontend.py

FastAPI Server
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

API endpoint: http://localhost:8000
Swagger docs: http://localhost:8000/docs


# Terminal 1: Start API
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# Terminal 2: Start Frontend
streamlit run app/frontend.py