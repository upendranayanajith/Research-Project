## Project Overview

"MULTI MODEL ENSEMBLE ARCHITECTURE FOR ANALOG CLOCK READING" is a sophisticated deep learning system designed to analyze analog clock images through a multi-stage pipeline. The project combines state-of-the-art computer vision models with physics-based reasoning to accurately determine the time shown on a clock face.

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

### Processing Pipeline

Image Input
│
├─→ C1: Extract Clock ROI
│
├─→ C2: Detect Hand Keypoints
│
├─→ C3: Predict Hand Angles
│ └─→ Fast Mode (Direct ML) / Expert Mode (Physics Validation)
│
└─→ C4: Physics-Based Validation
└─→ Output Time & Metadata


---

## Project Structure
Research-Project/
├── app/ # Main application code
│ ├── init.py
│ ├── main.py # FastAPI backend server
│ ├── frontend.py # Streamlit UI application
│ └── core/ # Core processing logic
│ ├── init.py
│ ├── engine.py # ClockEngine (orchestrates all 4 components)
│ ├── metrics.py # Analytics & performance tracking
│ └── xai.py # Explainable AI visualizations (Grad-CAM)
│
├── models/ # Pre-trained model weights
│ ├── c1_localization/
│ │ └── best.pt # YOLOv8 clock localization model
│ ├── c2_hands_skeleton/
│ │ └── best.pt # YOLOv8 hand keypoint detection model
│ └── c3_angle_regression/
│ └── best.pth # ResNet-18 angle regression model
│
├── scripts/ # Utility and training scripts
│ ├── c1_localization.py # C1 model training & validation
│ ├── c2_.py # C2 dataset generation & testing
│ ├── c3_.py # C3 angle regression utilities
│ ├── c4_.py # C4 physics engine & reasoning
│ ├── final_inference.py # End-to-end inference pipeline
│ └── verify_models.py # Model verification utility
│
├── requirements.txt # Python dependencies
├── setup_env.bat # Windows environment setup
└── README.md # This file



---

## Project Dependencies

### Core AI & Computer Vision
- torch (>=2.5.0) - Deep learning framework
- torchvision (>=0.20.0) - Computer vision utilities
- ultralytics (>=8.3.0) - YOLOv8 implementation
- opencv-python (>=4.8.0) - Image processing
- numpy (>=1.24.0) - Numerical computing
- Pillow (>=9.5.0) - Image handling
- grad-cam (>=1.4.8) - Explainable AI visualizations

### Backend Framework
- fastapi (>=0.100.0) - Modern Python web framework
- uvicorn (>=0.23.0) - ASGI server
- python-multipart (>=0.0.6) - Form data handling

### Frontend Framework
- streamlit (>=1.25.0) - Interactive UI framework
- streamlit-webrtc (>=0.47.0) - Real-time video streaming
- av (>=10.0.0) - Audio/video processing
- requests (>=2.31.0) - HTTP client library

### Data Analytics & Visualization
- pandas (>=2.0.0) - Data manipulation
- plotly (>=5.15.0) - Interactive visualizations

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