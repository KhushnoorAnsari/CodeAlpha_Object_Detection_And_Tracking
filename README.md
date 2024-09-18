# CodeAlpha Object Detection and Tracking

## Overview

This project involves object detection and tracking using the YOLOv8 model and a custom tracker implementation. It processes video frames to detect and track objects, drawing bounding boxes around them and assigning unique IDs for tracking.

### Features
- Object detection with YOLOv8.
- Real-time object tracking with a custom tracker.
- Frame capture and saving from video.

## Installation

### Prerequisites

- Python 3.8 or higher
- Required Python libraries (see `requirements.txt`)

### Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/KhushnoorAnsari/KhushnoorAnsari.git
   ```

2. **Navigate to the project directory:**

   ```bash
   cd CodeAlpha_Object_Detection_And_Tracking
   ```

3. **Install the required packages:**

   You can install the required packages using `pip`:

   ```bash
   pip install -r requirements.txt
   ```

4. **Download the YOLOv8 model:**

   Place your YOLOv8 model file (`best.pt`) in the project directory.

5. **Prepare your class labels:**

   Ensure you have a file named `Coco.txt` with your class labels, one per line.

## Usage

### Capturing Frames from Video

To capture frames from a video and save them as images, run the `img.py` script:

```bash
python img.py
```

This script will save frames from the video `surf.mp4` to the `images-surf` directory.

### Object Detection and Tracking

To perform object detection and tracking on a video, run the `detect.py` script:

```bash
python detect.py
```

This script uses the YOLOv8 model to detect objects in `surf.mp4` and tracks them using the custom `Tracker` class.

## Code Structure

- `img.py`: Captures frames from a video and saves them as images.
- `tracker.py`: Defines a custom `Tracker` class for object tracking.
- `detect.py`: Performs object detection and tracking, displaying results in real-time.
- `best.pt`: Pre-trained YOLOv8 model file.
- `Coco.txt`: File containing class names used for object detection.

## Requirements

The project depends on the following Python libraries:

- `opencv-python`
- `pandas`
- `numpy`
- `ultralytics`

You can install these libraries using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Guidance

This project was completed with guidance from the following resources:

- YouTube Video: [YOLOv8 Custom Object Training & Tracking](https://youtu.be/-CGr7ryOH98?si=2KMKdZx_5Hs4kU0F)
- GitHub Repository: [freedomwebtech/yolov8-custom-object-training-tracking](https://github.com/freedomwebtech/yolov8-custom-object-training-tracking)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE).

## Acknowledgments

- YOLOv8 for object detection.
- OpenCV for computer vision functionalities.

## Contact

For questions or comments, please reach out to khushnoor1.ggitbca.2020@gmail.com
