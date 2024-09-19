# CodeAlpha Object Detection and Tracking

This repository contains a project for object detection and tracking using the YOLOv8 model. It includes frame capture from videos, object detection using a pre-trained YOLOv8 model, and custom object tracking. Below are details of each file in the repository.

## Files

### 1. `capture_frames.py`
This script captures frames from the video `surf.mp4` and saves them as images.

- Captures up to 100 frames from the video.
- Frames are resized, flipped, and saved in the `images-surf/` directory as `.jpg` files.
  
#### Usage:
Run this script to capture frames from the video:
```bash
python capture_frames.py
```

### 2. `tracker.py`
A custom tracker class that keeps track of objects across video frames. It uses the Euclidean distance between object centers to maintain consistent IDs for each object.

#### Features:
- **Object Center Tracking**: Each object is assigned an ID based on its center coordinates.
- **New Object Detection**: New objects are assigned unique IDs.
- **ID Consistency**: The object’s ID is maintained across frames if detected within a defined threshold distance.

#### Usage:
To run the tracker independently, execute:
```bash
python tracker.py
```

### 3. `object_detection_tracking.py`
This is the core script that integrates YOLOv8 object detection and tracking.

#### Features:
- **YOLOv8 Detection**: Utilizes the YOLOv8 model for object detection from the video `surf.mp4`.
- **Custom Tracker Integration**: Tracks detected objects across frames using the `Tracker` class.
- **Real-Time Processing**: Displays the processed video with bounding boxes and object IDs in real-time.

#### Usage:
Ensure the necessary models and dependencies are installed and run the script:
```bash
python object_detection_tracking.py
```

### 4. `yolov8_object_detection_on_custom_dataset.ipynb`
This Jupyter Notebook walks through installing YOLOv8 and running it on a custom dataset.

#### Features:
- **YOLOv8 Installation**: Step-by-step instructions for installing YOLOv8.
- **Pre-trained Model Inference**: Demonstrates using YOLOv8’s pre-trained model for inference.
- **Custom Training**: Walkthrough of training YOLOv8 on a custom dataset (e.g., `freedomtech.zip`).

#### How to Use:
1. Run the notebook cells to set up YOLOv8.
2. Follow instructions to train YOLOv8 on your custom dataset.

---

## Setup and Requirements

### Prerequisites
- **Python 3.x**
- **OpenCV**
- **Pandas**
- **NumPy**
- **Ultralytics YOLOv8**

### Installation

1. Install the required Python libraries:
   ```bash
   pip install ultralytics==8.0.20 opencv-python pandas numpy
   ```

2. (Optional) If working with the Jupyter notebook, ensure you have Jupyter installed:
   ```bash
   pip install notebook
   ```

## Running the Project

1. **Capture Frames**:
   Run `capture_frames.py` to extract frames from the video file `surf.mp4`.

2. **Object Detection and Tracking**:
   Run `object_detection_tracking.py` to perform object detection and tracking on the video in real-time.

3. **Train YOLOv8 on a Custom Dataset**:
   Follow the steps in `yolov8_object_detection_on_custom_dataset.ipynb` to train the YOLOv8 model on your custom dataset.

---

## Guidance and Learning

This project was completed as part of an internship with **Code Alpha**, where self-driven learning is emphasized. While no direct guidance was provided during the process, the experience allowed for deep self-learning through problem-solving and independent research.

Resources such as YouTube tutorials and GitHub repositories were utilized to complete the object detection and tracking implementation, specifically leveraging guidance from a [YouTube video](https://youtu.be/-CGr7ryOH98?si=2KMKdZx_5Hs4kU0F) and its associated [GitHub repository](https://github.com/freedomwebtech/yolov8-custom-object-training-tracking).

---

## Acknowledgments

- YOLOv8 model provided by [Ultralytics](https://ultralytics.com).
- Training and object detection workflow inspired by community tutorials and resources.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE).


## Contact

For questions or comments, please reach out to [us](khushnoor1.ggitbca.2020@gmail.com)

