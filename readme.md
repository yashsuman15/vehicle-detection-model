# Vehicle Detection Model with Class Counting

This project implements a real-time vehicle detection and counting system using YOLOv11 and OpenCV. It processes video input to detect various types of vehicles, draw bounding boxes around them, and display a live count for each vehicle class.

## Features

- Real-time vehicle detection using YOLOv8
- Classification of multiple vehicle types (e.g., cars, buses, trucks)
- Color-coded bounding boxes for each vehicle class
- Live counting of detected vehicles by class
- On-screen legend showing vehicle counts
- Confidence threshold for reducing false positives

## Preview
1.added counting each vehicle class

![image](https://github.com/user-attachments/assets/cfc2fcc3-20d3-4e59-a5bd-7f16b1057a29)

2.detection boxes with class name

![adit_mp4-289_jpg rf 47dc54510dd7ae00206e5808e58c898c](https://github.com/user-attachments/assets/ee3fe9de-3115-4c49-b534-b4de5c1ad212)


## Requirements

- Python 3.11
- OpenCV (`cv2`)
- Ultralytics YOLO (`ultralytics`)
- cvzone

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yashsuman15/vehicle-detection-model.git
   cd vehicle-detection-model
   ```

2. Install the required packages:
   ```
   pip install opencv-python ultralytics cvzone
   ```

3. Download the YOLOv8 weights file and place it in the appropriate directory (update the path in the script if necessary).

## Usage

1. Update the paths in the script:
   - Set the correct path for the YOLOv8 weights file
   - Set the correct path for the input video file

2. Run the script:
   ```
   python vehicle-detection-model-with-class-counting.py
   ```

3. The program will open a window showing the processed video with bounding boxes and vehicle counts. Press 'q' to quit the application.

## Customization

- Adjust the `confidence` threshold in the script to fine-tune detection sensitivity
- Modify the `class_colors` dictionary to change the color scheme for different vehicle types
- Add or remove vehicle classes as needed, ensuring the model is trained accordingly

## Contributing

Contributions to improve the project are welcome. Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for the YOLOv11 implementation
- [cvzone](https://github.com/cvzone/cvzone) for enhanced OpenCV utilities

