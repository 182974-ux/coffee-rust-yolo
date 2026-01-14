## Introduction
we present YOLO-CoffeeRust, a deep learning–based detection model specifically designed for the automatic identification of coffee leaf rust.
YOLO-CoffeeRust extends the YOLOv8 backbone to accurately detect rust-affected regions on coffee leaves,
while retaining the original object detection branch for real-time disease identification.
The main contributions of YOLO-CoffeeRust are as follows:
*	**Adapting the YOLOv8 architecture for coffee leaf rust detection**
*	**Developing a robust object detection model for identifying rust-affected regions**
*	**Training and evaluating the model using annotated coffee leaf datasets**
*	**Achieving accurate and efficient coffee rust detection for agricultural applications**
---
![jpeg](Images/Arquitecture.jpeg)
<div align="center"> <h4>The architecture of YOLO </h4> </div>

### Requirements
1. Python 3.8+
2. OpenCV 4.5+
3. PyTorch 1.8+
4. NumPy
5. Ultralytics (YOLOv8)

## How to use?
1. Clone the repository

  `git clone https://github.com/182974-ux/coffee-rust-yolo.git`
  
2. Install the required libraries

  `pip install -r requirements.txt`
  
3. Run the main file

  `python main.py`


---

### Visualization
![gif](Images/V1.gif)

### Citation