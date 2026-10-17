# 🏏 Jersey Number Detection & Player Tracking using TensorFlow

A Deep Learning project built with **TensorFlow** featuring a **Custom Self-Trained CNN (Convolutional Neural Network)** for automated sports analytics, player bounding box detection, and jersey number identification.

---

## 📌 Project Overview
Manual player tracking and jersey number identification in sports (like football and basketball) is time-consuming and prone to errors. This project automates the workflow by implementing a multi-task deep learning model trained from scratch.

- **Goal:** Automate player location detection and jersey number classification for sports analytics.
- **Model Type:** **Custom Self-Trained CNN** (Built from scratch, *not* a pre-trained or transfer-learning model).
- **Dataset:** Roboflow dataset structured in TFRecord format.
- **Input Size:** 128x128 RGB images.
- **Backbone Architecture:** Custom convolutional layers with `Conv2D`, `MaxPool`, and `GlobalAveragePooling` layers.
- **Outputs (Multi-Task Learning):** 
  1. **Bounding Box Regression:** 4 coordinates using sigmoid activation to locate players.
  2. **Classification:** Binary classification using sigmoid activation to identify jersey numbers.
- **Loss Functions:**
  - Bounding Box: Mean Squared Error (MSE)
  - Classification: Binary Crossentropy
- **Optimizer & Metrics:** Adam Optimizer, evaluation via MSE for bounding box regression and Binary Accuracy for classification.

---

## 📂 Repository Structure
```
jersey-number-detection/
│
├── notebooks/
│   └── player-detection-using-cnn.ipynb     # Self-training & evaluation notebook
├── presentation/
│   └── 2025_PlayerDetection_JerseyNumber_TensorFlow_Presentation_FINAL.pptx  # Presentation slides
├── .gitignore
├── LICENSE                                  # MIT
├── README.md                                # Project documentation
└── requirements.txt                         # Python dependencies
```

---

## ⚙️ Installation & Requirements
To set up the environment and install dependencies, run:

```bash
pip install -r requirements.txt
```

---

## 📦 Dataset

The dataset is a Roboflow export in TFRecord format. It is not included in this repository. Create your own download link on Roboflow Universe and set it as the `ROBOFLOW_URL` variable before running the notebook. Never commit API keys or download tokens.

---

## 🚀 Usage
1. Open Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Navigate to the `notebooks/` directory and open `player-detection-using-cnn.ipynb`.
3. Execute the cells sequentially to run the data pipeline, parse TFRecord files, construct and train the custom CNN, and visualize bounding box predictions on validation images.

---

## 📄 Licence

Released under the [MIT Licence](LICENSE).

## 👨‍💻 Author
**Muzammal Hussain**, Software Engineer and Data Scientist
