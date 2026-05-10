# ✋ Sign language detetcton system – Real-Time Hand Gesture Recognition

this is a real-time hand gesture recognition system using a webcam to identify five predefined gestures: "Hello", "Yes", "No", "Thanks", and "Sorry". It leverages **MediaPipe** for hand landmark detection and a machine learning pipeline involving **PCA** and **Random Forest Classifier** to make accurate predictions in real time.

---

## 🧠 How It Works

1. Captures hand landmarks (21 points) in 3D using **MediaPipe Hands**
2. Flattens and scales the data using **StandardScaler**
3. Applies **PCA** to reduce dimensions while retaining 95% of variance
4. Trains a **Random Forest Classifier** (with GridSearch hyperparameter tuning)
5. Uses **OpenCV** to display predictions in real-time along with confidence scores

---

## 📁 Dataset

- File: `signfinity/hand_sign_landmarks.csv`
- Format:
  - Column 0: Label (1 to 5)
  - Columns 1–63: Flattened (x, y, z) coordinates from 21 landmarks

### Label Mapping

| Label | Gesture |
|-------|---------|
| 1     | No      |
| 2     | Sorry   |
| 3     | Thanks  |
| 4     | Yes     |
| 5     | Hello   |

---

## 📦 Requirements

Install dependencies with pip:

```bash
pip install opencv-python mediapipe scikit-learn numpy pandas
```

---

## 🚀 Usage

1. Ensure your dataset is in the correct path: `signfinity/hand_sign_landmarks.csv`
2. Run the script:

```bash
python signfinity.py
```

3. The webcam will launch and begin gesture detection. Detected gestures will be displayed on the screen along with a confidence score.

---

## 🧪 Model Training

The Random Forest model is trained using a pipeline that includes:

- **StandardScaler** for normalization  
- **PCA** for dimensionality reduction (retain 95% variance)
- **GridSearchCV** for hyperparameter optimization

### Example best parameters:

```python
{
  'n_estimators': 150,
  'max_depth': 20,
  'min_samples_split': 2,
  'min_samples_leaf': 1
}
```

---

## 🖥️ Real-Time Display

- Live webcam feed with gesture label and confidence score
- Bounding box around detected hand using **OpenCV**
- Gestures supported:
  - ✋ Hello
  - 👍 Yes
  - 👎 No
  - 🙏 Thanks
  - 😔 Sorry

---

## 📈 Output Example

```
Recognized gesture: Hello, Confidence: 0.96
Recognized gesture: Thanks, Confidence: 0.91
```

---

## 📂 Project Structure

```
signfinity/
├── hand_sign_landmarks.csv     # Training data (landmark features + labels)
├── signfinity.py               # Main script for gesture recognition
├── README.md                   # Project documentation
```

---

## 📌 Limitations

- Only supports **single-hand detection** at a time
- Lighting and hand visibility affect recognition accuracy
- Not a full ASL interpreter — designed for 5 static gestures

---

## ✅ Possible Improvements

- Add more gestures and label categories
- Use temporal data and deep learning (e.g., LSTM or CNN)
- Export trained model for reuse without retraining
- Add GUI for dataset generation and live testing

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

---

## 👨‍💻 Author

Kartikay Dimri
gmail-dimrikartikay@gmail.com
---

## ⭐️ Support

If you found this project helpful, please consider giving it a ⭐ on GitHub and sharing it!

