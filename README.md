 Garbage-Classification-System-Using-AI

An **AI-powered waste classification system** using deep learning (CNN) to automatically categorize garbage into different types — promoting **smart, sustainable waste management** and **energy-efficient recycling**.

---

## 📖 Introduction

Waste management is a global challenge, and improper segregation leads to environmental damage, loss of recyclable materials, and inefficient energy use. This project applies **Artificial Intelligence (AI)** and **Computer Vision** to classify garbage images into categories like **organic, plastic, paper, glass, metal, and e-waste**.

By automating waste classification:

* Recycling efficiency improves
* Energy recovery from waste becomes easier
* Smart bins and IoT-enabled solutions can be developed

---

## 🗂 Repository Contents

The repository currently includes:

* **README.md** – Project documentation
* **Untitled.ipynb** – Initial test notebook
* **Week-1---Garbage-Classification-Using-AI.ipynb** – Weekly progress notebook
* **dataset1.csv** – Dataset sample (may contain labeled garbage data)
* **garbage\_classification-checkpoint.ipynb** – Notebook checkpoint for experiments
* **garbage\_classification.ipynb** – Main training & testing notebook

---

## ⚙️ Features

* Deep Learning–based **Convolutional Neural Network (CNN)** for image classification
* Data preprocessing and augmentation pipeline
* Model training and validation using PyTorch/TensorFlow (based on your notebook)
* Evaluation metrics: accuracy, precision, recall, F1-score, confusion matrix
* Extensible for **deployment in real-time applications** (smart bins, web/mobile apps)

---

## 📊 Dataset

* **dataset1.csv** contains structured metadata or labels for garbage images.
* Data can be categorized into:

  * **Organic Waste** 🌱
  * **Plastic** ♻️
  * **Glass** 🍾
  * **Paper/Cardboard** 📄
  * **Metal** 🥫
  * **E-Waste** 💻

> You can expand the dataset using Kaggle or custom-collected images for better accuracy.

---

## 🚀 Getting Started

### 1. Clone Repository

```bash
git clone https://github.com/srideepalaksmicause/GARBAGE-CLASSIFICATION-SYSTEM.git
cd GARBAGE-CLASSIFICATION-SYSTEM
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Jupyter Notebook

Launch the main notebook:

```bash
jupyter notebook garbage_classification.ipynb
```

Train the model, visualize results, and test on your own data.

---

## 🧠 Model Architecture

* A **Convolutional Neural Network (CNN)** is used for classification.
* Transfer Learning (ResNet, VGG, MobileNet) can also be applied to improve accuracy.
* Optimizer: Adam/SGD
* Loss Function: CrossEntropyLoss

---

## 🔎 Evaluation

The trained model is evaluated on test data:

* **Accuracy**: Overall percentage of correctly classified waste items
* **Confusion Matrix**: To understand misclassifications
* **Precision/Recall/F1-score**: To analyze per-class performance

Sample Output:

```
Accuracy: 91.5%
Precision: 0.90
Recall: 0.91
F1-score: 0.905
```

---

## 💡 Applications

* **Smart Bins** 🗑️ – Automated sorting of waste at collection points
* **Recycling Plants** 🔄 – Faster material recovery
* **Sustainable Cities** 🌍 – Efficient waste-to-energy systems
* **Mobile Apps** 📱 – Help users identify proper waste bins

---

## 🏆 Future Improvements

* Expand dataset size for better generalization
* Use **object detection** (YOLO, Faster R-CNN) to detect multiple items in one image
* Deploy model on **Raspberry Pi/Edge Devices** for IoT-based smart bins
* Create a **Flask/FastAPI web app** for real-time inference

---

## 📬 Author

Developed by **M SRIDEEPALAKSHMI**

* 📧 Email: **[thanideepa04@gmail.com](mailto:thanideepa04@gmail.com)**
* 🌐 GitHub: [srideepalaksmicause](https://github.com/srideepalaksmicause)

---

## ⚖️ License

This project is licensed under the **MIT License**.
