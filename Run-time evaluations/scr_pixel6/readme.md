
# ⚙️ CRoPP Runtime Evaluation on Google Pixel 6 using Termux

This repository provides scripts for training and inference of deep learning models (DNN and BiLSTM) directly on a Google Pixel 6 using **Termux**. It includes runtime measurement of memory usage, energy consumption, and execution time on the mobile CPU.

---

## 📱 Setup Instructions

### 1. 📲 Install Termux

Download and install Termux from F-Droid:

- Link: [https://f-droid.org/en/packages/com.termux/](https://f-droid.org/en/packages/com.termux/)

> ⚠️ The termux version installed from the Play Store is outdated so the above version is preferred.

---

### 2. ⚙️ Prepare Termux Environment

Open Termux and run:

```bash
pkg update -y && pkg upgrade -y
pkg install -y python clang git wget curl libandroid-glob termux-api proot
```

Allow storage access:

```bash
termux-setup-storage
```

---

### 3. 📁 Transfer Project Files

Move the following files to your Pixel 6 (via `adb` or direct download):

```
CRoPP/
├── CRoPP_EDA_training_Pixel6.py
├── CRoPP_EDA_inference_Pixel6.py
├── CRoPP_Percept_training_Pixel6.py
├── CRoPP_Percept_inference_Pixel6.py
├── requirements.txt
```

Using ADB:

```bash
adb push CRoPP/ /sdcard/Download/
```

Then in Termux:

```bash
cd ~/storage/downloads/CRoPP/
```

---

## 📦 Install Python Dependencies

Make sure `pip` is updated:

```bash
pip install --upgrade pip
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Run the Code

Run any of the following depending on the task:

### Train on EDA:
```bash
python3 CRoPP_EDA_training_Pixel6.py
```

### Inference on EDA:
```bash
python3 CRoPP_EDA_inference_Pixel6.py
```

### Train on Perceptual Features:
```bash
python3 CRoPP_Percept_training_Pixel6.py
```

### Inference on Perceptual Features:
```bash
python3 CRoPP_Percept_inference_Pixel6.py
```

---

## 📊 Output

After running, you’ll get output like:

```
Training Time: 43.2 seconds
Used Memory: 0.57 GB (583.61 MB)
Energy Consumption: 9149.65 Joules
Utilization: 75.3 %
```

These statistics are automatically printed at the end of each script execution.

---

## 💡 Tips

- For model transfer:
```bash
adb pull /sdcard/Download/CRoPP/
```


---

## ✅ Verified On

- Device: **Google Pixel 6**
- Android: **14**
- Python: **3.12 (Termux)**
- Backend: **CPU (no GPU)**

---

## 📁 Directory Summary

```
CRoPP/
├── CRoPP_EDA_training_Pixel6.py
├── CRoPP_EDA_inference_Pixel6.py
├── CRoPP_Percept_training_Pixel6.py
├── CRoPP_Percept_inference_Pixel6.py
├── requirements.txt
```

---

## 📦 Optional: Package List Reference

The file `termux-packages.list` is included to document the Termux environment used during the experiments.
This file lists all manually installed packages via `pkg`.

You can recreate the environment using:

```bash
xargs pkg install -y < termux-packages.list
```

Or refer to it to verify compatibility with your local setup.

