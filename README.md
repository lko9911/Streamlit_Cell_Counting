# 🧪 Malaria Diagnostic System

AI 기반 적혈구 감염 분석 시스템  
(Cell Counting + Parasitemia Estimation)

---

## 📌 Overview

Malaria Diagnostic System은 현미경 혈액 도말 이미지를 기반으로 다음을 수행하는  
Streamlit 기반 웹 애플리케이션입니다:

- 🔬 적혈구 자동 분할 (Cellpose)
- 🧫 말라리아 감염 여부 자동 판별
- 🧑‍⚕️ 수동 보정 기능 제공
- 📊 Parasitemia(감염률) 자동 계산

> ⚠ This system is intended for research and educational purposes only and not for clinical diagnosis.

---

## 🚀 Features

### 1️⃣ AI Cell Segmentation
- 모델: Cellpose
- 적혈구 자동 인식
- 개별 세포 ID 생성

### 2️⃣ Automatic Infection Detection
- HSV 기반 parasite 색상 탐지
- 세포 내부 보라색 픽셀 수 기반 감염 판정

### 3️⃣ Manual Correction Mode
- 감염 토글 기능
- 유효 RBC 제외 기능
- 실시간 감염률 재계산

### 4️⃣ Diagnostic Summary
- Valid RBC Count
- Infected RBC Count
- Parasitemia (%)
- Risk Level Classification

---

## 🖥 Demo UI Structure
<code><pre>Sidebar
├── Image Upload
├── AI Analysis Button
├── Manual Correction Mode

Main Panel
├── Segmentation Result Image
├── Click-based Cell Editing
├── Diagnostic Summary</pre></code>

---

## 📦 Tech Stack

- Python
- Streamlit
- OpenCV
- NumPy
- Cellpose

---

## 🧠 Infection Detection Logic

1. RGB → HSV 변환
2. Parasite color range 추출
3. 세포 내부 픽셀 수 계산
4. Threshold 초과 시 감염 판정

<code><pre>if parasite_pixels > threshold:
    infected_cells.add(cell_id)</pre></code>

---

## 📊 Parasitemia Calculation
<code><pre>Parasitemia (%) = (Infected RBC / Valid RBC) × 100</pre></code>

---

## 📊 Risk Classification

| Parasitemia | Risk Level |
| ----------- | ---------- |
| < 1%        | Low        |
| 1–5%        | Moderate   |
| > 5%        | High       |

---

## ▶ Installation

<code><pre>git clone https://github.com/your-repo/malaria-diagnostic-system.git
cd malaria-diagnostic-system
pip install -r requirements.txt
streamlit run streamlit_app.py</pre></code>

### 📁 Project Structure


<code><pre>malaria-diagnostic-system/
│
├── streamlit_app.py
├── requirements.txt
└── README.md</pre></code>

---

## 💡 About This Project

This project demonstrates the integration of deep learning-based cell segmentation and interactive medical image analysis using Streamlit.

It combines automated AI detection with human-in-the-loop correction to improve diagnostic reliability.


