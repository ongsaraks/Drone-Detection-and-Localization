# Drone Detection and Localization

This project explores detecting and localizing drones using computer vision and machine learning techniques.

---

## Problem Statement
Drone detection is important for security and monitoring applications.  
This project investigates how vision-based models can detect drones and estimate their location.

---

## What We Did
- Implemented a detection model to identify drones in visual input
- Designed a pipeline to track detected drones
- Applied regression models to estimate latitude and longitude

---

## Technologies Used
- YOLO
- Gradient Boosting Regressor
- Python

---

## Outcome
- The system can detect, track, and estimate drone locations
- Demonstrates integration of detection and regression models

---

## Notes
Localization accuracy depends on environmental assumptions and training data.  
This project is exploratory and intended for research and learning.

---

## How To Run From Scratch

### 1. Prerequisites
- Python 3.11+
- Conda (Miniconda or Anaconda)
- Docker Desktop (with Docker Compose)
- Node.js 18+ and npm

### 2. Project Structure You Will Use
- Root: Python detection/localization pipeline (`Integrate.py`)
- `TESA19_2025_G38_v2`: Backend + MySQL + frontend app

### 3. Start Backend + Database
From the repository root:

```bash
docker compose -f .\TESA19_2025_G38_v2\docker-compose.yml up --build -d
```

This starts:
- Backend API: http://localhost:3000
- MySQL: localhost:3306
- phpMyAdmin: http://localhost:8080

### 4. Prepare Python Environment
From the repository root:

```bash
conda create -n Drone python=3.11 -y
conda run -n Drone pip install -r requirements.txt
```

Recommended (for better Socket.IO transport support):

```bash
conda run -n Drone pip install websocket-client
```

### 5. Create/Verify Admin User (if login fails)
Open phpMyAdmin (http://localhost:8080), choose database `mydb`, then run:

```sql
DELETE FROM users WHERE username = 'admin';
INSERT INTO users (username, password_hash, role)
VALUES ('admin', '$2a$10$Z8isJfWJ/P5t5mk6c7.vHurPV2nqSB25eIdi.PjNDhhd59vUYnBSy', 'admin');
```

Login credentials:
- Username: `admin`
- Password: `1234`

### 6. Run Detection + Localization Pipeline
From the repository root:

```bash
conda run -n Drone python -u .\Integrate.py
```

Expected startup logs include:
- `Authentication successful`
- `Socket.IO connected successfully`
- `Starting video processing`

### 7. Run Frontend Dashboard

```bash
cd .\TESA19_2025_G38_v2\my-vite-app
npm install
npm run dev
```

Open: http://localhost:5173

### 8. Stop Everything

```bash
docker compose -f .\TESA19_2025_G38_v2\docker-compose.yml down
```
