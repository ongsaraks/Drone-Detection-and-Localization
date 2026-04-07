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

## Quick Start (Necessary Steps Only)

### 1. Requirement
- Docker Desktop (with Docker Compose)
- Conda (to run `Integrate.py` locally)

### 2. Start Web Stack (Docker)
Run from repository root:

```bash
docker compose -f .\TESA19_2025_G38_v2\docker-compose.yml build
docker compose -f .\TESA19_2025_G38_v2\docker-compose.yml up -d
```

This starts backend, database, phpMyAdmin, and frontend.

### 3. Open The System
- Frontend: http://localhost:5173
- Backend API: http://localhost:3000
- phpMyAdmin: http://localhost:8080

### 4. Run Integrate.py Locally (After Docker Is Up)
From repository root:

```bash
conda create -n Drone python=3.11 -y
conda run -n Drone pip install -r requirements.txt
conda run -n Drone python -u .\Integrate.py
```

### 5. If Login Fails (Run Once)
In phpMyAdmin, database `mydb`, run:

```sql
DELETE FROM users WHERE username = 'admin';
INSERT INTO users (username, password_hash, role)
VALUES ('admin', '$2a$10$Z8isJfWJ/P5t5mk6c7.vHurPV2nqSB25eIdi.PjNDhhd59vUYnBSy', 'admin');
```

Username: admin
Password: 1234

### 6. Stop

```bash
docker compose -f .\TESA19_2025_G38_v2\docker-compose.yml down
```
