# StrainVisor

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/your-username/StrainVisor)

**An end-to-end, containerised web platform for calculating material stress-strain curves from high-speed impact footage and sensor data. Currently, it only gives the shape of the curve as the strain is not directly calculated as cross-sectional area and force-transducer coefficient are not taken into account** 🔬

---

## Overview

StrainVisor is a complete data analysis platform designed to automate materials science research. It processes raw data from drop-weight impact tests—consisting of high-speed `.cine` video and `.csv` force-time data—to generate synchronised, accurate stress-strain curves.

The entire application is architected using a modern microservices approach and is fully containerised with Docker for easy deployment and scalability.

### System Architecture

The platform is divided into three independent, communicating services:



1.  **Dash Frontend:** A user-friendly web interface for uploading experimental data, triggering analysis pipelines, and visualising the final stress-strain curve.
2.  **Data Analysis API:** A Python backend service responsible for parsing, cleaning, and processing time-series force data from sensor logs.
3.  **Video Analysis & Sync API:** The core computer vision engine. This powerful Python service implements advanced algorithms to analyse high-speed video, track material deformation, and synchronise the visual data with the force data to produce the final analytical result.

---

## Core Technology & Features

### Features

* **Multi-Modal Data Ingestion:** Seamlessly upload and process high-speed video and corresponding time-series sensor data.
* **Automated Impact Detection:** Intelligently identifies the precise frame of impact by synchronising video contours with force-transducer signal events.
* **Advanced Image Segmentation:** Uses a robust **Watershed Algorithm** pipeline to isolate and track the deforming sample, even after contact with the impactor.
* **Time-Series Synchronisation:** Accurately aligns the strain data (derived from video) with the stress data (derived from sensors) to create a unified dataset.
* **Interactive Visualisation:** Renders the final stress-strain curve in the web UI for immediate analysis.

### Tech Stack

* **Frontend:** Dash, Plotly, Python
* **Backend APIs:** Python, FastAPI
* **Computer Vision:** OpenCV
* **Numerical & Signal Processing:** NumPy, SciPy, Statsmodels, Pandas
* **Containerization:** Docker, Docker Compose

---

## Installation & Usage

This project is fully containerised. To get started, please make sure you have Docker and Docker Compose installed.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/MakMak445/StrainVisor.git](https://github.com/MakMak445/StrainVisor.git)
    cd StrainVisor
    ```

2.  **Build and run the containers:**
    ```bash
    docker-compose up --build
    ```

3.  **Access the application:**
    Open your web browser and navigate to `http://localhost:8050`. You can now upload your data and run the analysis.

---

## Project Structure
┌───────────────────────┐        ┌───────────────────────┐
│     Data Backend      │        │     Video Backend     │
│   (Sensor Parsing)    │        │  (CV & Sync Engine)   │
└──────────┬────────────┘        └────────── ┬───────────┘
           │                                 │
           └────────────────┬────────────────┘
                            │
                  ┌───────────────────────┐
                  │      Frontend UI      │
                  │   (Orchestrator)      │
                  │                       │
                  │     Dash + Plotly     │
                  │       Port 8050       │
                  └───────────────────────┘
```
