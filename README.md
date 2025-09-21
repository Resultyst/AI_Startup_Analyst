<div align="center">

# 🧠 AI Analyst | AI-Powered Startup Evaluation

*An AI-powered platform that synthesizes unstructured startup data into concise, actionable investment insights for venture capitalists and angel investors.*

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/) 
[![Google AI](https://img.shields.io/badge/Google%20AI-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev/) 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

[🚀 Features](#-features) • [🛠️ Tech Stack](#-tech-stack) • [📦 Installation](#-installation) • [🎯 Usage](#-usage) • [📖 API](#-api) • [🏗️ Architecture](#-architecture)

</div>

---

## 🤔 The Problem

Early-stage investors are drowning in unstructured data: pitch decks, founder call transcripts, emails, and scattered news reports. Traditional analysis is:
- **Time-consuming**: Hours spent per startup
- **Inconsistent**: Subjective human analysis
- **Prone to error**: Critical red flags are easily missed
- **Non-scalable**: Impossible to evaluate hundreds of opportunities

## 💡 The Solution

**AI Analyst** acts as a tireless, expert investment associate. It ingests raw, unstructured founder materials and public data to generate **consistent, benchmarked, and actionable** deal memos in minutes, not hours.

---

## 🚀 Features

### 🔍 Core Analysis Engine
- **📄 Pitch Deck Intelligence**: Extracts and analyzes key claims, metrics, and business models from PDFs
- **🎤 Founder Call Analysis**: Transcribes audio and evaluates narrative quality, consistency, and passion
- **🔄 Cross-Reference Validation**: Flags contradictions between what's in the deck and what the founder says

### 🧠 Advanced AI Insights
- **🏢 Automatic Sector Detection**: Identifies industry vertical (SaaS, FinTech, HealthTech, etc.) from content
- **📊 Sector-Specific Benchmarking**: Compares metrics against industry peers
- 😊 **Sentiment & Confidence Analysis**: Quantifies founder storytelling quality and conviction
- 💰 **Financial Health Checks**: Automated sanity checks on traction and monetization claims

### 👤 User Experience
- **🔐 Secure Authentication**: JWT-based user accounts with isolated workspaces
- **📁 Intelligent File Management**: Automated organization of analyses and uploads
- **📈 Interactive Dashboard**: Full history of all analyses with search and filter
- **📤 Export Ready**: Download structured JSON reports for investment committees

### ⚡ Platform Capabilities
- **Multi-Modal Analysis**: Process both decks and calls simultaneously
- **Batch Processing**: Handle multiple evaluations in sequence
- **Real-Time Processing**: Live feedback on analysis progress
- **Customizable Weightings**: Tailor analysis to your investment thesis

---

## 🛠️ Tech Stack

| Layer | Technology |
| :--- | :--- |
| **Backend Framework** | FastAPI (Python) |
| **Frontend** | Jinja2 Templates, HTML5, CSS3, JavaScript |
| **AI APIs** | Google Gemini Pro, Document AI, Speech-to-Text |
| **Authentication** | JWT, Bcrypt |
| **Database** | SQLite |
| **Storage** | Local File System (User-Isolated) |
| **Deployment** | Uvicorn ASGI Server |

---


# venv\Scripts\activate  # Windows
pip install -r requirements.txt
