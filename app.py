import os
import json
import re
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List

from fastapi import FastAPI, File, UploadFile, Form, Request, Response, HTTPException, Depends, Cookie
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette import status

import auth
from auth import get_current_user, create_access_token, verify_password, hash_password

from core_logic import (
    extract_text_from_pdf_with_docai,
    analyze_pitch_deck,
    transcribe_audio,
    analyze_call_transcript,
    advanced_sector_analysis,
    sentiment_analysis,
    financial_health_check
)

app = FastAPI(title="AI Analyst Prototype")

templates = Jinja2Templates(directory="templates")

SECRET_KEY = "iD-WadfdROQL5BOsvvX9eBEVR5yRUiY-lRy9Zb48sWQ"  
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 5

def detect_sector_from_text(text: str) -> str:
    """
    Automatically detect the industry sector from pitch deck text.
    Returns the most likely sector or 'general' if uncertain.
    """
    text_lower = text.lower()
    
    sector_keywords = {
        "saas": ["saas", "software as a service", "subscription", "mrr", "arr", "churn", "cac", "ltv"],
        "ecommerce": ["ecommerce", "e-commerce", "online store", "shopify", "amazon", "gmv", "aov"],
        "fintech": ["fintech", "financial technology", "banking", "payments", "lending", "insurance", "blockchain"],
        "healthtech": ["healthtech", "health tech", "medical", "healthcare", "fda", "clinical", "telemedicine"],
        "edtech": ["edtech", "education technology", "learning", "course", "online education", "lms"],
        "ai": ["artificial intelligence", "machine learning", "deep learning", "neural network", "ai model"],
        "biotech": ["biotech", "biotechnology", "pharmaceutical", "drug discovery", "genetics", "dna"]
    }
    
    sector_scores = {sector: 0 for sector in sector_keywords.keys()}
    sector_scores["general"] = 0
    
    for sector, keywords in sector_keywords.items():
        for keyword in keywords:
            if keyword in text_lower:
                sector_scores[sector] += 1
    
    if any(word in text_lower for word in ["$", "revenue", "profit", "valuation"]):
        sector_scores["fintech"] += 2
    
    if any(word in text_lower for word in ["patient", "hospital", "clinic", "medical device"]):
        sector_scores["healthtech"] += 2
    
    best_sector = max(sector_scores.items(), key=lambda x: x[1])
    
    if best_sector[1] >= 2 and best_sector[0] != "general":
        return best_sector[0]
    else:
        return "general"

def run_advanced_analyses(deck_text: str, transcript_text: Optional[str] = None, 
                         extracted_facts: Optional[Dict] = None) -> Dict:
    """
    Run all advanced analyses: sector analysis, sentiment analysis, and financial health check.
    """
    advanced_results = {}
    
    try:
        # 1. Detect sector and run sector-specific analysis
        sector = detect_sector_from_text(deck_text)
        advanced_results["detected_sector"] = sector
        advanced_results["sector_analysis"] = advanced_sector_analysis(deck_text, sector)
    except Exception as e:
        advanced_results["sector_analysis_error"] = f"Sector analysis failed: {str(e)}"
    
    try:
        # 2. Run financial health checks if we have extracted facts
        if extracted_facts:
            advanced_results["financial_health_checks"] = financial_health_check(extracted_facts)
    except Exception as e:
        advanced_results["financial_health_error"] = f"Financial health check failed: {str(e)}"
    
    try:
        # 3. Run sentiment analysis if we have transcript
        if transcript_text:
            advanced_results["sentiment_analysis"] = sentiment_analysis(transcript_text)
    except Exception as e:
        advanced_results["sentiment_analysis_error"] = f"Sentiment analysis failed: {str(e)}"
    
    return advanced_results

def init_storage(user_id: str = "anonymous"):
    """Initialize user-specific storage directories"""
    base_dirs = [
        f"storage/users/{user_id}",
        f"storage/users/{user_id}/uploads",
        f"storage/users/{user_id}/uploads/decks",
        f"storage/users/{user_id}/uploads/calls",
        f"storage/users/{user_id}/analysis_results"
    ]
    
    for dir_path in base_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def get_user_storage_path(user_id: str, file_type: str):
    """Get storage path for a user"""
    if file_type == "deck":
        return f"storage/users/{user_id}/uploads/decks"
    elif file_type == "call":
        return f"storage/users/{user_id}/uploads/calls"
    elif file_type == "results":
        return f"storage/users/{user_id}/analysis_results"
    return f"storage/users/{user_id}"

def get_next_analysis_number(analysis_type: str, user_id: str):
    """Finds the next available analysis number for a specific type and user."""
    user_results_dir = Path(f"storage/users/{user_id}/analysis_results")
    user_results_dir.mkdir(parents=True, exist_ok=True)
    
    pattern = f"analysis_*_{analysis_type}_*.json"
    analysis_files = list(user_results_dir.glob(pattern))
    
    if not analysis_files:
        return 1
    
    numbers = []
    for file in analysis_files:
        try:
            num = int(file.stem.split('_')[1])
            numbers.append(num)
        except (IndexError, ValueError):
            continue
    
    return max(numbers) + 1 if numbers else 1

def save_uploaded_file(file: UploadFile, upload_type: str, user_id: str):
    """Save uploaded file to user-specific directory"""
    user_upload_path = get_user_storage_path(user_id, upload_type)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    original_filename = Path(file.filename).stem
    extension = Path(file.filename).suffix
    new_filename = f"{original_filename}_{timestamp}{extension}"
    
    upload_path = Path(user_upload_path) / new_filename
    
    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return str(upload_path)

def save_analysis_results(analysis_data: dict, analysis_number: int, analysis_type: str, user_id: str):
    """Saves analysis results to a descriptive JSON file in user directory."""
    user_results_dir = Path(f"storage/users/{user_id}/analysis_results")
    user_results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"analysis_{analysis_number}_{analysis_type}_{timestamp}.json"
    results_path = user_results_dir / filename
    
    with open(results_path, 'w') as f:
        json.dump(analysis_data, f, indent=2)
    
    return str(results_path)

def determine_analysis_type(deck_file, call_file):
    """Determines the type of analysis based on uploaded files."""
    has_deck = deck_file and deck_file.filename
    has_call = call_file and call_file.filename
    
    if has_deck and has_call:
        return "combined"
    elif has_deck:
        return "deck"
    elif has_call:
        return "call"
    else:
        return "unknown"

@app.get("/register")
async def register_page(request: Request):
    """Serve registration page (GET request)"""
    return templates.TemplateResponse("register.html", {"request": request, "error": None})

@app.post("/register")
async def register_user(
    request: Request,
    username: str = Form(None),
    email: str = Form(None),
    password: str = Form(None),
    confirm_password: str = Form(None)
):
    """Handle user registration (POST request)"""
    if not all([username, email, password, confirm_password]):
        return templates.TemplateResponse("register.html", {
            "request": request,
            "error": "All fields are required"
        })
    
    if password != confirm_password:
        return templates.TemplateResponse("register.html", {
            "request": request,
            "error": "Passwords do not match"
        })
    
    if len(password) < 6:
        return templates.TemplateResponse("register.html", {
            "request": request,
            "error": "Password must be at least 6 characters long"
        })
    
    conn = auth.get_db()
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT id FROM users WHERE username = ? OR email = ?", (username, email))
        if cursor.fetchone():
            return templates.TemplateResponse("register.html", {
                "request": request,
                "error": "Username or email already exists"
            })
        
        hashed_password = hash_password(password)
        cursor.execute(
            "INSERT INTO users (username, email, hashed_password) VALUES (?, ?, ?)",
            (username, email, hashed_password)
        )
        conn.commit()
        
        return RedirectResponse(url="/login?message=Registration+successful", status_code=303)
        
    except Exception as e:
        return templates.TemplateResponse("register.html", {
            "request": request,
            "error": f"Registration failed: {str(e)}"
        })
    finally:
        conn.close()

@app.get("/login")
async def login_page(request: Request, message: str = None, error: str = None):
    """Serve login page"""
    return templates.TemplateResponse("login.html", {
        "request": request,
        "message": message,
        "error": error
    })

@app.post("/login")
async def login_user(
    request: Request,
    username: str = Form(...),
    password: str = Form(...)
):
    conn = auth.get_db()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        if not user or not verify_password(password, user["hashed_password"]):
            return templates.TemplateResponse("login.html", {
                "request": request,
                "error": "Invalid username or password"
            })

        access_token = create_access_token(
            data={"sub": user["username"]},
            expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        )

        redirect_response = RedirectResponse(url="/", status_code=303)
        redirect_response.set_cookie(
            key="token",
            value=access_token,
            httponly=True,
            max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            samesite="lax",
            path="/"
        )
        return redirect_response

    finally:
        conn.close()


@app.get("/logout")
async def logout(response: Response):
    """Logout user by clearing cookie and redirect to login"""
    response.delete_cookie(key="token")
    return RedirectResponse(url="/login?message=Logged+out+successfully", status_code=303)

@app.get("/", response_class=HTMLResponse)
async def root(request: Request, current_user: str = Depends(get_current_user)):
    """Serve the main upload page - requires authentication"""
    if not current_user:
        return templates.TemplateResponse("login.html", {
            "request": request,
            "error": "Please login first"
        })
    
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "current_user": current_user
    })

@app.get("/test-auth")
async def test_auth(current_user: str = Depends(get_current_user)):
    """Test if authentication is working"""
    if current_user:
        return {"message": f"Authenticated as {current_user}"}
    else:
        return {"message": "Not authenticated"}

@app.get("/debug-auth")
async def debug_auth(request: Request, token: str = Cookie(None)):
    """Debug authentication"""
    debug_info = {
        "has_token_cookie": token is not None,
        "token_value": token,
        "cookies_received": request.cookies
    }
    
    if token:
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            debug_info["token_decoded"] = payload
            debug_info["token_valid"] = True
        except Exception as e:
            debug_info["token_error"] = str(e)
            debug_info["token_valid"] = False
    
    return debug_info


@app.get("/debug/cookies")
async def debug_cookies(request: Request):
    """Debug endpoint to check cookies"""
    return {
        "cookies_received": dict(request.cookies),
        "headers": dict(request.headers)
    }

@app.post("/analyze/")
async def analyze_files(
    request: Request,
    deck_file: Optional[UploadFile] = File(None),
    call_file: Optional[UploadFile] = File(None),
    current_user: str = Depends(get_current_user)
):
    """Handles file uploads with user authentication and advanced analysis"""
    if not current_user:
        return RedirectResponse(url="/login", status_code=303)
    
    init_storage(current_user)
    analysis_type = determine_analysis_type(deck_file, call_file)
    analysis_number = get_next_analysis_number(analysis_type, current_user)
    
    analysis_result = {
        "analysis_id": f"{analysis_type}_{analysis_number}",
        "analysis_type": analysis_type,
        "user_id": current_user,
        "timestamp": datetime.now().isoformat(),
        "uploaded_files": {},
        "results": {},
        "advanced_analysis": {}  
    }

    deck_text = None
    transcript_text = None
    extracted_facts = None

    # 1. Process Pitch Deck if uploaded
    if deck_file and deck_file.filename:
        try:
            deck_path = save_uploaded_file(deck_file, "deck", current_user)
            analysis_result["uploaded_files"]["pitch_deck"] = deck_path
            
            deck_text = extract_text_from_pdf_with_docai(deck_path)
            deck_analysis = analyze_pitch_deck(deck_text)
            analysis_result["results"]["deck_analysis"] = deck_analysis
            extracted_facts = deck_analysis.get("extracted_facts")
            
        except Exception as e:
            analysis_result["results"]["deck_error"] = f"Failed to process deck: {str(e)}"
            print(f"Deck processing error: {e}")

    # 2. Process Call Audio if uploaded
    if call_file and call_file.filename:
        try:
            call_path = save_uploaded_file(call_file, "call", current_user)
            analysis_result["uploaded_files"]["founder_call"] = call_path
            
            transcript_text = transcribe_audio(call_path)
            deck_facts = analysis_result["results"].get("deck_analysis", {}).get("extracted_facts") if deck_file and deck_file.filename else None
            call_analysis = analyze_call_transcript(transcript_text, deck_facts)
            analysis_result["results"]["call_analysis"] = call_analysis
            
        except Exception as e:
            analysis_result["results"]["call_error"] = f"Failed to process call: {str(e)}"
            print(f"Call processing error: {e}")

    # 3. Run Advanced Analyses if we have deck text
    if deck_text:
        try:
            advanced_results = run_advanced_analyses(
                deck_text, 
                transcript_text, 
                extracted_facts
            )
            analysis_result["advanced_analysis"] = advanced_results
        except Exception as e:
            analysis_result["advanced_analysis_error"] = f"Advanced analyses failed: {str(e)}"
            print(f"Advanced analysis error: {e}")

    # 4. Save the complete analysis results
    results_path = save_analysis_results(analysis_result, analysis_number, analysis_type, current_user)
    analysis_result["results_path"] = results_path

    # 5. Display the results page
    return templates.TemplateResponse(
        "results.html",
        {
            "request": request,
            "analysis_id": analysis_result["analysis_id"],
            "analysis_type": analysis_type,
            "deck_analysis": analysis_result["results"].get("deck_analysis"),
            "call_analysis": analysis_result["results"].get("call_analysis"),
            "advanced_analysis": analysis_result.get("advanced_analysis", {}),
            "errors": {
                "deck": analysis_result["results"].get("deck_error"),
                "call": analysis_result["results"].get("call_error"),
                "advanced": analysis_result.get("advanced_analysis_error")
            },
            "current_user": current_user
        }
    )

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, current_user: str = Depends(get_current_user)):
    """Display dashboard - requires authentication"""
    if not current_user:
        return RedirectResponse(url="/login", status_code=303)

    analyses = []
    user_results_dir = Path(f"storage/users/{current_user}/analysis_results")
    
    if not user_results_dir.exists():
        user_results_dir.mkdir(parents=True, exist_ok=True)
    
    analysis_files = sorted(user_results_dir.glob("analysis_*.json"), key=os.path.getmtime, reverse=True)
    
    for file_path in analysis_files:
        try:
            with open(file_path, 'r') as f:
                analysis_data = json.load(f)
            
            uploaded_files = analysis_data.get("uploaded_files", {})
            
            analyses.append({
                "id": analysis_data.get("analysis_id", "unknown"),
                "type": analysis_data.get("analysis_type", "unknown"),
                "timestamp": analysis_data.get("timestamp", ""),
                "filename": file_path.name,
                "has_deck": "pitch_deck" in uploaded_files,
                "has_call": "founder_call" in uploaded_files,
                "deck_path": uploaded_files.get("pitch_deck", ""),
                "call_path": uploaded_files.get("founder_call", ""),
                "file_path": str(file_path)
            })
        except Exception as e:
            print(f"Error loading analysis file {file_path}: {e}")
            continue
    
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "analyses": analyses,
            "current_user": current_user
        }
    )

@app.get("/analysis/{filename}", response_class=HTMLResponse)
async def view_analysis(request: Request, filename: str, current_user: str = Depends(get_current_user)):
    """View details of a specific past analysis."""
    if not current_user:
        return RedirectResponse(url="/login", status_code=303)
        
    file_path = Path(f"storage/users/{current_user}/analysis_results") / filename
    
    if not file_path.exists():
        return templates.TemplateResponse(
            "error.html",
            {"request": request, "error": f"Analysis {filename} not found"}
        )
    
    try:
        with open(file_path, 'r') as f:
            analysis_data = json.load(f)
        
        return templates.TemplateResponse(
            "analysis_detail.html",
            {
                "request": request,
                "analysis": analysis_data,
                "filename": filename,
                "current_user": current_user
            }
        )
    except Exception as e:
        return templates.TemplateResponse(
            "error.html",
            {"request": request, "error": f"Error loading analysis: {str(e)}"}
        )

@app.get("/download/{filetype}/{filepath:path}")
async def download_file(filetype: str, filepath: str, current_user: str = Depends(get_current_user)):
    """Download an original uploaded file or analysis results."""
    if not current_user:
        return {"error": "Not authenticated"}
        
    if filetype not in ["deck", "call", "results"]:
        return {"error": "Invalid file type"}
    
    try:
        if filetype == "results":
            file_path = Path(f"storage/users/{current_user}/analysis_results") / filepath
        else:
            file_path = Path(filepath)
        
        if not file_path.exists():
            return {"error": "File not found"}
        
        return FileResponse(
            path=file_path, 
            filename=file_path.name,
            media_type="application/octet-stream"
        )
    except Exception as e:
        return {"error": f"Download error: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)