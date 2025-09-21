import os
import google.generativeai as genai
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from google.api_core.client_options import ClientOptions
from google.cloud import documentai
import json
import re
from google.cloud import speech
import io


load_dotenv()

genai.configure(api_key=os.getenv('GEMINI_API_KEY'))


def extract_text_from_pdf_with_docai(file_path):
    """
    Extracts text from a local PDF using Google's Document AI API.
    """
    project_id = os.getenv("DOCAI_PROJECT_ID")
    location = os.getenv("DOCAI_LOCATION")
    processor_id = os.getenv("DOCAI_PROCESSOR_ID")
    
    if None in (project_id, location, processor_id):
        raise ValueError("Please set DOCAI_PROJECT_ID, DOCAI_LOCATION, and DOCAI_PROCESSOR_ID in your .env file.")

    endpoint = f"{location}-documentai.googleapis.com"
    client_options = ClientOptions(api_endpoint=endpoint)
    
    client = documentai.DocumentProcessorServiceClient(client_options=client_options)
    processor_name = client.processor_path(project_id, location, processor_id)

    with open(file_path, "rb") as file:
        file_content = file.read()

    raw_document = documentai.RawDocument(
        content=file_content,
        mime_type="application/pdf",  s
    )

    request = documentai.ProcessRequest(
        name=processor_name, 
        raw_document=raw_document
    )

    print("Sending document to Document AI API...")
    result = client.process_document(request=request)
    
    full_text = result.document.text
    print(f"Successfully extracted {len(full_text)} characters.")
    
    return full_text


def transcribe_audio(audio_file_path):
    """
    Transcribes a short audio file (e.g., < 1 minute) using Google Speech-to-Text API.
    For longer files, you would use asynchronous recognition.
    """
    client = speech.SpeechClient()

    with io.open(audio_file_path, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.ENCODING_UNSPECIFIED,
        language_code="en-US",  
        sample_rate_hertz=16000, 
    )

    print("Sending audio for transcription...")
    response = client.recognize(config=config, audio=audio)

    transcript = ""
    for result in response.results:
        transcript += result.alternatives[0].transcript + " "

    print(f"Transcription complete. Got {len(transcript)} characters.")
    return transcript.strip()

def analyze_call_transcript(transcript_text, deck_facts=None):
    """
    Analyzes a founder call transcript for qualitative insights and consistency with the deck.
    'deck_facts' is optional for cross-referencing.
    """
    model = genai.GenerativeModel('gemini-2.5-pro')

    cross_reference_context = ""
    if deck_facts:
        cross_reference_context = f"""
        Additionally, compare the statements in this call with the following facts claimed in their pitch deck:
        {json.dumps(deck_facts, indent=2)}
        Note any major contradictions or reinforcements.
        """

    analysis_prompt = f"""
    You are an expert VC analyst. Analyze the following transcript from a founder investor call.

    <transcript>
    {transcript_text}
    </transcript>

    Focus on qualitative aspects and narrative. 
    {cross_reference_context}

    Output ONLY a JSON object with the following keys:
    - "founder_story_quality": (string: 'compelling', 'average', 'unconvincing') - Assess the clarity and passion of the narrative.
    - "key_theme": (string) - What was the single most emphasized topic?
    - "notable_investor_questions": (list of strings) - What insightful questions did investors ask?
    - "consistency_with_deck": (list of strings) - Note any specific contradictions or confirmations vs. the deck. Omit if no deck data was provided.
    - "call_specific_risks": (list of strings) - e.g., "Founder was evasive on unit economics.", "Admitted to a key technical challenge."
    - "call_specific_strengths": (list of strings) - e.g., "Articulated a clear vision for the next 18 months.", "Demonstrated deep customer empathy."

    """
    analysis_response = model.generate_content(analysis_prompt)
    return parse_json_response(analysis_response.text)


def analyze_pitch_deck(deck_text):
    """
    NEW: A more robust analysis using a multi-prompt approach.
    Returns a dictionary with the combined analysis.
    """
    model = genai.GenerativeModel('gemini-2.5-pro')

    # PROMPT 1: FACT EXTRACTION
    print("1. Extracting key facts...")
    extraction_prompt = f"""
    Read the following pitch deck text carefully. Your task is to act as a data extractor and list all key facts, figures, and claims exactly as they are presented.

    <deck_text>
    {deck_text}
    </deck_text>

    Extract the following information. If information is not available, use "Not Provided".
    - Company Name
    - One-line Description
    - Problem Statement
    - Proposed Solution
    - Target Market Size (TAM/SAM/SOM)
    - Business Model (How they make money)
    - Key Metrics (Traction, MRR, Users, Growth etc.)
    - The Team (Founders and key members)
    - Amount they are raising (e.g., "Seeking $500K Seed")

    Output ONLY a JSON object with keys: "company_name", "one_liner", "problem", "solution", "tam", "business_model", "traction", "team", "ask".
    """
    extraction_response = model.generate_content(extraction_prompt)
    extracted_facts = parse_json_response(extraction_response.text)

    # PROMPT 2: CRITICAL ANALYSIS
    print("2. Analyzing for strengths and risks...")
    analysis_prompt = f"""
    You are a critical VC analyst. Based on the following extracted facts from a pitch deck, identify the most compelling strengths and the most serious potential red flags.

    <extracted_facts>
    {json.dumps(extracted_facts, indent=2)}
    </extracted_facts>

    For your analysis, follow these rules:
    - Be critical but fair.
    - For each strength and risk, cite the specific fact that led you to that conclusion.
    - Risks should be specific and actionable (e.g., not "market is risky", but "Claimed TAM of $500B is not cited and seems inflated for a niche solution").
    - Strengths should be significant (e.g., not "good team", but "Founder is a former Google PM with 10 years of domain experience").

    Output ONLY a JSON object with the following keys:
    - "strengths": (list of strings)
    - "red_flags": (list of strings)
    - "investor_questions": (list of critical questions an investor should ask the founder based on the risks)
    """
    analysis_response = model.generate_content(analysis_prompt)
    critical_analysis = parse_json_response(analysis_response.text)

    # PROMPT 3: FINAL SYNTHESIS
    print("3. Generating final summary...")
    synthesis_prompt = f"""
    You are a senior investment partner. Write a concise, three-paragraph investment memo based on the following facts and analysis.

    **Company Facts:**
    {json.dumps(extracted_facts, indent=2)}

    **VC Analysis:**
    {json.dumps(critical_analysis, indent=2)}

    **Structure your memo as follows:**
    Paragraph 1 (The Deal): Summarize what the company does, the problem it solves, and the investment ask.
    Paragraph 2 (The Upside): Summarize the core strengths and compelling reasons to invest.
    Paragraph 3 (The Risk): Summarize the key risks and critical questions that need to be answered.

    Output ONLY the text of the three-paragraph memo. Do not use headings.
    """
    synthesis_response = model.generate_content(synthesis_prompt)
    final_summary = synthesis_response.text.strip()

    # COMBINE ALL RESULTS INTO ONE DICTIONARY
    full_analysis = {
        "extracted_facts": extracted_facts,
        "critical_analysis": critical_analysis,
        "final_summary": final_summary
    }
    return full_analysis

def parse_json_response(response_text):
    """
    NEW: A robust helper function to find and parse JSON from the model's response.
    Handles cases where the model adds extra text around the JSON.
    """
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        json_pattern = r'```json\s*(\{.*\})\s*```'  
        match = re.search(json_pattern, response_text, re.DOTALL)
        
        if not match:
            match = re.search(r'(\{.*\})', response_text, re.DOTALL)
        
        if match:
            json_str = match.group(1)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"Found potential JSON but couldn't parse it: {e}")
                return {"error": "Failed to parse JSON from response."}
        else:
            print("Could not find any JSON in the response.")
            print(f"Response was: {response_text}")
            return {"error": "No JSON object found in response."}



def advanced_sector_analysis(deck_text, sector):
    """Add industry-specific analysis"""
    sector_prompts = {
        "saas": "Analyze for common SaaS metrics: MRR growth, churn rate, LTV:CAC ratio, magic number",
        "ecommerce": "Analyze for GMV, AOV, customer acquisition cost, retention rates",
        "fintech": "Focus on regulatory compliance, TAM for financial services, unit economics",
        "healthtech": "Analyze FDA approvals, clinical trials, healthcare partnerships"
    }
    
    prompt = f"""
    Perform advanced {sector} sector analysis on this pitch deck:
    
    {deck_text}
    
    {sector_prompts.get(sector, 'Analyze for industry-standard metrics')}
    
    Provide a JSON with: sector_specific_metrics, competitive_landscape, regulatory_risks, growth_potential
    """

def sentiment_analysis(transcript_text):
    """Analyze founder sentiment and confidence"""
    prompt = f"""
    Analyze this founder call transcript for sentiment and communication style:
    
    {transcript_text}
    
    Assess: confidence_level, passion, clarity, objection_handling, potential_red_flags_in_tone
    """

def financial_health_check(extracted_facts):
    """Basic financial sanity checks"""
    checks = []
    if extracted_facts.get('traction'):
        if "1000%" in extracted_facts['traction'] and "month" in extracted_facts['traction']:
            checks.append("Extremely high growth rate claimed - verify authenticity")
    return checks


if __name__ == "__main__":
    print("AI Analyst Core Logic Module")
    print("This module contains the core analysis functions.")
    print("Import these functions in your main application.")