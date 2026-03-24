import os
import pandas as pd
import re
import json
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Enable CORS so your frontend can talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # This allows ANY website to talk to your backend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenRouter Configuration
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

def anonymize_medical_data(df):
    """
    STORY FOR GRADUATION: This function scans for PII (Personally Identifiable Information)
    and drops those columns to protect patient privacy before AI processing.
    """
    pii_cols = ['name', 'email', 'phone', 'ssn', 'address', 'patient_id']
    to_drop = [c for c in df.columns if any(p in c.lower() for p in pii_cols)]
    return df.drop(columns=to_drop)

@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "MedVizion AI Clinical Engine is running",
        "version": "1.0.0",
        "author": "AI intellectuals"
    }
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    # 1. Load the CSV
    df = pd.read_csv(file.file)
    
    # 2. Anonymize
    df_clean = anonymize_medical_data(df)
    
    # 3. Create a summary for the AI
    stats = df_clean.describe().to_json()
    
    # 4. Prompt OpenRouter
    prompt = f"Analyze these medical stats: {stats}. Identify 1 key trend and suggest 2 visual charts. Return as JSON: {{'trend': '...', 'summary': '...'}}"
    
    response = client.chat.completions.create(
        model="google/gemini-2.0-flash-001",
        messages=[{"role": "user", "content": prompt}],
        response_format={ "type": "json_object" }
    )
    
    return {
        "analysis": json.loads(response.choices[0].message.content),
        "preview": df_clean.head(5).to_dict(orient='records')
  }
