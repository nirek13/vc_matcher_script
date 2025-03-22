from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
import re
import os

from sentence_transformers import SentenceTransformer, util
from concurrent.futures import ProcessPoolExecutor

app = FastAPI()

# CORS fix
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # for local dev
        "https://gen-ai-genissis.vercel.app/",  # your actual Vercel frontend
        "https://summaforfounders.com/"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


df = None
model = None

def load_data_and_model():
    global df, model
    if df is None:
        print("📊 Loading CSV data...")
        df = pd.read_csv("investors.csv")
        df = df.dropna(subset=[col for col in df.columns if col != "Investment Thesis"])
        df["Investment Thesis"] = df["Investment Thesis"].fillna("")
    if model is None:
        print("🤖 Loading SentenceTransformer model...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Done loading models...")

load_data_and_model()

def safe_split(col):
    return [part.strip().lower() for part in str(col).replace("+", ",").replace("/", ",").split(",")]

def parse_check_size(size_str):
    if not isinstance(size_str, str):
        return (0, 0)
    size_str = size_str.lower().replace("$", "").replace("usd", "")
    parts = re.findall(r'(\d+(?:\.\d+)?)([kmb])', size_str)
    multipliers = {"k": 1, "m": 1000, "b": 1_000_000}
    values = [float(num) * multipliers[suffix] for num, suffix in parts]
    if len(values) == 1:
        return (values[0], values[0])
    elif len(values) == 2:
        return (min(values), max(values))
    return (0, 0)

def check_overlap(user_range, vc_range):
    return user_range[0] <= vc_range[1] and vc_range[0] <= user_range[1]

def compute_structured_score(vc, user_startup):
    score = 0
    industry_match = user_startup["industry"].lower() in str(vc.get("Investment Thesis", "")).lower()
    stage_focus = safe_split(vc.get("Stages", ""))
    geo_focus = safe_split(vc.get("Geography", ""))

    if industry_match:
        score += 0.3
    if user_startup["stage"].lower() in stage_focus:
        score += 0.3
    if any(loc in user_startup["location"].lower() for loc in geo_focus) or "global" in geo_focus:
        score += 0.2
    if user_startup["business_model"].lower() in str(vc.get("Investment Thesis", "")).lower():
        score += 0.1
    if user_startup["team"].lower().startswith("2 technical"):
        score += 0.1

    user_range = parse_check_size(user_startup.get("preferred_check_size", ""))
    vc_range = parse_check_size(vc.get("Check Size", ""))
    if check_overlap(user_range, vc_range):
        score += 0.1

    return score

def compute_score_wrapper(row_user_tuple):
    row, user = row_user_tuple
    return compute_structured_score(row, user)

class Startup(BaseModel):
    industry: str
    stage: str
    business_model: str
    location: str
    traction: str
    team: str
    pitch: str
    preferred_check_size: str

@app.post("/rank-vcs")
def rank_vcs(startup: Startup):
    print("📥 Received a request to /rank-vcs")
    user = startup.dict()

    user_vector = model.encode(user["pitch"], convert_to_tensor=True)
    thesis_vectors = model.encode(df["Investment Thesis"].tolist(), convert_to_tensor=True, batch_size=64)
    nlp_sims = [util.cos_sim(user_vector, vc_vector).item() for vc_vector in thesis_vectors]

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        structured_scores = list(
            executor.map(
                compute_score_wrapper,
                [(row, user) for row in df.to_dict("records")]
            )
        )

    final_scores = 0.6 * np.array(structured_scores) + 0.4 * np.array(nlp_sims)
    min_score, max_score = final_scores.min(), final_scores.max()
    compat_scores = (final_scores - min_score) / (max_score - min_score)

    result_df = df.copy()
    result_df["structured_score"] = structured_scores
    result_df["nlp_similarity"] = nlp_sims
    result_df["final_score"] = final_scores
    result_df["compatibility_score"] = compat_scores

    top_matches = result_df.sort_values("compatibility_score", ascending=False).head(250)

    return top_matches[[
        "Investor Name", "Investment Thesis", "Check Size", "Geography", "Stages", "Fake Email",
        "structured_score", "nlp_similarity", "final_score", "compatibility_score"
    ]].to_dict(orient="records")

@app.get("/healthcheck")
def healthcheck() -> str:
    return "ok"