import os
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any
import logging
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from google.genai import types 
import re
import json
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

from sentence_transformers import SentenceTransformer

from google import genai

# Load environment variables
load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Qdrant in-memory setup
quadclient = QdrantClient(":memory:")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# --- CONFIG ---
API_KEY = os.getenv("GEMINI_API_KEY")  # set this in your deployment environment
if not API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY environment variable")

MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB - tune per your needs
MODEL_NAME = "gemini-2.5-flash"      # change if you have another model

# --- CLIENT SETUP ---
client = genai.Client(api_key=API_KEY)

# --- FASTAPI SETUP ---
app = FastAPI(title="CrowdDetector API", version="1.0")
logger = logging.getLogger("uvicorn.error")


class TextInput(BaseModel):
    text: str

class CrowdResult(BaseModel):
    people_count: Optional[int]
    crowd_score: Optional[int]            # 1-10
    crowd_label: Optional[str]            # e.g., "Low", "Medium", "High"
    confidence: Optional[float]           # 0-100
    rationale: Optional[str]
    # Departure board detection fields
    screen_detected: Optional[bool]       # Whether a screen/monitor/board was detected
    departure_type: Optional[str]         # e.g., "flight", "train", "bus", "none"
    departure_info: Optional[List[Dict[str, Any]]]  # List of departure entries with details


def extract_first_json(text: str):
    """
    Try to find a JSON object inside the response text.
    Returns Python object or raises ValueError.
    """
    # Greedy-ish regex to capture the first balanced-looking JSON object
    m = re.search(r"(\{[\s\S]*\})", text)
    if not m:
        raise ValueError("No JSON object found in model response")
    candidate = m.group(1)

    # Try progressive trimming in case of trailing commas or minor issues
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # attempt to strip trailing commas that often break JSON
        cleaned = re.sub(r",\s*}", "}", candidate)
        cleaned = re.sub(r",\s*]", "]", cleaned)
        return json.loads(cleaned)


SYSTEM_PROMPT = """
You are a safety-first multimodal vision assistant. Analyze the provided image and return ONLY a JSON object (no surrounding explanation or markdown).

Required fields:
- people_count: integer (estimated number of people visible in the image)
- crowd_score: integer 1-10 (1 = empty, 10 = extremely crowded)
- crowd_label: string ("Low", "Medium", or "High")
- confidence: float 0-100 (how confident you are about the count & score)
- rationale: short string (1-2 sentences) explaining how you derived the result

Additional fields for departure board detection:
- screen_detected: boolean (true if any screen, monitor, display board, or information board is visible in the image)
- departure_type: string (one of: "flight", "train", "bus", "subway", "ferry", or "none" if no departure board detected)
- departure_info: array of objects (only if screen_detected is true). Each object should contain:
* flight_number or train_number or route_number: string (the identifier)
* destination: string (destination city/station name)
* departure_time: string (scheduled departure time if visible)
* status: string (e.g., "On Time", "Delayed", "Boarding", "Gate", "Platform", etc.)
* gate or platform: string (if visible on the board)

If multiple departures are visible, include all of them in the array. If only partial information is visible, include what you can read.

If uncertain, set confidence lower and approximate the people_count as best as possible.
If no screen/board is detected, set screen_detected to false, departure_type to "none", and departure_info to an empty array.

Return the JSON object and nothing else.
"""

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze-image", response_model=CrowdResult)
async def analyze_image(file: UploadFile = File(...), user_id: Optional[str] = None):
    """
    Accepts multipart/form-data with a single image file.
    Returns a structured JSON with crowd analysis and departure board information (if detected).
    Fields include: people_count, crowd_score, crowd_label, confidence, rationale,
    screen_detected, departure_type, and departure_info.
    """
    # 1) Basic validations
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty file")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            detail=f"File too large. Max {MAX_UPLOAD_BYTES} bytes allowed.")

    # 2) Build image PART for Gemini
    try:
        image_part = types.Part.from_bytes(data=contents, mime_type=file.content_type or "image/jpeg")
    except Exception as e:
        logger.exception("Failed to create image part")
        raise HTTPException(status_code=500, detail="Failed to prepare image for analysis")

    # 3) Prompt + call Gemini (synchronous)
    prompt = SYSTEM_PROMPT.strip()
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[prompt, image_part],
            # optionally set other params like temperature, max_output_tokens if SDK supports them:
            # temperature=0.0, max_output_tokens=400
        )
        raw_text = response.text or ""
    except Exception as e:
        logger.exception("Gemini API call failed")
        raise HTTPException(status_code=502, detail="Vision model request failed")

    # 4) Extract JSON from response text robustly
    try:
        parsed = extract_first_json(raw_text)
    except Exception as e:
        logger.exception("Failed to parse JSON from model response", exc_info=e)
        # As a fallback, return a structured error payload that the frontend can handle
        raise HTTPException(status_code=502, detail="Model returned unexpected output format")

    # 5) Sanitize/normalize the parsed data into expected fields
    try:
        people_count = parsed.get("people_count")
        # try to coerce numeric types
        if people_count is not None:
            people_count = int(people_count)

        crowd_score = parsed.get("crowd_score")
        if crowd_score is not None:
            crowd_score = int(crowd_score)
            crowd_score = max(1, min(10, crowd_score))

        crowd_label = parsed.get("crowd_label")
        confidence = parsed.get("confidence")
        if confidence is not None:
            confidence = float(confidence)

        rationale = parsed.get("rationale", "")

        # Departure board fields
        screen_detected = parsed.get("screen_detected")
        if screen_detected is not None:
            screen_detected = bool(screen_detected)
        
        departure_type = parsed.get("departure_type")
        if departure_type and isinstance(departure_type, str):
            departure_type = departure_type.lower()
            valid_types = ["flight", "train", "bus", "subway", "ferry", "none"]
            if departure_type not in valid_types:
                departure_type = "none"
        else:
            departure_type = "none" if not screen_detected else None

        departure_info = parsed.get("departure_info")
        if departure_info is None:
            departure_info = []
        elif not isinstance(departure_info, list):
            departure_info = []
        else:
            # Ensure each entry is a dict
            departure_info = [entry for entry in departure_info if isinstance(entry, dict)]

        result = CrowdResult(
            people_count=people_count,
            crowd_score=crowd_score,
            crowd_label=crowd_label,
            confidence=confidence,
            rationale=rationale,
            screen_detected=screen_detected,
            departure_type=departure_type,
            departure_info=departure_info
        )
    except Exception as e:
        logger.exception("Failed to normalize model JSON")
        raise HTTPException(status_code=500, detail="Failed to normalize model response")

    return JSONResponse(status_code=200, content=result.dict())


@app.post("/askAi/")
async def analyze_image(
    image: UploadFile = File(...),
    user_prompt: str = Form(...),
    userPreferences: str = Form(...)
    
):
    try:
        image_bytes = await image.read()
        print("Received user_prompt:", user_prompt)
        print("Received userPreferences:", userPreferences)
        print("Received image filename:", image.filename, "size:", len(image_bytes))
        image.file.seek(0)

        base_prompt = """
        You are a visual assistant for blind navigation.
        - Only answer using information visible in the image.
        - If a user asks for information that is not visible (e.g., bus number or gate),
        reply with: "Not visible. Please turn the camera slightly left or right and ask again."
        - For crowd density questions tell if how crowded the place is from the picture.
        - Be concise.
        """

        final_prompt = f"{base_prompt}, User Preferences: {userPreferences}, User Prompt: {user_prompt}, Please answer according to the user's preferences."

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=image.content_type or "image/jpeg",
                ),
                final_prompt
            ]
        )
        
        return {"response": response.text}
    
    except Exception as e:
        print("Error processing request:", e)
        raise HTTPException(status_code=400, detail=str(e))



COLLECTION = "transit_memory_db"
if not quadclient.collection_exists(COLLECTION):
    quadclient.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

# --- DATA MODELS ---
class Preference(BaseModel):
    userId: str
    key: str
    value: str

class UserQuery(BaseModel): # 👈 NEW: Model for logging voice queries
    userId: str
    query: str
# --------------------

# --- PREFERENCE ENDPOINTS ---

@app.get("/store-preference")
async def store_get():
    return {"error": "Use POST to save data"}

@app.post("/store-preference")
async def store(pref: Preference):
    text = f"User's {pref.key}: {pref.value}"
    vector = embedder.encode(text).tolist()
    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{pref.userId}_{pref.key}"))
    
    quadclient.upsert(
        collection_name=COLLECTION,
        points=[PointStruct(
            id=point_id,
            vector=vector,
            payload={
                "type": "USER_PREF",
                "user_id": pref.userId,
                pref.key: pref.value
            }
        )]
    )
    return {"status": "preference saved"}

@app.get("/get-prefs/{user_id}")
async def get_prefs(user_id: str):
    # Your semantic query
    query = "user preferences crowd seating anxiety wait time"
    
    # Encode the query to a vector (assuming embedder is already defined)
    vector = embedder.encode(query).tolist()

    # Search in Qdrant using query_points
    response = quadclient.query_points(
        collection_name=COLLECTION,
        query=vector,
        limit=10,
        query_filter=Filter(
            must=[
                FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                FieldCondition(key="type", match=MatchValue(value="USER_PREF"))
            ]
        ),
        with_payload=True
    )

    # Extract preferences from the payload
    prefs = {}
    for hit in response.points:
        payload = hit.payload
        for k, v in payload.items():
            if k not in ["type", "user_id"]:
                prefs[k] = v

    return prefs

# --- QUERY LOGGING ENDPOINT (NEW) ---

@app.post("/store-query")
async def store_query_endpoint(uq: UserQuery):
    """Stores the user's raw voice query into Qdrant for history/context."""
    text = f"User asked: {uq.query}"
    vector = embedder.encode(text).tolist()
    # Use a time-based UUID for unique IDs for each query
    point_id = str(uuid.uuid1())
    
    quadclient.upsert(
        collection_name=COLLECTION,
        points=[PointStruct(
            id=point_id,
            vector=vector,
            payload={
                "type": "USER_QUERY",
                "user_id": uq.userId,
                "query_text": uq.query,
                "timestamp": str(datetime.now())
            }
        )]
    )
    return {"status": "query logged"}

# --- DATABASE INSPECTION ENDPOINT (NEW) ---

@app.get("/inspect-db")
async def inspect_db():
    """Fetches all points from the collection for debugging."""
    
    all_points = []
    # Use scroll to efficiently iterate over all points
    scroll_result = quadclient.scroll(
        collection_name=COLLECTION,
        limit=50, 
        with_payload=True,
        with_vectors=False
    )
    records = scroll_result[0]
    next_offset = scroll_result[1]

    # Function to process records from one scroll
    def process_records(recs):
        for record in recs:
            all_points.append({
                "id": record.id,
                "payload": record.payload
            })

    process_records(records)

    # Loop through remaining pages
    while next_offset is not None:
        scroll_result = quadclient.scroll(
            collection_name=COLLECTION,
            limit=50,
            with_payload=True,
            with_vectors=False,
            offset=next_offset
        )
        records = scroll_result[0]
        next_offset = scroll_result[1]
        process_records(records)

    return {"collection": COLLECTION, "total_records": len(all_points), "records": all_points}


# --- MAIN PIPELINE ENDPOINTS ---

@app.post("/process-query")
async def process_query(data: dict):
    query = data.get("query", "").lower()
    needs_vision = any(word in query for word in ["bus", "platform", "crowd", "seat", "smell"])
    return {"needsVision": needs_vision}

@app.post("/fusion")
async def fusion(data: dict):
    user_id = data.get("userId")
    query = data.get("query", "")
    image_data = data.get("image", "")

    # 🚨 STEP 1: LOG THE USER QUERY
    # Store the query text as part of the user's history
    await store_query_endpoint(UserQuery(userId=user_id, query=query))

    # --- 🔍 CHECK DATA ARRIVAL HERE ---
    print("\n--- INCOMING FUSION DATA ---")
    print(f"User ID: {user_id}")
    print(f"Query: {query}")
    if image_data:
        print(f"Image Data Size: {len(image_data)} characters")
        print(f"Image Data Starts With: {image_data[:30]}...") 
    else:
        print("Image Data: MISSING")
    print("--------------------------\n")
    # -----------------------------------

    # STEP 2: GET PREFERENCES
    prefs = await get_prefs(user_id)
    
    # STEP 3: MOCK VISION AND FUSION LOGIC (Replace with Gemini/LLM calls later)
    vision_result = "Bus 101 is here. Crowd: medium. Seat: front available."

    instruction = f"{vision_result} "
    if prefs.get("crowdTolerance") == "low": # Assuming "low" is the stored value for < 5 tolerance
        instruction += "Avoid front due to crowd anxiety. "
    if prefs.get("seatingPreference") == "rear":
        instruction += "Go to rear for safety. "

    return {"instruction": instruction.strip()}

@app.get("/get-user-summary/{user_id}")
async def get_user_summary(user_id: str):
    """
    Fetches the user's current preferences and their last few queries.
    """
    # 1. Fetch current preferences (using existing function)
    preferences = await get_prefs(user_id)
    
    # 2. Fetch last 5 queries
    queries = []
    
    # We use scroll with filtering (filter = payload.type == "USER_QUERY")
    scroll_result = quadclient.scroll(
        collection_name=COLLECTION,
        limit=5, # Limit to the last 5 queries
        with_payload=True,
        with_vectors=False,
        # Filter: Only points where 'type' is 'USER_QUERY' and 'user_id' matches
        # Note: Qdrant filtering uses JSON-like structure
        scroll_filter={
            "must": [
                {"key": "type", "match": {"value": "USER_QUERY"}},
                {"key": "user_id", "match": {"value": user_id}},
            ]
        }
    )
    
    # The scroll result is sorted by creation time (uuid1), so the first 5 are the latest
    records = scroll_result[0]
    
    for record in records:
        # Extract the query text and timestamp
        queries.append({
            "query": record.payload.get("query_text"),
            "time": record.payload.get("timestamp").split('.')[0] # Clean up the timestamp
        })

    return {
        "userId": user_id,
        "preferences": preferences,
        "recentQueries": queries
    }



