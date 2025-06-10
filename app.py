import base64
import io
import json
import os
import requests
import time
import logging
import asyncio
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from PIL import Image
import uvicorn
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import cv2
from pyzbar import pyzbar
import numpy as np

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('api.log')
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(override=True)
logger.info("[INIT] Loading environment variables...")

# Initialize OpenAI client for nutrition label reading only
try:
    from openai import OpenAI
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.warning("[WARN] OPENAI_API_KEY not found in environment variables")
        openai_client = None
    else:
        openai_client = OpenAI(api_key=openai_api_key)
        logger.info("[INIT] OpenAI client initialized successfully")
except Exception as e:
    logger.error(f"[ERROR] Error initializing OpenAI client: {e}")
    openai_client = None

# Initialize Gemini client for nutrition label reading only
try:
    import google.generativeai as genai
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        logger.warning("[WARN] GEMINI_API_KEY not found in environment variables")
        gemini_client = None
    else:
        genai.configure(api_key=gemini_api_key)
        gemini_client = genai.GenerativeModel('gemini-1.5-flash')
        logger.info("[INIT] Gemini client initialized successfully")
except Exception as e:
    logger.error(f"[ERROR] Error initializing Gemini client: {e}")
    gemini_client = None

# Check if at least one AI client is available for nutrition reading
if not openai_client and not gemini_client:
    logger.error("[ERROR] No AI clients available! Please configure OPENAI_API_KEY or GEMINI_API_KEY")
    client = None
else:
    client = openai_client or gemini_client
    logger.info(f"[INIT] AI clients available: OpenAI={bool(openai_client)}, Gemini={bool(gemini_client)}")

# Thread pool for CPU-intensive tasks
thread_pool = ThreadPoolExecutor(max_workers=50)
logger.info("[INIT] Thread pool initialized with 50 workers for concurrent processing")

app = FastAPI(
    title="Nutrition Facts API", 
    version="4.2.0",
    description="High-performance nutrition analysis API with barcode scanning using free APIs"
)
security = HTTPBearer()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.info("[INIT] CORS middleware configured")

# Request counter for monitoring
request_counter = {"total": 0, "success": 0, "errors": 0}

async def get_available_ai_client(request_id: str):
    """Get available AI client with OpenAI preference and Gemini fallback for nutrition reading only"""
    
    # Try OpenAI first
    if openai_client:
        try:
            # Quick test to see if OpenAI is responsive
            test_response = await asyncio.get_event_loop().run_in_executor(
                thread_pool,
                lambda: openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=1,
                    timeout=5
                )
            )
            logger.info(f"[AI] [{request_id}] OpenAI is available and responsive")
            return "openai", openai_client
        except Exception as e:
            logger.warning(f"[AI] [{request_id}] OpenAI unavailable: {str(e)}, trying Gemini...")
    
    # Fallback to Gemini
    if gemini_client:
        try:
            # Quick test for Gemini
            test_response = await asyncio.get_event_loop().run_in_executor(
                thread_pool,
                lambda: gemini_client.generate_content("test")
            )
            logger.info(f"[AI] [{request_id}] Gemini is available and responsive")
            return "gemini", gemini_client
        except Exception as e:
            logger.error(f"[AI] [{request_id}] Gemini also unavailable: {str(e)}")
    
    logger.error(f"[AI] [{request_id}] No AI clients available")
    return None, None

def log_request_start(endpoint: str, request_id: str):
    """Log request start with unique ID"""
    request_counter["total"] += 1
    logger.info(f"[START] [{request_id}] Starting {endpoint} request (Total: {request_counter['total']})")
    return time.time()

def log_request_end(endpoint: str, request_id: str, start_time: float, success: bool = True):
    """Log request completion with timing"""
    duration = round(time.time() - start_time, 3)
    if success:
        request_counter["success"] += 1
        logger.info(f"[SUCCESS] [{request_id}] {endpoint} completed successfully in {duration}s (Success: {request_counter['success']})")
    else:
        request_counter["errors"] += 1
        logger.error(f"[ERROR] [{request_id}] {endpoint} failed after {duration}s (Errors: {request_counter['errors']})")

def authenticate_user(credentials: HTTPAuthorizationCredentials):
    """Authenticate API requests"""
    api_secret_key = os.getenv("API_SECRET_KEY")
    if not api_secret_key:
        logger.error("[ERROR] API secret key not configured")
        raise HTTPException(status_code=500, detail="API secret key not configured")
    
    if credentials.credentials != api_secret_key:
        logger.warning(f"[WARN] Invalid authentication attempt with token: {credentials.credentials[:10]}...")
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

async def extract_nutrition_with_ai(image_bytes, request_id: str):
    """Extract nutrition data using AI with OpenAI/Gemini fallback"""
    logger.info(f"[AI] [{request_id}] Starting AI nutrition extraction...")
    
    ai_provider, ai_client = await get_available_ai_client(request_id)
    
    if not ai_client:
        logger.error(f"[ERROR] [{request_id}] No AI clients available")
        return {
            "success": False,
            "error": "No AI clients available. Check API keys."
        }
    
    try:
        logger.info(f"[ENCODE] [{request_id}] Encoding image to base64...")
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        logger.info(f"[ENCODE] [{request_id}] Image encoded successfully ({len(base64_image)} chars)")
        
        prompt = """
You are a nutrition label reading expert. Extract the exact values from this nutrition facts label.

CRITICAL INSTRUCTIONS:
1. Look ONLY for "per 100g" or "per 100 grams" values
2. For CALORIES/ENERGY: 
   - ONLY extract the kcal value (kilocalories)
   - IGNORE kJ values (kilojoules) completely
   - Look for numbers followed by "kcal" or "cal"
   - Common format: "2252 kJ / 539 kcal" → use 539
3. Extract these exact values:
   - Energy/Calories in kcal ONLY (ignore kJ)
   - Protein in grams
   - Sugar in grams (usually under carbohydrates section)
   - Total Carbohydrates in grams

4. Return ONLY this JSON format:
{"calories": X, "protein_grams": X, "sugar_grams": X, "carbs_grams": X}

5. EXAMPLES:
   - If you see "2252 kJ / 539 kcal" → use calories: 539
   - If you see "Energy 2252 kJ (539 kcal)" → use calories: 539
   - If you see only "539 kcal" → use calories: 539
   - NEVER use the kJ number for calories

6. If any value is unclear, use 0
7. Return ONLY the JSON, no explanations
"""
        
        logger.info(f"[AI] [{request_id}] Sending request to {ai_provider.upper()}...")
        ai_start = time.time()
        
        if ai_provider == "openai":
            response = await asyncio.get_event_loop().run_in_executor(
                thread_pool,
                lambda: ai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                                }
                            ]
                        }
                    ],
                    max_tokens=150,
                    temperature=0.0
                )
            )
            ai_response = response.choices[0].message.content.strip()
            
        else:  # Gemini
            # Convert base64 to PIL Image for Gemini
            image_data = base64.b64decode(base64_image)
            pil_image = Image.open(io.BytesIO(image_data))
            
            response = await asyncio.get_event_loop().run_in_executor(
                thread_pool,
                lambda: ai_client.generate_content([prompt, pil_image])
            )
            ai_response = response.text.strip()
        
        ai_duration = round(time.time() - ai_start, 3)
        logger.info(f"[AI] [{request_id}] {ai_provider.upper()} response received in {ai_duration}s")
        
        logger.info(f"[PARSE] [{request_id}] Raw AI response: {ai_response[:100]}...")
        
        # Clean JSON response
        logger.info(f"[CLEAN] [{request_id}] Cleaning and parsing AI response...")
        if '```json' in ai_response:
            ai_response = ai_response.split('```json')[1].split('```')[0].strip()
        elif '```' in ai_response:
            ai_response = ai_response.split('```')[1].strip()
        
        start_idx = ai_response.find('{')
        end_idx = ai_response.rfind('}') + 1
        if start_idx != -1 and end_idx != 0:
            ai_response = ai_response[start_idx:end_idx]
        
        logger.info(f"[CLEAN] [{request_id}] Cleaned response: {ai_response}")
        
        nutrition_data = json.loads(ai_response)
        logger.info(f"[SUCCESS] [{request_id}] Successfully parsed nutrition data")
        
        # Validate required fields
        required_fields = ['calories', 'protein_grams', 'sugar_grams', 'carbs_grams']
        validated_data = {}
        
        logger.info(f"[VALIDATE] [{request_id}] Validating nutrition fields...")
        for field in required_fields:
            try:
                validated_data[field] = float(nutrition_data.get(field, 0))
                logger.debug(f"[CHECK] [{request_id}] {field}: {validated_data[field]}")
            except (ValueError, TypeError):
                validated_data[field] = 0.0
                logger.warning(f"[WARN] [{request_id}] Invalid {field}, defaulting to 0")
        
        logger.info(f"[SUCCESS] [{request_id}] AI nutrition extraction completed successfully using {ai_provider.upper()}")
        return {
            "success": True,
            "nutrition_data": validated_data,
            "raw_ai_response": ai_response,
            "processing_time": ai_duration,
            "ai_provider": ai_provider
        }
        
    except json.JSONDecodeError as e:
        logger.error(f"[ERROR] [{request_id}] JSON parsing failed: {str(e)}")
        return {
            "success": False,
            "error": f"Failed to parse AI response as JSON: {str(e)}",
            "raw_ai_response": ai_response if 'ai_response' in locals() else "No response",
            "ai_provider": ai_provider
        }
    except Exception as e:
        logger.error(f"[ERROR] [{request_id}] AI extraction failed: {str(e)}")
        return {
            "success": False,
            "error": f"AI extraction failed: {str(e)}",
            "ai_provider": ai_provider
        }

async def extract_barcode_local(image_bytes, request_id: str):
    """Extract barcode using local libraries (pyzbar + OpenCV)"""
    logger.info(f"[BARCODE] [{request_id}] Starting local barcode extraction...")
    
    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            logger.error(f"[ERROR] [{request_id}] Failed to decode image")
            return {"success": False, "error": "Failed to decode image"}
        
        logger.info(f"[SCAN] [{request_id}] Scanning for barcodes with pyzbar...")
        barcode_start = time.time()
        
        # Try multiple image preprocessing approaches
        approaches = [
            ("original", image),
            ("grayscale", cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)),
            ("enhanced", cv2.convertScaleAbs(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), alpha=1.5, beta=30))
        ]
        
        for approach_name, processed_image in approaches:
            logger.info(f"[SCAN] [{request_id}] Trying {approach_name} approach...")
            barcodes = pyzbar.decode(processed_image)
            
            if barcodes:
                barcode_data = barcodes[0].data.decode('utf-8')
                barcode_duration = round(time.time() - barcode_start, 3)
                logger.info(f"[SUCCESS] [{request_id}] Barcode found with {approach_name}: {barcode_data} in {barcode_duration}s")
                
                return {
                    "success": True,
                    "barcode": barcode_data,
                    "method": f"local_scan_{approach_name}",
                    "processing_time": barcode_duration
                }
        
        barcode_duration = round(time.time() - barcode_start, 3)
        logger.warning(f"[WARN] [{request_id}] No barcode found with local scanning in {barcode_duration}s")
        return {
            "success": False,
            "error": "No barcode detected with local scanning",
            "processing_time": barcode_duration
        }
        
    except Exception as e:
        logger.error(f"[ERROR] [{request_id}] Local barcode extraction failed: {str(e)}")
        return {
            "success": False,
            "error": f"Local barcode extraction failed: {str(e)}"
        }

async def search_product_openfoodfacts(barcode: str, country: str, request_id: str):
    """Search product using Open Food Facts API"""
    logger.info(f"[API1] [{request_id}] Searching Open Food Facts API for barcode {barcode} in {country}...")
    
    try:
        url = f"https://{country}.openfoodfacts.net/api/v2/product/{barcode}"
        params = {
            "fields": "product_name,nutriments,nutrition_grades,brands,countries_tags"
        }
        
        logger.info(f"[API1] [{request_id}] Making API request to: {url}")
        api_start = time.time()
        
        response = await asyncio.get_event_loop().run_in_executor(
            thread_pool,
            lambda: requests.get(url, params=params, timeout=10)
        )
        
        api_duration = round(time.time() - api_start, 3)
        logger.info(f"[API1] [{request_id}] API response received in {api_duration}s (Status: {response.status_code})")
        
        if response.status_code != 200:
            logger.warning(f"[WARN] [{request_id}] Open Food Facts API request failed with status {response.status_code}")
            return {
                "success": False,
                "error": f"API request failed with status {response.status_code}",
                "api_name": "openfoodfacts"
            }
        
        data = response.json()
        logger.info(f"[API1] [{request_id}] API response parsed successfully")
        
        if data.get("status") != 1:
            logger.warning(f"[WARN] [{request_id}] Product not found in Open Food Facts database")
            return {
                "success": False,
                "error": "Product not found in database",
                "api_response": data,
                "api_name": "openfoodfacts"
            }
        
        product = data.get("product", {})
        nutriments = product.get("nutriments", {})
        
        logger.info(f"[PRODUCT] [{request_id}] Product found: {product.get('product_name', 'Unknown')}")
        
        # Extract nutrition data (per 100g)
        nutrition_data = {
            "calories": float(nutriments.get("energy-kcal_100g", 0)),
            "protein_grams": float(nutriments.get("proteins_100g", 0)),
            "sugar_grams": float(nutriments.get("sugars_100g", 0)),
            "carbs_grams": float(nutriments.get("carbohydrates_100g", 0))
        }
        
        logger.info(f"[DATA] [{request_id}] Nutrition data extracted: {nutrition_data}")
        logger.info(f"[SUCCESS] [{request_id}] Open Food Facts search completed successfully")
        
        return {
            "success": True,
            "product_name": product.get("product_name", "Unknown Product"),
            "brands": product.get("brands", "Unknown Brand"),
            "nutrition_data": nutrition_data,
            "nutrition_grade": product.get("nutrition_grades", "unknown"),
            "data_source": "openfoodfacts_api",
            "barcode": barcode,
            "country": country,
            "processing_time": api_duration,
            "api_name": "openfoodfacts"
        }
        
    except requests.RequestException as e:
        logger.error(f"[ERROR] [{request_id}] Network error during Open Food Facts API request: {str(e)}")
        return {
            "success": False,
            "error": f"Network error: {str(e)}",
            "api_name": "openfoodfacts"
        }
    except Exception as e:
        logger.error(f"[ERROR] [{request_id}] Open Food Facts search failed: {str(e)}")
        return {
            "success": False,
            "error": f"Search failed: {str(e)}",
            "api_name": "openfoodfacts"
        }

async def search_product_upc_database(barcode: str, request_id: str):
    """Search product using UPC Database API (free tier)"""
    logger.info(f"[API2] [{request_id}] Searching UPC Database API for barcode {barcode}...")
    
    try:
        url = f"https://api.upcitemdb.com/prod/trial/lookup"
        params = {
            "upc": barcode
        }
        
        logger.info(f"[API2] [{request_id}] Making API request to UPC Database...")
        api_start = time.time()
        
        response = await asyncio.get_event_loop().run_in_executor(
            thread_pool,
            lambda: requests.get(url, params=params, timeout=10)
        )
        
        api_duration = round(time.time() - api_start, 3)
        logger.info(f"[API2] [{request_id}] UPC Database response received in {api_duration}s (Status: {response.status_code})")
        
        if response.status_code != 200:
            logger.warning(f"[WARN] [{request_id}] UPC Database API request failed with status {response.status_code}")
            return {
                "success": False,
                "error": f"API request failed with status {response.status_code}",
                "api_name": "upc_database"
            }
        
        data = response.json()
        logger.info(f"[API2] [{request_id}] UPC Database response parsed successfully")
        
        if not data.get("items") or len(data["items"]) == 0:
            logger.warning(f"[WARN] [{request_id}] Product not found in UPC Database")
            return {
                "success": False,
                "error": "Product not found in UPC Database",
                "api_name": "upc_database"
            }
        
        item = data["items"][0]
        product_name = item.get("title", "Unknown Product")
        brand = item.get("brand", "Unknown Brand")
        
        logger.info(f"[PRODUCT] [{request_id}] Product found in UPC Database: {product_name}")
        
        # UPC Database doesn't provide nutrition data, so we return basic info
        # This will trigger a fallback to other APIs for nutrition data
        return {
            "success": True,
            "product_name": product_name,
            "brands": brand,
            "nutrition_data": None,  # No nutrition data available
            "data_source": "upc_database_api",
            "barcode": barcode,
            "processing_time": api_duration,
            "api_name": "upc_database",
            "note": "Product found but no nutrition data available from this API"
        }
        
    except requests.RequestException as e:
        logger.error(f"[ERROR] [{request_id}] Network error during UPC Database API request: {str(e)}")
        return {
            "success": False,
            "error": f"Network error: {str(e)}",
            "api_name": "upc_database"
        }
    except Exception as e:
        logger.error(f"[ERROR] [{request_id}] UPC Database search failed: {str(e)}")
        return {
            "success": False,
            "error": f"Search failed: {str(e)}",
            "api_name": "upc_database"
        }

async def search_product_barcode_lookup(barcode: str, request_id: str):
    """Search product using Barcode Lookup API (free tier)"""
    logger.info(f"[API3] [{request_id}] Searching Barcode Lookup API for barcode {barcode}...")
    
    try:
        # Note: This API requires an API key for full access, but has a limited free tier
        url = f"https://api.barcodelookup.com/v3/products"
        params = {
            "barcode": barcode,
            "formatted": "y",
            "key": os.getenv("BARCODE_LOOKUP_API_KEY", "")
        }
        
        # If no API key, try the free endpoint (very limited)
        if not params["key"]:
            logger.info(f"[API3] [{request_id}] No API key found, using limited free access")
            url = f"https://world.openfoodfacts.org/api/v0/product/{barcode}.json"
            params = {}
        
        logger.info(f"[API3] [{request_id}] Making API request to Barcode Lookup...")
        api_start = time.time()
        
        response = await asyncio.get_event_loop().run_in_executor(
            thread_pool,
            lambda: requests.get(url, params=params, timeout=10)
        )
        
        api_duration = round(time.time() - api_start, 3)
        logger.info(f"[API3] [{request_id}] Barcode Lookup response received in {api_duration}s (Status: {response.status_code})")
        
        if response.status_code != 200:
            logger.warning(f"[WARN] [{request_id}] Barcode Lookup API request failed with status {response.status_code}")
            return {
                "success": False,
                "error": f"API request failed with status {response.status_code}",
                "api_name": "barcode_lookup"
            }
        
        data = response.json()
        logger.info(f"[API3] [{request_id}] Barcode Lookup response parsed successfully")
        
        # Handle different response formats
        if "products" in data and len(data["products"]) > 0:
            # Standard Barcode Lookup API response
            product = data["products"][0]
            product_name = product.get("product_name", product.get("title", "Unknown Product"))
            brand = product.get("brand", "Unknown Brand")
        elif "product" in data:
            # OpenFoodFacts fallback response
            product = data["product"]
            product_name = product.get("product_name", "Unknown Product")
            brand = product.get("brands", "Unknown Brand")
        else:
            logger.warning(f"[WARN] [{request_id}] Product not found in Barcode Lookup")
            return {
                "success": False,
                "error": "Product not found in Barcode Lookup",
                "api_name": "barcode_lookup"
            }
        
        logger.info(f"[PRODUCT] [{request_id}] Product found in Barcode Lookup: {product_name}")
        
        return {
            "success": True,
            "product_name": product_name,
            "brands": brand,
            "nutrition_data": None,  # Limited nutrition data from this API
            "data_source": "barcode_lookup_api",
            "barcode": barcode,
            "processing_time": api_duration,
            "api_name": "barcode_lookup",
            "note": "Product found but limited nutrition data available from this API"
        }
        
    except requests.RequestException as e:
        logger.error(f"[ERROR] [{request_id}] Network error during Barcode Lookup API request: {str(e)}")
        return {
            "success": False,
            "error": f"Network error: {str(e)}",
            "api_name": "barcode_lookup"
        }
    except Exception as e:
        logger.error(f"[ERROR] [{request_id}] Barcode Lookup search failed: {str(e)}")
        return {
            "success": False,
            "error": f"Search failed: {str(e)}",
            "api_name": "barcode_lookup"
        }

async def search_product_with_multiple_apis(barcode: str, country: str, request_id: str):
    """Search product using multiple free APIs with fallback chain"""
    logger.info(f"[MULTI_API] [{request_id}] Starting multi-API search for barcode {barcode}...")
    
    # API chain: OpenFoodFacts -> UPC Database -> Barcode Lookup
    apis = [
        ("OpenFoodFacts", lambda: search_product_openfoodfacts(barcode, country, request_id)),
        ("UPC Database", lambda: search_product_upc_database(barcode, request_id)),
        ("Barcode Lookup", lambda: search_product_barcode_lookup(barcode, request_id))
    ]
    
    api_results = []
    
    for api_name, api_func in apis:
        logger.info(f"[MULTI_API] [{request_id}] Trying {api_name}...")
        
        try:
            result = await api_func()
            api_results.append({"api": api_name, "result": result})
            
            if result["success"]:
                # Check if we have nutrition data
                if result.get("nutrition_data") and any(result["nutrition_data"].values()):
                    logger.info(f"[SUCCESS] [{request_id}] Found complete nutrition data from {api_name}")
                    result["api_chain_used"] = [r["api"] for r in api_results]
                    return result
                elif result.get("product_name"):
                    logger.info(f"[PARTIAL] [{request_id}] Found product info from {api_name}, but no nutrition data")
                    # Continue to next API for nutrition data
                    continue
            
        except Exception as e:
            logger.error(f"[ERROR] [{request_id}] {api_name} failed: {str(e)}")
            api_results.append({"api": api_name, "result": {"success": False, "error": str(e)}})
    
    # If we reach here, no API provided complete nutrition data
    logger.warning(f"[WARN] [{request_id}] No API provided complete nutrition data")
    
    # Return the best partial result we found
    for api_result in api_results:
        if api_result["result"].get("success") and api_result["result"].get("product_name"):
            result = api_result["result"]
            result["api_chain_used"] = [r["api"] for r in api_results]
            result["warning"] = "Product found but no complete nutrition data available from any API"
            return result
    
    # No APIs found the product
    logger.error(f"[ERROR] [{request_id}] Product not found in any API")
    return {
        "success": False,
        "error": "Product not found in any of the available APIs",
        "api_chain_used": [r["api"] for r in api_results],
        "api_results": api_results
    }

def calculate_nutrition_for_grams(nutrition_per_100g: dict, target_grams: float, request_id: str) -> dict:
    """Calculate nutrition values for specific gram amount with logging"""
    logger.info(f"[CALC] [{request_id}] Calculating nutrition for {target_grams}g...")
    
    try:
        scaling_factor = target_grams / 100.0
        logger.debug(f"[DATA] [{request_id}] Scaling factor: {scaling_factor}")
        
        calculated_nutrition = {
            "calories": round(nutrition_per_100g["calories"] * scaling_factor, 1),
            "protein_grams": round(nutrition_per_100g["protein_grams"] * scaling_factor, 1),
            "sugar_grams": round(nutrition_per_100g["sugar_grams"] * scaling_factor, 1),
            "carbs_grams": round(nutrition_per_100g["carbs_grams"] * scaling_factor, 1)
        }
        
        logger.info(f"[SUCCESS] [{request_id}] Nutrition calculation completed: {calculated_nutrition}")
        
        return {
            "success": True,
            "base_serving_size_grams": 100.0,
            "requested_grams": target_grams,
            "scaling_factor": round(scaling_factor, 4),
            "nutrition_per_100g": nutrition_per_100g,
            "nutrition_for_requested_grams": calculated_nutrition
        }
        
    except Exception as e:
        logger.error(f"[ERROR] [{request_id}] Nutrition calculation failed: {str(e)}")
        return {
            "success": False,
            "error": f"Calculation failed: {str(e)}"
        }

@app.get("/health")
async def health_check():
    """Health check endpoint with system status"""
    logger.info("[HEALTH] Health check requested")
    return {
        "status": "healthy",
        "message": "Nutrition Facts API v4.2",
        "timestamp": datetime.now().isoformat(),
        "openai_status": "connected" if openai_client else "not configured",
        "gemini_status": "connected" if gemini_client else "not configured",
        "ai_clients_available": bool(openai_client or gemini_client),
        "thread_pool_workers": thread_pool._max_workers,
        "request_stats": request_counter,
        "features": [
            "AI nutrition extraction with OpenAI/Gemini fallback", 
            "Multi-API barcode scanning (OpenFoodFacts, UPC Database, Barcode Lookup)", 
            "Local barcode detection with pyzbar",
            "Local calculations", 
            "4 core nutrients",
            "Concurrent processing",
            "Detailed logging",
            "No AI dependency for barcode scanning"
        ],
        "barcode_apis": [
            "OpenFoodFacts (primary)",
            "UPC Database (fallback)",
            "Barcode Lookup (fallback)"
        ]
    }

@app.post("/analyze-nutrition-barcode")
async def analyze_nutrition_barcode(
    image: UploadFile = File(...),
    grams: float = Form(...),
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Analyze nutrition from barcode image using multiple free APIs"""
    request_id = f"barcode_{int(time.time() * 1000)}_{id(image)}"
    start_time = log_request_start("analyze-nutrition-barcode", request_id)
    
    try:
        logger.info(f"[AUTH] [{request_id}] Authenticating user...")
        authenticate_user(credentials)
        
        if grams <= 0:
            logger.warning(f"[WARN] [{request_id}] Invalid grams value: {grams}")
            raise HTTPException(status_code=400, detail="Grams must be positive")
        
        logger.info(f"[FILE] [{request_id}] Reading barcode image ({image.filename})...")
        image_data = await image.read()
        logger.info(f"[FILE] [{request_id}] Barcode image read ({len(image_data)} bytes)")
        
        logger.info(f"[IMAGE] [{request_id}] Processing barcode image with PIL...")
        pil_image = Image.open(io.BytesIO(image_data))
        logger.info(f"[IMAGE] [{request_id}] Image processed: {pil_image.size[0]}x{pil_image.size[1]}")
        
        # Step 1: Extract barcode using local scanning
        logger.info(f"[STEP1] [{request_id}] Step 1: Extracting barcode from image...")
        barcode_result = await extract_barcode_local(image_data, request_id)
        
        if not barcode_result["success"]:
            logger.error(f"[ERROR] [{request_id}] Barcode extraction failed")
            log_request_end("analyze-nutrition-barcode", request_id, start_time, False)
            return {
                "status": "error",
                "message": "Failed to extract barcode from image",
                "error_details": barcode_result,
                "grams": grams,
                "image_size": list(pil_image.size),
                "request_id": request_id
            }
        
        barcode = barcode_result["barcode"]
        extraction_method = barcode_result.get("method", "local_scan")
        logger.info(f"[SUCCESS] [{request_id}] Barcode extracted: {barcode} (method: {extraction_method})")
        
        # Step 2: Search product using multiple APIs (using default world database)
        logger.info(f"[STEP2] [{request_id}] Step 2: Searching product using multiple APIs...")
        api_result = await search_product_with_multiple_apis(barcode, "world", request_id)
        
        if not api_result["success"]:
            logger.error(f"[ERROR] [{request_id}] Product search failed in all APIs")
            log_request_end("analyze-nutrition-barcode", request_id, start_time, False)
            return {
                "status": "error",
                "message": "Product not found in any available API",
                "barcode": barcode,
                "extraction_method": extraction_method,
                "error_details": api_result,
                "grams": grams,
                "request_id": request_id
            }
        
        # Check if we have nutrition data
        if not api_result.get("nutrition_data") or not any(api_result["nutrition_data"].values()):
            logger.warning(f"[WARN] [{request_id}] Product found but no nutrition data available")
            log_request_end("analyze-nutrition-barcode", request_id, start_time, False)
            return {
                "status": "partial_success",
                "message": "Product found but no nutrition data available",
                "barcode": barcode,
                "product_name": api_result.get("product_name", "Unknown"),
                "brands": api_result.get("brands", "Unknown"),
                "extraction_method": extraction_method,
                "data_source": api_result.get("data_source", "unknown"),
                "api_chain_used": api_result.get("api_chain_used", []),
                "grams": grams,
                "request_id": request_id
            }
        
        # Step 3: Calculate nutrition for requested grams
        logger.info(f"[STEP3] [{request_id}] Step 3: Calculating nutrition for {grams}g...")
        calculation_result = calculate_nutrition_for_grams(
            api_result["nutrition_data"], grams, request_id
        )
        
        if not calculation_result["success"]:
            logger.error(f"[ERROR] [{request_id}] Nutrition calculation failed")
            log_request_end("analyze-nutrition-barcode", request_id, start_time, False)
            return {
                "status": "error",
                "message": "Failed to calculate nutrition",
                "error_details": calculation_result,
                "grams": grams,
                "request_id": request_id
            }
        
        log_request_end("analyze-nutrition-barcode", request_id, start_time, True)
        
        return {
            "status": "success",
            "message": "Nutrition analysis completed via API",
            "barcode": barcode,
            "product_name": api_result["product_name"],
            "brands": api_result.get("brands", "Unknown"),
            "nutrition_grade": api_result.get("nutrition_grade", "unknown"),
            "grams": grams,
            "image_size": list(pil_image.size),
            "data_source": api_result.get("data_source", "unknown"),
            "extraction_method": extraction_method,
            "analysis_method": f"{extraction_method} + multi_api + local_calculation",
            "api_chain_used": api_result.get("api_chain_used", []),
            "nutrition_analysis": calculation_result,
            "processing_times": {
                "barcode_extraction": barcode_result.get("processing_time", 0),
                "api_search": api_result.get("processing_time", 0)
            },
            "request_id": request_id
        }
        
    except Exception as e:
        logger.error(f"[ERROR] [{request_id}] Unexpected error: {str(e)}")
        log_request_end("analyze-nutrition-barcode", request_id, start_time, False)
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/analyze-nutrition-raw")
async def analyze_nutrition_raw(
    request: Request,
    grams: float,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Analyze nutrition from raw image data with detailed progress logging"""
    request_id = f"raw_{int(time.time() * 1000)}_{id(request)}"
    start_time = log_request_start("analyze-nutrition-raw", request_id)
    
    try:
        logger.info(f"[AUTH] [{request_id}] Authenticating user...")
        authenticate_user(credentials)
        
        if grams <= 0:
            logger.warning(f"[WARN] [{request_id}] Invalid grams value: {grams}")
            raise HTTPException(status_code=400, detail="Grams must be positive")
        
        logger.info(f"[FILE] [{request_id}] Reading raw image data...")
        image_data = await request.body()
        logger.info(f"[FILE] [{request_id}] Raw image data read ({len(image_data)} bytes)")
        
        logger.info(f"[IMAGE] [{request_id}] Processing raw image with PIL...")
        pil_image = Image.open(io.BytesIO(image_data))
        logger.info(f"[IMAGE] [{request_id}] Image processed: {pil_image.size[0]}x{pil_image.size[1]}")
        
        ai_result = await extract_nutrition_with_ai(image_data, request_id)
        
        if not ai_result["success"]:
            logger.error(f"[ERROR] [{request_id}] AI extraction failed")
            log_request_end("analyze-nutrition-raw", request_id, start_time, False)
            return {
                "status": "error",
                "message": "Failed to extract nutrition data",
                "error_details": ai_result,
                "grams": grams,
                "image_size": list(pil_image.size),
                "request_id": request_id
            }
        
        calculation_result = calculate_nutrition_for_grams(
            ai_result["nutrition_data"], grams, request_id
        )
        
        if not calculation_result["success"]:
            logger.error(f"[ERROR] [{request_id}] Nutrition calculation failed")
            log_request_end("analyze-nutrition-raw", request_id, start_time, False)
            return {
                "status": "error",
                "message": "Failed to calculate nutrition",
                "error_details": calculation_result,
                "grams": grams,
                "image_size": list(pil_image.size),
                "request_id": request_id
            }
        
        log_request_end("analyze-nutrition-raw", request_id, start_time, True)
        
        return {
            "status": "success",
            "message": "Nutrition analysis completed",
            "grams": grams,
            "image_size": list(pil_image.size),
            "analysis_method": "ai_vision + local_calculation",
            "nutrition_analysis": calculation_result,
            "ai_response": ai_result.get("raw_ai_response", "No AI response"),
            "processing_time": ai_result.get("processing_time", 0),
            "request_id": request_id
        }
        
    except Exception as e:
        logger.error(f"[ERROR] [{request_id}] Unexpected error: {str(e)}")
        log_request_end("analyze-nutrition-raw", request_id, start_time, False)
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/analyze-nutrition-barcode-manual")
async def analyze_nutrition_barcode_manual(
    barcode: str = Form(...),
    grams: float = Form(...),
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Analyze nutrition from manually entered barcode using multiple free APIs"""
    request_id = f"manual_{int(time.time() * 1000)}_{hash(barcode)}"
    start_time = log_request_start("analyze-nutrition-barcode-manual", request_id)
    
    try:
        logger.info(f"[AUTH] [{request_id}] Authenticating user...")
        authenticate_user(credentials)
        
        if grams <= 0:
            logger.warning(f"[WARN] [{request_id}] Invalid grams value: {grams}")
            raise HTTPException(status_code=400, detail="Grams must be positive")
        
        if not barcode or len(barcode.strip()) == 0:
            logger.warning(f"[WARN] [{request_id}] Empty barcode provided")
            raise HTTPException(status_code=400, detail="Barcode cannot be empty")
        
        barcode = barcode.strip()
        logger.info(f"[INPUT] [{request_id}] Manual barcode input: {barcode}")
        
        # Search product using multiple APIs (using default world database)
        logger.info(f"[STEP1] [{request_id}] Searching product using multiple APIs...")
        api_result = await search_product_with_multiple_apis(barcode, "world", request_id)
        
        if not api_result["success"]:
            logger.error(f"[ERROR] [{request_id}] Product search failed in all APIs")
            log_request_end("analyze-nutrition-barcode-manual", request_id, start_time, False)
            return {
                "status": "error",
                "message": "Product not found in any available API",
                "barcode": barcode,
                "error_details": api_result,
                "grams": grams,
                "request_id": request_id
            }
        
        # Check if we have nutrition data
        if not api_result.get("nutrition_data") or not any(api_result["nutrition_data"].values()):
            logger.warning(f"[WARN] [{request_id}] Product found but no nutrition data available")
            log_request_end("analyze-nutrition-barcode-manual", request_id, start_time, False)
            return {
                "status": "partial_success",
                "message": "Product found but no nutrition data available",
                "barcode": barcode,
                "product_name": api_result.get("product_name", "Unknown"),
                "brands": api_result.get("brands", "Unknown"),
                "data_source": api_result.get("data_source", "unknown"),
                "api_chain_used": api_result.get("api_chain_used", []),
                "grams": grams,
                "request_id": request_id
            }
        
        # Calculate nutrition for requested grams
        logger.info(f"[STEP2] [{request_id}] Calculating nutrition for {grams}g...")
        calculation_result = calculate_nutrition_for_grams(
            api_result["nutrition_data"], grams, request_id
        )
        
        if not calculation_result["success"]:
            logger.error(f"[ERROR] [{request_id}] Nutrition calculation failed")
            log_request_end("analyze-nutrition-barcode-manual", request_id, start_time, False)
            return {
                "status": "error",
                "message": "Failed to calculate nutrition",
                "error_details": calculation_result,
                "grams": grams,
                "request_id": request_id
            }
        
        log_request_end("analyze-nutrition-barcode-manual", request_id, start_time, True)
        
        return {
            "status": "success",
            "message": "Nutrition analysis completed via manual barcode entry",
            "barcode": barcode,
            "product_name": api_result["product_name"],
            "brands": api_result.get("brands", "Unknown"),
            "nutrition_grade": api_result.get("nutrition_grade", "unknown"),
            "grams": grams,
            "data_source": api_result.get("data_source", "unknown"),
            "analysis_method": "manual_barcode + multi_api + local_calculation",
            "api_chain_used": api_result.get("api_chain_used", []),
            "nutrition_analysis": calculation_result,
            "processing_time": api_result.get("processing_time", 0),
            "request_id": request_id
        }
        
    except Exception as e:
        logger.error(f"[ERROR] [{request_id}] Unexpected error: {str(e)}")
        log_request_end("analyze-nutrition-barcode-manual", request_id, start_time, False)
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"[START] Starting Nutrition Facts API v4.2 on port {port}")
    logger.info(f"[INIT] Configured for high-concurrency with {thread_pool._max_workers} worker threads")
    logger.info("[INIT] Using multiple free APIs for barcode scanning: OpenFoodFacts, UPC Database, Barcode Lookup")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        workers=1,
        loop="asyncio",
        access_log=True
    )