import base64
import io
import json
import os
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from PIL import Image
import uvicorn
from dotenv import load_dotenv

# Load environment variables first
load_dotenv(override=True)

# Import OpenAI after loading env vars
try:
    from openai import OpenAI
    # Initialize OpenAI client with error handling
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Warning: OPENAI_API_KEY not found in environment variables")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Environment variables loaded: {bool(os.getenv('API_SECRET_KEY'))}")
        client = None
    else:
        print(f"OpenAI API key loaded: {openai_api_key[:10]}...")
        client = OpenAI(api_key=openai_api_key)
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    client = None

app = FastAPI(title="Nutrition Facts API", version="3.0.0")
security = HTTPBearer()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def authenticate_user(credentials: HTTPAuthorizationCredentials):
    """Authenticate API requests"""
    api_secret_key = os.getenv("API_SECRET_KEY")
    print(f"Expected API key: {api_secret_key}")
    print(f"Received API key: {credentials.credentials}")
    
    if not api_secret_key:
        raise HTTPException(status_code=500, detail="API secret key not configured")
    
    if credentials.credentials != api_secret_key:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

def extract_nutrition_with_ai(image_bytes: bytes) -> dict:
    """Extract nutrition data using GPT-4o-mini with highly controlled prompt"""
    if client is None:
        return {
            "success": False,
            "error": "OpenAI client not initialized. Check API key."
        }
    
    try:
        # Convert image to base64
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        # Improved prompt for accurate nutrition label reading
        controlled_prompt = """
You are a nutrition label reading expert. Look at this nutrition facts label image and extract the exact values shown.

CRITICAL INSTRUCTIONS:
1. Look for the "per 100g" or "per 100 grams" column on the nutrition label
2. If you see multiple columns (like "per serving" and "per 100g"), ALWAYS use the "per 100g" values
3. Read the EXACT numbers shown on the label - do not estimate or round
4. Look for these specific terms:
   - Energy/Energia/Energie in kcal (NOT kJ) 
   - Protein/Proteine/Eiweiß
   - Sugar/Sucre/Zucker (under carbohydrates)
   - Carbohydrates/Glucides/Kohlenhydrate

5. Return ONLY a JSON object with these exact keys:
   - "calories": the kcal value per 100g (NOT kJ)
   - "protein_grams": protein in grams per 100g
   - "sugar_grams": sugar in grams per 100g  
   - "carbs_grams": total carbohydrates in grams per 100g

6. If any value is unclear, use 0
7. Do NOT include any explanations, just the JSON

Example of what I expect:
{"calories": 539, "protein_grams": 6.3, "sugar_grams": 56.3, "carbs_grams": 57.5}

Now read the nutrition label and return the JSON with the EXACT values shown per 100g:"""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": controlled_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=150,
            temperature=0.0,  # Zero temperature for consistent results
            top_p=0.1  # Very focused responses
        )
        
        # Get AI response
        ai_response = response.choices[0].message.content.strip()
        
        # Clean response to extract pure JSON
        if '```json' in ai_response:
            ai_response = ai_response.split('```json')[1].split('```')[0].strip()
        elif '```' in ai_response:
            ai_response = ai_response.split('```')[1].strip()
        
        # Remove any extra text before/after JSON
        start_idx = ai_response.find('{')
        end_idx = ai_response.rfind('}') + 1
        if start_idx != -1 and end_idx != 0:
            ai_response = ai_response[start_idx:end_idx]
        
        # Parse JSON
        nutrition_data = json.loads(ai_response)
        
        # Validate and ensure all required fields exist
        required_fields = ['calories', 'protein_grams', 'sugar_grams', 'carbs_grams']
        validated_data = {}
        
        for field in required_fields:
            if field in nutrition_data:
                try:
                    validated_data[field] = float(nutrition_data[field])
                except (ValueError, TypeError):
                    validated_data[field] = 0.0
            else:
                validated_data[field] = 0.0
        
        return {
            "success": True,
            "nutrition_per_100g": validated_data,
            "raw_ai_response": ai_response
        }
        
    except json.JSONDecodeError as e:
        return {
            "success": False,
            "error": f"Invalid JSON from AI: {str(e)}",
            "raw_ai_response": ai_response if 'ai_response' in locals() else "No response"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"AI extraction failed: {str(e)}"
        }

def calculate_nutrition_for_grams(nutrition_per_100g: dict, target_grams: float) -> dict:
    """Calculate nutrition values for specific gram amount using local math"""
    try:
        # Calculate scaling factor
        scaling_factor = target_grams / 100.0
        
        # Calculate nutrition for target grams
        calculated_nutrition = {
            "calories": round(nutrition_per_100g["calories"] * scaling_factor, 1),
            "protein_grams": round(nutrition_per_100g["protein_grams"] * scaling_factor, 1),
            "sugar_grams": round(nutrition_per_100g["sugar_grams"] * scaling_factor, 1),
            "carbs_grams": round(nutrition_per_100g["carbs_grams"] * scaling_factor, 1)
        }
        
        return {
            "success": True,
            "base_serving_size_grams": 100.0,
            "requested_grams": target_grams,
            "scaling_factor": round(scaling_factor, 4),
            "nutrition_per_100g": nutrition_per_100g,
            "nutrition_for_requested_grams": calculated_nutrition
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Calculation failed: {str(e)}"
        }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    openai_status = "connected" if client else "not configured"
    return {
        "status": "healthy", 
        "message": "Nutrition Facts API v3.0 - AI Only",
        "openai_status": openai_status,
        "features": ["AI nutrition extraction", "Local calculations", "4 core nutrients"],
        "analysis_method": "OpenAI GPT-4o-mini only"
    }

@app.post("/analyze-nutrition")
async def analyze_nutrition(
    image: UploadFile = File(...), 
    grams: float = Form(...), 
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Analyze nutrition from uploaded image using AI only"""
    try:
        # Authenticate
        authenticate_user(credentials)
        
        # Validate grams input
        if grams <= 0:
            raise HTTPException(status_code=400, detail="Grams must be a positive number")
        
        # Process image
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data))
        
        # Extract nutrition using AI only
        ai_result = extract_nutrition_with_ai(image_data)
        
        if not ai_result["success"]:
            return {
                "status": "error",
                "message": "Failed to extract nutrition data with AI",
                "error_details": ai_result,
                "grams": grams,
                "image_size": list(pil_image.size)
            }
        
        # Calculate nutrition for requested grams locally
        calculation_result = calculate_nutrition_for_grams(
            ai_result["nutrition_per_100g"], 
            grams
        )
        
        if not calculation_result["success"]:
            return {
                "status": "error",
                "message": "Failed to calculate nutrition",
                "error_details": calculation_result,
                "grams": grams,
                "image_size": list(pil_image.size)
            }
        
        return {
            "status": "success",
            "message": "Nutrition analysis completed successfully",
            "grams": grams,
            "image_size": list(pil_image.size),
            "analysis_method": "ai_vision_only + local_calculation",
            "nutrition_analysis": calculation_result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/analyze-nutrition-raw")
async def analyze_nutrition_raw(
    request: Request, 
    grams: float, 
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Analyze nutrition from raw image data using AI only"""
    try:
        # Authenticate
        authenticate_user(credentials)
        
        # Validate grams input
        if grams <= 0:
            raise HTTPException(status_code=400, detail="Grams must be a positive number")
        
        # Read raw image data
        image_data = await request.body()
        pil_image = Image.open(io.BytesIO(image_data))
        
        # Extract nutrition using AI only
        ai_result = extract_nutrition_with_ai(image_data)
        
        if not ai_result["success"]:
            return {
                "status": "error",
                "message": "Failed to extract nutrition data with AI",
                "error_details": ai_result,
                "grams": grams,
                "image_size": list(pil_image.size)
            }
        
        # Calculate nutrition for requested grams locally
        calculation_result = calculate_nutrition_for_grams(
            ai_result["nutrition_per_100g"], 
            grams
        )
        
        if not calculation_result["success"]:
            return {
                "status": "error",
                "message": "Failed to calculate nutrition",
                "error_details": calculation_result,
                "grams": grams,
                "image_size": list(pil_image.size)
            }
        
        return {
            "status": "success",
            "message": "Nutrition analysis completed successfully",
            "grams": grams,
            "image_size": list(pil_image.size),
            "analysis_method": "ai_vision_only + local_calculation",
            "nutrition_analysis": calculation_result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    # Remove the duplicate load_dotenv() call here
    uvicorn.run(app, host="0.0.0.0", port=port)