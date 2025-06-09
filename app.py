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

# Load environment variables
load_dotenv(override=True)

# Initialize OpenAI client
try:
    from openai import OpenAI
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Warning: OPENAI_API_KEY not found in environment variables")
        client = None
    else:
        client = OpenAI(api_key=openai_api_key)
        print("OpenAI client initialized successfully")
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
    if not api_secret_key:
        raise HTTPException(status_code=500, detail="API secret key not configured")
    
    if credentials.credentials != api_secret_key:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

def extract_nutrition_with_ai(image_bytes):
    """Extract nutrition data using GPT-4o-mini"""
    if client is None:
        return {
            "success": False,
            "error": "OpenAI client not initialized. Check API key."
        }
    
    try:
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
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
        
        response = client.chat.completions.create(
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
        
        ai_response = response.choices[0].message.content.strip()
        
        # Clean JSON response
        if '```json' in ai_response:
            ai_response = ai_response.split('```json')[1].split('```')[0].strip()
        elif '```' in ai_response:
            ai_response = ai_response.split('```')[1].strip()
        
        start_idx = ai_response.find('{')
        end_idx = ai_response.rfind('}') + 1
        if start_idx != -1 and end_idx != 0:
            ai_response = ai_response[start_idx:end_idx]
        
        nutrition_data = json.loads(ai_response)
        
        # Validate required fields
        required_fields = ['calories', 'protein_grams', 'sugar_grams', 'carbs_grams']
        validated_data = {}
        
        for field in required_fields:
            try:
                validated_data[field] = float(nutrition_data.get(field, 0))
            except (ValueError, TypeError):
                validated_data[field] = 0.0
        
        return {
            "success": True,
            "nutrition_data": validated_data,
            "raw_ai_response": ai_response
        }
        
    except json.JSONDecodeError as e:
        return {
            "success": False,
            "error": f"Failed to parse AI response as JSON: {str(e)}",
            "raw_ai_response": ai_response if 'ai_response' in locals() else "No response"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"AI extraction failed: {str(e)}"
        }

def calculate_nutrition_for_grams(nutrition_per_100g: dict, target_grams: float) -> dict:
    """Calculate nutrition values for specific gram amount"""
    try:
        scaling_factor = target_grams / 100.0
        
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
    return {
        "status": "healthy",
        "message": "Nutrition Facts API v3.0",
        "openai_status": "connected" if client else "not configured",
        "features": ["AI nutrition extraction", "Local calculations", "4 core nutrients"]
    }

@app.post("/analyze-nutrition")
async def analyze_nutrition(
    image: UploadFile = File(...),
    grams: float = Form(...),
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Analyze nutrition from uploaded image"""
    try:
        authenticate_user(credentials)
        
        if grams <= 0:
            raise HTTPException(status_code=400, detail="Grams must be positive")
        
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data))
        
        ai_result = extract_nutrition_with_ai(image_data)
        
        if not ai_result["success"]:
            return {
                "status": "error",
                "message": "Failed to extract nutrition data",
                "error_details": ai_result,
                "grams": grams,
                "image_size": list(pil_image.size)
            }
        
        calculation_result = calculate_nutrition_for_grams(
            ai_result["nutrition_data"], grams
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
            "message": "Nutrition analysis completed",
            "grams": grams,
            "image_size": list(pil_image.size),
            "analysis_method": "ai_vision + local_calculation",
            "nutrition_analysis": calculation_result,
            "ai_response": ai_result.get("raw_ai_response", "No AI response")
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/analyze-nutrition-raw")
async def analyze_nutrition_raw(
    request: Request,
    grams: float,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Analyze nutrition from raw image data"""
    try:
        authenticate_user(credentials)
        
        if grams <= 0:
            raise HTTPException(status_code=400, detail="Grams must be positive")
        
        image_data = await request.body()
        pil_image = Image.open(io.BytesIO(image_data))
        
        ai_result = extract_nutrition_with_ai(image_data)
        
        if not ai_result["success"]:
            return {
                "status": "error",
                "message": "Failed to extract nutrition data",
                "error_details": ai_result,
                "grams": grams,
                "image_size": list(pil_image.size)
            }
        
        calculation_result = calculate_nutrition_for_grams(
            ai_result["nutrition_data"], grams  # Changed from nutrition_per_100g to nutrition_data
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
            "message": "Nutrition analysis completed",
            "grams": grams,
            "image_size": list(pil_image.size),
            "analysis_method": "ai_vision + local_calculation",
            "nutrition_analysis": calculation_result,
            "ai_response": ai_result.get("raw_ai_response", "No AI response")
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)