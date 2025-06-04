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
load_dotenv()

# Import OpenAI after loading env vars
try:
    from openai import OpenAI
    # Initialize OpenAI client with error handling
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Warning: OPENAI_API_KEY not found in environment variables")
        client = None
    else:
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
    if credentials.credentials != "k3y":
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
        
        # Highly controlled prompt for exact nutrition extraction
        controlled_prompt = """
You are a nutrition label analysis expert. Analyze this nutrition label image and extract EXACTLY 4 values.

IMPORTANT RULES:
1. Return ONLY a valid JSON object with these exact keys: calories, protein_grams, sugar_grams, carbs_grams
2. All values must be numbers (integers or decimals) representing amounts per 100 grams
3. If the label shows a different serving size, convert ALL values to per 100g basis
4. If any value is not clearly visible, use 0
5. Do not include any text, explanations, or formatting - ONLY the JSON object
6. Use decimal points for precision (e.g., 12.5, not 12 or 13)

Required JSON format:
{
  "calories": [number],
  "protein_grams": [number],
  "sugar_grams": [number],
  "carbs_grams": [number]
}

Example output:
{"calories": 250, "protein_grams": 8.5, "sugar_grams": 12.3, "carbs_grams": 45.2}

Analyze the nutrition label and return the JSON:"""
        
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
            max_tokens=200,
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
    uvicorn.run(app, host="0.0.0.0", port=8000)