# app.py - FastAPI deployment using optimized model
from model import AutoFillEmbeddingNN
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
import pickle
import logging
import os
from typing import List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="🔧 WakaFix AutoFill API", 
    description="AI-powered technician recommendation system",
    version="1.0.0"
)

# Define the model architecture (copied from your model.py)
class AutoFillEmbeddingNN(nn.Module):
    def __init__(self, n_service, n_location, n_time, embedding_dim, hidden_size, output_size):
        super(AutoFillEmbeddingNN, self).__init__()
        self.service_embed = nn.Embedding(n_service, embedding_dim)
        self.location_embed = nn.Embedding(n_location, embedding_dim)
        self.time_embed = nn.Embedding(n_time, embedding_dim)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)

        # Fully connected layers
        self.fc1 = nn.Linear(embedding_dim * 3, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)

        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)

    def forward(self, s, l, t):
        s_embed = self.service_embed(s)
        l_embed = self.location_embed(l)
        t_embed = self.time_embed(t)

        x = torch.cat([s_embed, l_embed, t_embed], dim=1)
        x = torch.nn.functional.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.nn.functional.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        return self.fc3(x)

# Global variables for model and artifacts
model = None
artifacts = None

class PredictionRequest(BaseModel):
    service_type: str
    location: str
    time_slot: str
    top_k: int = 3

class PredictionResponse(BaseModel):
    predictions: List[str]
    confidence_scores: List[float]
    status: str
    inference_time_ms: float

@app.on_event("startup")
async def startup_event():
    """Load optimized model and artifacts at startup"""
    global model, artifacts
    
    try:
        logger.info("🚀 Loading WakaFix model...")
        
        # First try to load artifacts (needed for both model types)
        if os.path.exists('wakafix_model_artifacts.pkl'):
            with open('wakafix_model_artifacts.pkl', 'rb') as f:
                artifacts = pickle.load(f)
            logger.info("✅ Model artifacts loaded!")
        else:
            logger.warning("⚠️ Model artifacts file not found!")
            # Create dummy artifacts for testing
            artifacts = {
                'label_encoders': {
                    'service': type('MockEncoder', (), {
                        'classes_': ['Plumbing', 'Electrical', 'HVAC'],
                        'transform': lambda self, x: [['Plumbing', 'Electrical', 'HVAC'].index(x[0])],
                        'inverse_transform': lambda self, x: [['Plumbing', 'Electrical', 'HVAC'][x[0]]]
                    })(),
                    'location': type('MockEncoder', (), {
                        'classes_': ['Downtown', 'Uptown', 'Suburbs'],
                        'transform': lambda self, x: [['Downtown', 'Uptown', 'Suburbs'].index(x[0])],
                        'inverse_transform': lambda self, x: [['Downtown', 'Uptown', 'Suburbs'][x[0]]]
                    })(),
                    'time': type('MockEncoder', (), {
                        'classes_': ['Morning', 'Afternoon', 'Evening'],
                        'transform': lambda self, x: [['Morning', 'Afternoon', 'Evening'].index(x[0])],
                        'inverse_transform': lambda self, x: [['Morning', 'Afternoon', 'Evening'][x[0]]]
                    })(),
                    'technician': type('MockEncoder', (), {
                        'classes_': ['John Doe', 'Jane Smith', 'Bob Wilson'],
                        'transform': lambda self, x: [['John Doe', 'Jane Smith', 'Bob Wilson'].index(x[0])],
                        'inverse_transform': lambda self, x: [['John Doe', 'Jane Smith', 'Bob Wilson'][x[0]]]
                    })()
                },
                'model_config': {
                    'n_service': 3,
                    'n_location': 3,
                    'n_time': 3,
                    'embedding_dim': 50,
                    'hidden_size': 128,
                    'output_size': 3
                }
            }
            logger.info("✅ Using mock artifacts for testing!")
        
        # Check if optimized model exists
        if os.path.exists('wakafix_optimized.pt'):
            logger.info("📦 Loading optimized model...")
            model = torch.jit.load('wakafix_optimized.pt', map_location='cpu')
            model.eval()
            logger.info("✅ Optimized model loaded successfully!")
            
        elif os.path.exists('best_wakafix_model.pth'):
            logger.info("📦 Loading standard model...")
            
            # Load checkpoint first to inspect it
            checkpoint = torch.load('best_wakafix_model.pth', map_location='cpu')
            logger.info(f"📋 Checkpoint keys: {list(checkpoint.keys())}")
            
            # Check if model_state_dict exists
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                logger.info(f"📋 Model state dict keys: {list(state_dict.keys())}")
            else:
                # Sometimes the entire checkpoint is the state dict
                state_dict = checkpoint
                logger.info(f"📋 Direct state dict keys: {list(state_dict.keys())}")
                logger.info("⚠️ No 'model_state_dict' key found, using checkpoint directly")
            
            # Try to infer model architecture from state dict
            try:
                # Get embedding dimensions from the saved weights
                service_emb_weight = None
                location_emb_weight = None
                time_emb_weight = None
                
                for key, tensor in state_dict.items():
                    if 'service_embed.weight' in key:
                        service_emb_weight = tensor
                        logger.info(f"📏 Service embedding: {tensor.shape}")
                    elif 'location_embed.weight' in key:
                        location_emb_weight = tensor
                        logger.info(f"📏 Location embedding: {tensor.shape}")
                    elif 'time_embed.weight' in key:
                        time_emb_weight = tensor
                        logger.info(f"📏 Time embedding: {tensor.shape}")
                    elif 'fc1.weight' in key:
                        logger.info(f"📏 FC1 layer: {tensor.shape}")
                    elif 'fc2.weight' in key:
                        logger.info(f"📏 FC2 layer: {tensor.shape}")
                    elif 'fc3.weight' in key:
                        logger.info(f"📏 FC3 layer (output): {tensor.shape}")
                    elif 'bn1' in key:
                        logger.info(f"📏 BatchNorm1: {key} - {tensor.shape}")
                    elif 'bn2' in key:
                        logger.info(f"📏 BatchNorm2: {key} - {tensor.shape}")
                
                # Infer parameters from saved weights
                if service_emb_weight is not None and location_emb_weight is not None and time_emb_weight is not None:
                    n_service = service_emb_weight.shape[0]
                    n_location = location_emb_weight.shape[0]
                    n_time = time_emb_weight.shape[0]
                    embedding_dim = service_emb_weight.shape[1]
                    
                    # Get hidden sizes from FC layers
                    fc1_key = next((k for k in state_dict.keys() if 'fc1.weight' in k), None)
                    fc2_key = next((k for k in state_dict.keys() if 'fc2.weight' in k), None)
                    fc3_key = next((k for k in state_dict.keys() if 'fc3.weight' in k), None)
                    
                    if fc1_key and fc2_key and fc3_key:
                        hidden_size = state_dict[fc1_key].shape[0]
                        output_size = state_dict[fc3_key].shape[0]
                        
                        logger.info(f"🔍 Inferred model params:")
                        logger.info(f"   n_service: {n_service}")
                        logger.info(f"   n_location: {n_location}")
                        logger.info(f"   n_time: {n_time}")
                        logger.info(f"   embedding_dim: {embedding_dim}")
                        logger.info(f"   hidden_size: {hidden_size}")
                        logger.info(f"   output_size: {output_size}")
                        
                        # Create model with inferred parameters
                        model = AutoFillEmbeddingNN(
                            n_service, n_location, n_time,
                            embedding_dim, hidden_size, output_size
                        )
                        
                        # Load the state dict
                        model.load_state_dict(state_dict)
                        model.eval()
                        logger.info("✅ Standard model loaded successfully with inferred parameters!")
                        
                        # Update artifacts with correct config
                        artifacts['model_config'] = {
                            'n_service': n_service,
                            'n_location': n_location,
                            'n_time': n_time,
                            'embedding_dim': embedding_dim,
                            'hidden_size': hidden_size,
                            'output_size': output_size
                        }
                        
                    else:
                        raise ValueError("Could not find all required FC layers in state dict")
                else:
                    raise ValueError("Could not find all embedding layers in state dict")
                    
            except Exception as e:
                logger.error(f"❌ Could not infer model architecture: {e}")
                logger.info("🔄 Falling back to config from artifacts...")
                
                # Fallback to original method
                config = artifacts['model_config']
                model = AutoFillEmbeddingNN(
                    config['n_service'], 
                    config['n_location'], 
                    config['n_time'],
                    config['embedding_dim'], 
                    config['hidden_size'], 
                    config['output_size']
                )
                
                try:
                    model.load_state_dict(state_dict)
                    model.eval()
                    logger.info("✅ Standard model loaded successfully with config fallback!")
                except Exception as load_error:
                    logger.error(f"❌ Failed to load model with config: {load_error}")
                    raise load_error
        else:
            logger.warning("⚠️ No trained model found, creating dummy model for testing...")
            # Create a dummy model for testing
            config = artifacts['model_config']
            model = AutoFillEmbeddingNN(
                config['n_service'], 
                config['n_location'], 
                config['n_time'],
                config['embedding_dim'], 
                config['hidden_size'], 
                config['output_size']
            )
            model.eval()
            logger.info("✅ Dummy model created for testing!")
            
        # Log model info
        logger.info(f"📊 Service types: {len(artifacts['label_encoders']['service'].classes_)}")
        logger.info(f"📊 Locations: {len(artifacts['label_encoders']['location'].classes_)}")
        logger.info(f"📊 Time slots: {len(artifacts['label_encoders']['time'].classes_)}")
        logger.info(f"📊 Technicians: {len(artifacts['label_encoders']['technician'].classes_)}")
        
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        raise e

@app.get("/predict", include_in_schema=False)
async def predict_info():
    """GET endpoint for /predict - provides usage information"""
    if artifacts is None:
        return {
            "error": "Model not loaded",
            "message": "Please check /health endpoint for model status"
        }
    
    # Get sample values for demonstration
    le_service = artifacts['label_encoders']['service']
    le_location = artifacts['label_encoders']['location'] 
    le_time = artifacts['label_encoders']['time']
    
    return {
        "message": "Use POST method with JSON body to get predictions",
        "method": "POST",
        "endpoint": "/predict",
        "content_type": "application/json",
        "request_format": {
            "service_type": "string (required)",
            "location": "string (required)", 
            "time_slot": "string (required)",
            "top_k": "integer (optional, default: 3)"
        },
        "example_request": {
            "service_type": le_service.classes_[0] if len(le_service.classes_) > 0 else "plumbing",
            "location": le_location.classes_[0] if len(le_location.classes_) > 0 else "Downtown",
            "time_slot": le_time.classes_[0] if len(le_time.classes_) > 0 else "Morning",
            "top_k": 3
        },
        "curl_example": f"""curl -X POST "http://localhost:8000/predict" \\
     -H "Content-Type: application/json" \\
     -d '{{"service_type": "{le_service.classes_[0] if len(le_service.classes_) > 0 else 'plumbing'}", "location": "{le_location.classes_[0] if len(le_location.classes_) > 0 else 'Downtown'}", "time_slot": "{le_time.classes_[0] if len(le_time.classes_) > 0 else 'Morning'}", "top_k": 3}}'""",
        "helpful_endpoints": {
            "service_types": "/service-types - Get all accepted service type variations",
            "model_info": "/model-info - Get available options for all fields",
            "test": "/test-predict - Quick test with sample data",
            "docs": "/docs - Interactive API documentation"
        }
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_technician(request: PredictionRequest):
    """Make technician predictions with flexible service type mapping"""
    if model is None or artifacts is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        import time
        start_time = time.time()
        
        # Service type mapping for flexible input handling
        service_type_map = {
            # Plumbing variations
            "plumbing": "plumber",
            "plumber": "plumber",
            "pipe repair": "plumber",
            "water leak": "plumber",
            "drain cleaning": "plumber",
            "faucet repair": "plumber",
            
            # Electrical variations
            "electrical": "electrician",
            "electrician": "electrician",
            "electric": "electrician",
            "wiring": "electrician",
            "electrical repair": "electrician",
            "power issue": "electrician",
            "outlet repair": "electrician",
            
            # HVAC/AC variations
            "hvac": "HVAC technician",
            "ac": "HVAC technician",
            "air conditioning": "HVAC technician",
            "ac repair": "HVAC technician",
            "hvac technician": "HVAC technician",
            "heating": "HVAC technician",
            "cooling": "HVAC technician",
            "air conditioner": "HVAC technician",
            
            # Cleaning variations
            "cleaning": "cleaner",
            "cleaner": "cleaner",
            "house cleaning": "cleaner",
            "deep cleaning": "cleaner",
            "maintenance cleaning": "cleaner",
            
            # Painting variations
            "painting": "painter",
            "painter": "painter",
            "paint job": "painter",
            "wall painting": "painter",
            "interior painting": "painter",
            "exterior painting": "painter",
            
            # Carpentry variations
            "carpentry": "carpenter",
            "carpenter": "carpenter",
            "wood work": "carpenter",
            "furniture repair": "carpenter",
            "cabinet work": "carpenter",
            
            # Appliance variations
            "appliance repair": "appliance technician",
            "appliance": "appliance technician",
            "appliance technician": "appliance technician",
            "washing machine": "appliance technician",
            "refrigerator": "appliance technician",
            "dishwasher": "appliance technician"
        }
        
        # Normalize input to lowercase for matching
        input_service = request.service_type.lower().strip()
        
        # Try to map the input to a standardized service type
        mapped_service = service_type_map.get(input_service)
        
        if mapped_service:
            normalized_service_type = mapped_service
            logger.info(f"🔄 Mapped '{request.service_type}' → '{normalized_service_type}'")
        else:
            # If no mapping found, try the original input
            normalized_service_type = request.service_type
            logger.info(f"⚠️ No mapping found for '{request.service_type}', using original")
        
        # Validate inputs exist in training data
        le_service = artifacts['label_encoders']['service']
        le_location = artifacts['label_encoders']['location']
        le_time = artifacts['label_encoders']['time']
        le_technician = artifacts['label_encoders']['technician']
        
        # Check if the normalized service type is valid
        if normalized_service_type not in le_service.classes_:
            # If still not valid, provide helpful error with available options
            available_services = list(le_service.classes_)
            available_inputs = [k for k, v in service_type_map.items() if v in available_services]
            
            raise HTTPException(
                status_code=400, 
                detail={
                    "error": f"Invalid service_type: '{request.service_type}'",
                    "available_exact_matches": available_services,
                    "accepted_variations": available_inputs[:10],  # Show first 10 variations
                    "suggestion": "Try variations like 'plumbing', 'electrical', 'hvac', 'cleaning', etc."
                }
            )
            
        if request.location not in le_location.classes_:
            available_locations = list(le_location.classes_)[:5]
            raise HTTPException(
                status_code=400,
                detail=f"Invalid location. Available: {available_locations}{'...' if len(le_location.classes_) > 5 else ''}"
            )
            
        if request.time_slot not in le_time.classes_:
            available_times = list(le_time.classes_)
            raise HTTPException(
                status_code=400,
                detail=f"Invalid time_slot. Available: {available_times}"
            )
        
        # Encode inputs using the normalized service type
        service_enc = le_service.transform([normalized_service_type])[0]
        location_enc = le_location.transform([request.location])[0]
        time_enc = le_time.transform([request.time_slot])[0]
        
        # Convert to tensors
        service_tensor = torch.tensor([service_enc], dtype=torch.long)
        location_tensor = torch.tensor([location_enc], dtype=torch.long)
        time_tensor = torch.tensor([time_enc], dtype=torch.long)
        
        # Get predictions
        with torch.no_grad():
            if hasattr(model, 'forward'):
                # Standard PyTorch model
                outputs = model(service_tensor, location_tensor, time_tensor)
            else:
                # JIT traced model
                outputs = model(service_tensor, location_tensor, time_tensor)
                
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Get top-k predictions
            top_k = min(request.top_k, len(le_technician.classes_))
            _, top_indices = torch.topk(probabilities, top_k, dim=1)
            
        # Decode predictions
        predictions = []
        confidence_scores = []
        
        for i in range(top_k):
            tech_idx = top_indices[0][i].item()
            tech_name = le_technician.inverse_transform([tech_idx])[0]
            confidence = probabilities[0][tech_idx].item()
            predictions.append(tech_name)
            confidence_scores.append(round(confidence, 4))
            
        inference_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        return PredictionResponse(
            predictions=predictions,
            confidence_scores=confidence_scores,
            status="success",
            inference_time_ms=round(inference_time, 2)
        )
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "🔧 WakaFix AutoFill API is running!",
        "status": "healthy",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "model_info": "/model-info",
            "service_types": "/service-types",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_loaded = model is not None
    artifacts_loaded = artifacts is not None
    
    return {
        "status": "healthy" if model_loaded and artifacts_loaded else "unhealthy",
        "model_loaded": model_loaded,
        "artifacts_loaded": artifacts_loaded,
        "model_type": "optimized" if hasattr(model, '_c') else "standard" if model_loaded else "none"
    }

@app.get("/model-info")
async def model_info():
    """Get model information"""
    if artifacts is None:
        raise HTTPException(status_code=500, detail="Model artifacts not loaded")
    
    le_service = artifacts['label_encoders']['service']
    le_location = artifacts['label_encoders']['location'] 
    le_time = artifacts['label_encoders']['time']
    le_technician = artifacts['label_encoders']['technician']
    
    return {
        "model_config": artifacts['model_config'],
        "available_options": {
            "service_types": list(le_service.classes_),
            "locations": list(le_location.classes_),
            "time_slots": list(le_time.classes_),
            "technicians": list(le_technician.classes_)
        },
        "optimization_info": artifacts.get('optimization_info', {})
    }

@app.get("/service-types")
async def get_service_types():
    """Get available service types and their accepted variations"""
    if artifacts is None:
        raise HTTPException(status_code=500, detail="Model artifacts not loaded")
    
    # Service type mapping (same as in predict endpoint)
    service_type_map = {
        # Plumbing variations
        "plumbing": "plumber",
        "plumber": "plumber",
        "pipe repair": "plumber",
        "water leak": "plumber",
        "drain cleaning": "plumber",
        "faucet repair": "plumber",
        
        # Electrical variations
        "electrical": "electrician",
        "electrician": "electrician",
        "electric": "electrician",
        "wiring": "electrician",
        "electrical repair": "electrician",
        "power issue": "electrician",
        "outlet repair": "electrician",
        
        # HVAC/AC variations
        "hvac": "HVAC technician",
        "ac": "HVAC technician",
        "air conditioning": "HVAC technician",
        "ac repair": "HVAC technician",
        "hvac technician": "HVAC technician",
        "heating": "HVAC technician",
        "cooling": "HVAC technician",
        "air conditioner": "HVAC technician",
        
        # Cleaning variations
        "cleaning": "cleaner",
        "cleaner": "cleaner",
        "house cleaning": "cleaner",
        "deep cleaning": "cleaner",
        "maintenance cleaning": "cleaner",
        
        # Painting variations
        "painting": "painter",
        "painter": "painter",
        "paint job": "painter",
        "wall painting": "painter",
        "interior painting": "painter",
        "exterior painting": "painter",
        
        # Carpentry variations
        "carpentry": "carpenter",
        "carpenter": "carpenter",
        "wood work": "carpenter",
        "furniture repair": "carpenter",
        "cabinet work": "carpenter",
        
        # Appliance variations
        "appliance repair": "appliance technician",
        "appliance": "appliance technician",
        "appliance technician": "appliance technician",
        "washing machine": "appliance technician",
        "refrigerator": "appliance technician",
        "dishwasher": "appliance technician"
    }
    
    le_service = artifacts['label_encoders']['service']
    available_services = list(le_service.classes_)
    
    # Group variations by service type
    service_variations = {}
    for variation, service in service_type_map.items():
        if service in available_services:
            if service not in service_variations:
                service_variations[service] = []
            service_variations[service].append(variation)
    
    return {
        "available_service_types": available_services,
        "accepted_variations": service_variations,
        "total_variations": len(service_type_map),
        "usage_examples": [
            {"input": "plumbing", "maps_to": "plumber"},
            {"input": "ac repair", "maps_to": "HVAC technician"},
            {"input": "electrical", "maps_to": "electrician"},
            {"input": "cleaning", "maps_to": "cleaner"}
        ]
    }

# Test endpoint for development
@app.post("/test-predict")
async def test_predict():
    """Test prediction with sample data"""
    if artifacts is None:
        raise HTTPException(status_code=500, detail="Model not ready")
    
    # Get first available option for each category
    le_service = artifacts['label_encoders']['service']
    le_location = artifacts['label_encoders']['location'] 
    le_time = artifacts['label_encoders']['time']
    
    sample_request = PredictionRequest(
        service_type=le_service.classes_[0],
        location=le_location.classes_[0],
        time_slot=le_time.classes_[0],
        top_k=3
    )
    
    return await predict_technician(sample_request)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)