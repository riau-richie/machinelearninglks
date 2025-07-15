import json
import boto3
import joblib
import os
import pickle
from datetime import datetime
import logging
import numpy as np

# Enhanced logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')

s3 = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')

# Cache for models
MODEL_CACHE = None
LAST_MODEL_LOAD_TIME = None
CACHE_EXPIRY_MINUTES = 30

def load_model_from_s3(bucket, key):
    global MODEL_CACHE, LAST_MODEL_LOAD_TIME
    
    current_time = datetime.now()
    if (MODEL_CACHE is None or 
        LAST_MODEL_LOAD_TIME is None or
        (current_time - LAST_MODEL_LOAD_TIME).total_seconds() > CACHE_EXPIRY_MINUTES * 60):
        
        local_model_path = '/tmp/model.pkl'
        try:
            s3.download_file(bucket, key, local_model_path)
            
            # Try both joblib and pickle loading
            try:
                MODEL_CACHE = joblib.load(local_model_path)
            except:
                with open(local_model_path, 'rb') as f:
                    MODEL_CACHE = pickle.load(f)
                    
            LAST_MODEL_LOAD_TIME = current_time
            logger.info("Successfully loaded and cached model")
            
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            raise
            
    return MODEL_CACHE

def lambda_handler(event, context):
    try:
        # Parse input with enhanced validation
        body = parse_input(event)
        user_id = body.get('user_id')
        product_id = body.get('product_id')
        
        # Load model with cache
        bucket = os.environ.get('MODEL_BUCKET', 'techmart-ml-handi')
        key = os.environ.get('MODEL_KEY', 'models/hybrid_model.pkl')
        model = load_model_from_s3(bucket, key)
        
        # Get enhanced features
        user_features, product_features = get_enhanced_features(user_id, product_id)
        
        # Make prediction
        prediction_result = make_prediction(model, user_features, product_features)
        
        return format_response(prediction_result)
        
    except Exception as e:
        logger.error(f"Error in lambda_handler: {str(e)}", exc_info=True)
        return error_response(str(e))

def parse_input(event):
    """Enhanced input parsing with validation"""
    if not event:
        raise ValueError("Empty event received")
        
    if 'body' in event:
        try:
            body = json.loads(event['body']) if isinstance(event['body'], str) else event['body']
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON in request body")
    else:
        body = event
        
    if not isinstance(body, dict):
        raise ValueError("Input must be a dictionary")
        
    return body

def get_enhanced_features(user_id, product_id):
    """Enhanced feature lookup with fallbacks"""
    user_features = None
    product_features = None
    
    try:
        # User features with more fields
        if user_id:
            user_table = dynamodb.Table('UserProfiles')
            response = user_table.get_item(Key={'user_id': user_id})
            if 'Item' in response:
                user_data = response['Item']
                user_features = {
                    'purchase_history': user_data.get('total_purchases', 0),
                    'avg_spending': user_data.get('avg_order_value', 0),
                    'demographics': {
                        'age': user_data.get('age'),
                        'gender': user_data.get('gender'),
                        'income': user_data.get('income_range')
                    },
                    'preferences': user_data.get('favorite_categories', [])
                }
                
        # Product features with more fields
        if product_id:
            product_table = dynamodb.Table('ProductEmbeddings')
            response = product_table.get_item(Key={'product_id': product_id})
            if 'Item' in response:
                product_data = response['Item']
                product_features = {
                    'category': product_data.get('category'),
                    'brand': product_data.get('brand'),
                    'price': product_data.get('price'),
                    'popularity': product_data.get('popularity', 0),
                    'quality': product_data.get('avg_rating', 0)
                }
                
    except Exception as e:
        logger.warning(f"Feature lookup warning: {str(e)}")
        
    return user_features, product_features

def make_prediction(model, user_features, product_features):
    """Enhanced prediction with better fallback logic"""
    try:
        if model and user_features and product_features:
            # Check if model has the expected method
            if hasattr(model, 'predict_purchase_probability'):
                prediction = model.predict_purchase_probability(user_features, product_features)
                source = 'full_model'
            else:
                # Try alternative prediction methods
                prediction = fallback_prediction(user_features, product_features)
                source = 'fallback_with_features'
        else:
            prediction = basic_fallback()
            source = 'basic_fallback'
            
        return {
            'prediction': float(prediction),
            'source': source,
            'features_used': {
                'user': bool(user_features),
                'product': bool(product_features)
            }
        }
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return {
            'prediction': basic_fallback(),
            'source': 'error_fallback',
            'error': str(e)
        }

def fallback_prediction(user_features, product_features):
    """Improved fallback prediction logic"""
    base_score = 0.5
    
    # User factors
    if user_features:
        base_score += user_features.get('purchase_history', 0) * 0.01
        if user_features['demographics']['income'] in ['>200k', '100-200k']:
            base_score += 0.1
            
    # Product factors
    if product_features:
        if product_features['price'] < 100:
            base_score += 0.15
        if product_features['quality'] > 4:
            base_score += 0.2
            
    return min(max(base_score, 0), 1)  # Clamp between 0-1

def basic_fallback():
    """Basic fallback when no features available"""
    return random.uniform(0.3, 0.6)

def format_response(prediction_result):
    """Enhanced response formatting"""
    base_response = {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        },
        'body': json.dumps({
            'prediction': prediction_result['prediction'],
            'prediction_percentage': round(prediction_result['prediction'] * 100, 2),
            'model_source': prediction_result.get('source', 'unknown'),
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        })
    }
    
    # Add additional info if available
    if 'features_used' in prediction_result:
        base_response['body']['features_used'] = prediction_result['features_used']
    if 'error' in prediction_result:
        base_response['body']['error'] = prediction_result['error']
        
    return base_response

def error_response(error_msg):
    """Standard error response"""
    return {
        'statusCode': 500,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        },
        'body': json.dumps({
            'error': error_msg,
            'message': 'Prediction failed',
            'timestamp': datetime.now().isoformat()
        })
    }