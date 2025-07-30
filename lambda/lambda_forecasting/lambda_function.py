import json
import boto3
import os
from datetime import datetime, timedelta
import logging
import random
from typing import Dict, Any, List, Tuple, Optional
from boto3.dynamodb.conditions import Key
import math

# Only import pickle if needed to avoid potential conflicts
try:
    import pickle
except ImportError:
    pickle = None

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
s3 = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')

# DynamoDB table names from environment variables
USER_INTERACTIONS_TABLE = os.environ.get('USER_INTERACTIONS_TABLE', 'UserInteractions')
PRODUCT_EMBEDDINGS_TABLE = os.environ.get('PRODUCT_EMBEDDINGS_TABLE', 'ProductEmbeddings')
SALES_HISTORY_TABLE = os.environ.get('SALES_HISTORY_TABLE', 'SalesHistory')

# Helper functions to replace statistics functions
def calculate_mean(data: List[float]) -> float:
    """Calculate mean without using statistics module"""
    return sum(data) / len(data) if data else 0

def calculate_stdev(data: List[float]) -> float:
    """Calculate standard deviation without using statistics module"""
    if len(data) < 2:
        return 0
    
    mean_val = calculate_mean(data)
    variance = sum((x - mean_val) ** 2 for x in data) / (len(data) - 1)
    return math.sqrt(variance)

# 1. DEFINE THE FORECASTING MODEL CLASS
class SalesForecastingModel:
    """Sales forecasting model with multiple methods"""
    def __init__(self):
        self.model_version = "1.0"
        
    @staticmethod
    def moving_average_forecast(data: List[float], window: int = 7, periods: int = 30) -> List[float]:
        """Simple moving average forecasting"""
        if len(data) < window:
            # If not enough data, use simple average
            avg = calculate_mean(data) if data else 0
            return [avg] * periods
            
        forecasts = []
        for i in range(periods):
            if i == 0:
                # First forecast uses last 'window' actual values
                window_data = data[-window:]
            else:
                # Subsequent forecasts use mix of actual and predicted values
                window_data = data[-(window-i):] + forecasts[:i] if i < window else forecasts[-window:]
            
            forecast = calculate_mean(window_data)
            forecasts.append(forecast)
            
        return forecasts
    
    @staticmethod
    def exponential_smoothing_forecast(data: List[float], alpha: float = 0.3, periods: int = 30) -> List[float]:
        """Exponential smoothing forecasting"""
        if not data:
            return [0] * periods
            
        # Calculate initial smoothed value
        smoothed = data[0]
        for value in data[1:]:
            smoothed = alpha * value + (1 - alpha) * smoothed
            
        # Generate forecasts (constant for simple exponential smoothing)
        return [smoothed] * periods
    
    @staticmethod
    def linear_trend_forecast(data: List[float], periods: int = 30) -> List[float]:
        """Linear trend forecasting"""
        if len(data) < 2:
            avg = calculate_mean(data) if data else 0
            return [avg] * periods
            
        # Calculate linear trend
        n = len(data)
        x = list(range(n))
        y = data
        
        # Linear regression calculation
        x_mean = calculate_mean(x)
        y_mean = calculate_mean(y)
        
        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return [y_mean] * periods
            
        slope = numerator / denominator
        intercept = y_mean - slope * x_mean
        
        # Generate forecasts
        forecasts = []
        for i in range(periods):
            forecast = intercept + slope * (n + i)
            forecasts.append(max(0, forecast))  # Ensure non-negative
            
        return forecasts
    
    @staticmethod
    def seasonal_forecast(data: List[float], seasonality: int = 7, periods: int = 30) -> List[float]:
        """Simple seasonal forecasting"""
        if len(data) < seasonality:
            avg = calculate_mean(data) if data else 0
            return [avg] * periods
            
        forecasts = []
        for i in range(periods):
            # Use seasonal pattern from historical data
            seasonal_index = i % seasonality
            seasonal_data = [data[j] for j in range(seasonal_index, len(data), seasonality)]
            forecast = calculate_mean(seasonal_data) if seasonal_data else 0
            forecasts.append(forecast)
            
        return forecasts

# 2. MODEL LOADING WITH PROPER ERROR HANDLING
def load_forecasting_model(bucket: str, key: str) -> SalesForecastingModel:
    """Safe model loading with multiple fallbacks"""
    try:
        # Skip model loading if pickle is not available or if there's an issue
        if pickle is None:
            logger.warning("Pickle not available, using fallback implementation")
            return SalesForecastingModel()
            
        # Try loading from S3 first
        local_path = '/tmp/forecasting_model.pkl'
        s3.download_file(bucket, key, local_path)
        
        with open(local_path, 'rb') as f:
            model = pickle.load(f)
            logger.info("Successfully loaded forecasting model from S3")
            return model
            
    except Exception as e:
        logger.warning(f"Model load failed: {str(e)}. Using fallback implementation")
        return SalesForecastingModel()

# 3. DATA EXTRACTION FUNCTIONS
def get_historical_sales_data(product_id: str = None, category: str = None, days: int = 90) -> List[Dict[str, Any]]:
    """Get historical sales data from DynamoDB"""
    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Get sales data from UserInteractions table
        user_interactions_table = dynamodb.Table(USER_INTERACTIONS_TABLE)
        
        # Scan for purchase interactions within date range
        filter_expression = Key('action').eq('purchase')
        
        if product_id:
            filter_expression = filter_expression & Key('product_id').eq(product_id)
            
        # Note: For production, you might want to use a GSI with date as sort key
        response = user_interactions_table.scan(
            FilterExpression=filter_expression,
            Limit=1000  # Limit to prevent large scans
        )
        
        sales_data = []
        for item in response.get('Items', []):
            try:
                item_date = datetime.fromisoformat(item['timestamp'].replace('Z', '+00:00'))
                if start_date <= item_date <= end_date:
                    sales_data.append({
                        'date': item_date.strftime('%Y-%m-%d'),
                        'product_id': item.get('product_id'),
                        'amount': float(item.get('amount', 0)),
                        'quantity': int(item.get('quantity', 1)),
                        'category': item.get('category', 'unknown')
                    })
            except (ValueError, KeyError) as e:
                logger.warning(f"Skipping invalid sales record: {e}")
                continue
                
        logger.info(f"Retrieved {len(sales_data)} sales records")
        return sales_data
        
    except Exception as e:
        logger.error(f"Failed to get historical sales data: {str(e)}")
        return []

def aggregate_sales_by_date(sales_data: List[Dict[str, Any]], metric: str = 'amount') -> Dict[str, float]:
    """Aggregate sales data by date"""
    aggregated = {}
    
    for sale in sales_data:
        date = sale['date']
        value = sale.get(metric, 0)
        
        if date in aggregated:
            aggregated[date] += value
        else:
            aggregated[date] = value
            
    return aggregated

def fill_missing_dates(data: Dict[str, float], start_date: datetime, end_date: datetime) -> List[float]:
    """Fill missing dates with zeros and return as ordered list"""
    result = []
    current_date = start_date
    
    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')
        result.append(data.get(date_str, 0))
        current_date += timedelta(days=1)
        
    return result

# 4. MAIN FORECASTING FUNCTION
def generate_forecast(product_id: str = None, category: str = None, method: str = 'moving_average', 
                     periods: int = 30, metric: str = 'amount') -> Dict[str, Any]:
    """Generate sales forecast"""
    try:
        # Get historical data
        historical_data = get_historical_sales_data(product_id, category, days=90)
        
        if not historical_data:
            logger.warning("No historical data found, using random forecast")
            return {
                'forecast': [random.uniform(100, 500) for _ in range(periods)],
                'method': 'fallback',
                'confidence': 'low',
                'historical_data_points': 0
            }
        
        # Aggregate by date
        aggregated_data = aggregate_sales_by_date(historical_data, metric)
        
        # Fill missing dates
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        time_series = fill_missing_dates(aggregated_data, start_date, end_date)
        
        # Load model
        model = load_forecasting_model(
            os.environ.get('FORECASTING_MODEL_BUCKET', 'techmart-ml-handi'),
            os.environ.get('FORECASTING_MODEL_KEY', 'models/forecasting_model.pkl')
        )
        
        # Generate forecast based on method
        if method == 'moving_average':
            forecast = model.moving_average_forecast(time_series, periods=periods)
        elif method == 'exponential_smoothing':
            forecast = model.exponential_smoothing_forecast(time_series, periods=periods)
        elif method == 'linear_trend':
            forecast = model.linear_trend_forecast(time_series, periods=periods)
        elif method == 'seasonal':
            forecast = model.seasonal_forecast(time_series, periods=periods)
        else:
            # Default to moving average
            forecast = model.moving_average_forecast(time_series, periods=periods)
            
        # Calculate confidence based on data quality
        confidence = 'high' if len(historical_data) > 50 else 'medium' if len(historical_data) > 20 else 'low'
        
        return {
            'forecast': forecast,
            'method': method,
            'confidence': confidence,
            'historical_data_points': len(historical_data),
            'historical_average': calculate_mean(time_series) if time_series else 0,
            'forecast_dates': [(datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d') 
                             for i in range(1, periods + 1)]
        }
        
    except Exception as e:
        logger.error(f"Forecast generation failed: {str(e)}")
        return {
            'forecast': [random.uniform(100, 500) for _ in range(periods)],
            'method': 'fallback',
            'confidence': 'low',
            'error': str(e)
        }

# 5. LAMBDA HANDLER
def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    try:
        # Parse input
        body = event
        if isinstance(event.get('body'), str):
            body = json.loads(event['body'])
            
        # Extract parameters
        product_id = body.get('product_id')
        category = body.get('category')
        method = body.get('method', 'moving_average')
        periods = int(body.get('periods', 30))
        metric = body.get('metric', 'amount')  # 'amount' or 'quantity'
        
        # Validate parameters
        valid_methods = ['moving_average', 'exponential_smoothing', 'linear_trend', 'seasonal']
        if method not in valid_methods:
            return {
                'statusCode': 400,
                'body': json.dumps({
                    'error': 'Invalid method',
                    'valid_methods': valid_methods
                })
            }
            
        if periods > 365:
            return {
                'statusCode': 400,
                'body': json.dumps({
                    'error': 'Periods cannot exceed 365 days'
                })
            }
        
        logger.info(f"Generating forecast - product_id: {product_id}, category: {category}, method: {method}")
        
        # Generate forecast
        forecast_result = generate_forecast(product_id, category, method, periods, metric)
        
        # Calculate summary statistics
        forecast_values = forecast_result['forecast']
        summary = {
            'total_forecast': sum(forecast_values),
            'average_daily': calculate_mean(forecast_values),
            'min_daily': min(forecast_values),
            'max_daily': max(forecast_values),
            'std_deviation': calculate_stdev(forecast_values) if len(forecast_values) > 1 else 0
        }
        
        # Format response
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'forecast': forecast_result['forecast'],
                'forecast_dates': forecast_result.get('forecast_dates', []),
                'method': forecast_result['method'],
                'confidence': forecast_result['confidence'],
                'metric': metric,
                'periods': periods,
                'summary': summary,
                'historical_data_points': forecast_result.get('historical_data_points', 0),
                'historical_average': forecast_result.get('historical_average', 0),
                'timestamp': datetime.now().isoformat(),
                'status': 'success'
            })
        }
        
    except Exception as e:
        logger.error(f"Handler error: {str(e)}", exc_info=True)
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': 'Internal server error',
                'message': str(e),
                'timestamp': datetime.now().isoformat()
            })
        }