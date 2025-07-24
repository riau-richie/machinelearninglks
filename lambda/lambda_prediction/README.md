# Lambda Prediction

## ENVIRONMENT VARIABLES

### AWS Configuration
`MODEL_BUCKET=your bucket model`<br/>
`MODEL_KEY=models/hybrid_model.pkl`

### Dynamo Tables
`USERS_TABLE=techmart-users`
`PRODUCTS_TABLE=ProductEmbeddings`

## Prediction Test
### Method POST Testing

## Method POST Testing
```json
{
  "user_id": "user123",
  "product_id": "LAPTOP001"
}


