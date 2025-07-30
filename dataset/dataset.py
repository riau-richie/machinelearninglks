import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import random
import uuid

# Set random seed untuk reproducibility
np.random.seed(42)
random.seed(42)

# 1. GENERATE USER PROFILES DATASET
print("Generating user_profiles.csv...")

cities = ['Jakarta', 'Surabaya', 'Bandung', 'Medan', 'Semarang', 'Makassar', 'Palembang', 'Tangerang', 'Depok', 'Bekasi']
provinces = ['DKI Jakarta', 'Jawa Timur', 'Jawa Barat', 'Sumatera Utara', 'Jawa Tengah', 'Sulawesi Selatan', 'Sumatera Selatan', 'Banten', 'Jawa Barat', 'Jawa Barat']
income_ranges = ['<25k', '25-50k', '50-100k', '100-200k', '>200k']
segments = ['New', 'Regular', 'VIP', 'Premium']

users_data = []
for i in range(1, 1001):  # 1000 users
    user_id = f"user_{i:05d}"
    reg_date = datetime.now() - timedelta(days=random.randint(30, 730))
    age = random.randint(18, 65)
    gender = random.choice(['M', 'F'])
    city_idx = random.randint(0, len(cities)-1)
    city = cities[city_idx]
    province = provinces[city_idx]
    income = random.choice(income_ranges)
    total_purchases = random.randint(0, 50)
    total_spent = total_purchases * random.uniform(50, 500)
    avg_order_value = total_spent / total_purchases if total_purchases > 0 else 0
    last_active = datetime.now() - timedelta(days=random.randint(0, 30))
    
    if total_purchases >= 20:
        segment = 'VIP'
    elif total_purchases >= 10:
        segment = 'Regular'
    elif total_purchases >= 5:
        segment = 'Premium'
    else:
        segment = 'New'
    
    users_data.append({
        'user_id': user_id,
        'registration_date': reg_date.strftime('%Y-%m-%d'),
        'age': age,
        'gender': gender,
        'location_city': city,
        'location_province': province,
        'income_range': income,
        'total_purchases': total_purchases,
        'total_spent': round(total_spent, 2),
        'avg_order_value': round(avg_order_value, 2),
        'last_active': last_active.strftime('%Y-%m-%d %H:%M:%S'),
        'customer_segment': segment
    })

users_df = pd.DataFrame(users_data)
users_df.to_csv('user_profiles.csv', index=False)
print(f"✓ Generated user_profiles.csv with {len(users_df)} records")

# 2. GENERATE PRODUCT CATALOG DATASET
print("\nGenerating product_catalog.csv...")

categories = ['Electronics', 'Fashion', 'Home & Garden', 'Books', 'Sports', 'Beauty', 'Automotive', 'Toys']
brands = ['Samsung', 'Apple', 'Nike', 'Adidas', 'Sony', 'LG', 'Xiaomi', 'Uniqlo', 'Zara', 'H&M']

products_data = []
for i in range(1, 501):  # 500 products
    product_id = f"prod_{i:05d}"
    category = random.choice(categories)
    brand = random.choice(brands)
    
    # Generate realistic product names
    product_names = {
        'Electronics': [f'{brand} Smartphone Pro', f'{brand} Laptop Ultra', f'{brand} Headphones'],
        'Fashion': [f'{brand} T-Shirt', f'{brand} Jeans', f'{brand} Sneakers'],
        'Home & Garden': [f'{brand} Vacuum Cleaner', f'{brand} Coffee Maker', f'{brand} Air Purifier'],
        'Books': ['Python Programming Guide', 'Data Science Handbook', 'Machine Learning Basics'],
        'Sports': [f'{brand} Running Shoes', f'{brand} Fitness Tracker', f'{brand} Yoga Mat'],
        'Beauty': [f'{brand} Skincare Set', f'{brand} Makeup Kit', f'{brand} Perfume'],
        'Automotive': [f'{brand} Car Accessories', f'{brand} Motor Oil', f'{brand} Tire Set'],
        'Toys': [f'{brand} Action Figure', f'{brand} Building Blocks', f'{brand} Board Game']
    }
    
    name = random.choice(product_names.get(category, [f'{brand} Product']))
    original_price = random.uniform(50, 2000)
    discount = random.uniform(0, 30)
    price = original_price * (1 - discount/100)
    
    avg_rating = random.uniform(3.0, 5.0)
    num_reviews = random.randint(10, 1000)
    stock = random.randint(0, 200)
    release_date = datetime.now() - timedelta(days=random.randint(0, 1000))
    
    products_data.append({
        'product_id': product_id,
        'name': name,
        'category': category,
        'brand': brand,
        'price': round(price, 2),
        'original_price': round(original_price, 2),
        'discount_percentage': round(discount, 1),
        'avg_rating': round(avg_rating, 1),
        'num_reviews': num_reviews,
        'stock_quantity': stock,
        'release_date': release_date.strftime('%Y-%m-%d')
    })

products_df = pd.DataFrame(products_data)
products_df.to_csv('product_catalog.csv', index=False)
print(f"✓ Generated product_catalog.csv with {len(products_df)} records")

# 3. GENERATE USER INTERACTIONS DATASET
print("\nGenerating user_interactions.csv...")

interaction_types = ['view', 'cart', 'purchase', 'like']
device_types = ['mobile', 'desktop', 'tablet']

interactions_data = []
for i in range(10000):  # 10,000 interactions
    user_id = f"user_{random.randint(1, 1000):05d}"
    product_id = f"prod_{random.randint(1, 500):05d}"
    interaction_type = random.choice(interaction_types)
    
    # Make purchases less frequent than views
    if random.random() < 0.7:
        interaction_type = 'view'
    elif random.random() < 0.15:
        interaction_type = 'cart'
    elif random.random() < 0.1:
        interaction_type = 'purchase'
    else:
        interaction_type = 'like'
    
    timestamp = datetime.now() - timedelta(days=random.randint(0, 90))
    session_id = f"sess_{uuid.uuid4().hex[:8]}"
    device_type = random.choice(device_types)
    duration = random.randint(5, 300)
    
    # Only purchases have ratings
    rating = round(random.uniform(1, 5), 1) if interaction_type == 'purchase' else None
    
    interactions_data.append({
        'user_id': user_id,
        'product_id': product_id,
        'interaction_type': interaction_type,
        'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
        'session_id': session_id,
        'device_type': device_type,
        'duration_seconds': duration,
        'rating': rating
    })

interactions_df = pd.DataFrame(interactions_data)
interactions_df.to_csv('user_interactions.csv', index=False)
print(f"✓ Generated user_interactions.csv with {len(interactions_df)} records")

# 4. GENERATE TRANSACTION HISTORY DATASET
print("\nGenerating transaction_history.csv...")

payment_methods = ['credit_card', 'debit_card', 'e_wallet', 'bank_transfer']
order_statuses = ['completed', 'pending', 'cancelled', 'processing']

# Get only purchase interactions for transactions
purchase_interactions = interactions_df[interactions_df['interaction_type'] == 'purchase'].copy()

transactions_data = []
for _, interaction in purchase_interactions.iterrows():
    transaction_id = f"txn_{uuid.uuid4().hex[:8]}"
    user_id = interaction['user_id']
    product_id = interaction['product_id']
    
    # Get product price
    product_price = products_df[products_df['product_id'] == product_id]['price'].iloc[0]
    
    quantity = random.randint(1, 3)
    unit_price = product_price
    total_amount = unit_price * quantity
    discount_applied = total_amount * random.uniform(0, 0.2)  # 0-20% discount
    final_amount = total_amount - discount_applied
    
    payment_method = random.choice(payment_methods)
    transaction_date = interaction['timestamp']
    
    # Get user location for delivery
    user_location = users_df[users_df['user_id'] == user_id]['location_city'].iloc[0]
    delivery_address = f"{user_location}, Indonesia"
    
    order_status = random.choice(order_statuses)
    if order_status == 'cancelled':
        order_status = random.choice(['completed', 'completed', 'completed', 'cancelled'])  # 75% completed
    
    shipping_cost = random.uniform(10, 30)
    
    # Delivery date 2-7 days after transaction
    delivery_date = datetime.strptime(transaction_date, '%Y-%m-%d %H:%M:%S') + timedelta(days=random.randint(2, 7))
    
    transactions_data.append({
        'transaction_id': transaction_id,
        'user_id': user_id,
        'product_id': product_id,
        'quantity': quantity,
        'unit_price': round(unit_price, 2),
        'total_amount': round(total_amount, 2),
        'discount_applied': round(discount_applied, 2),
        'final_amount': round(final_amount, 2),
        'payment_method': payment_method,
        'transaction_date': transaction_date,
        'delivery_address': delivery_address,
        'order_status': order_status,
        'shipping_cost': round(shipping_cost, 2),
        'delivery_date': delivery_date.strftime('%Y-%m-%d')
    })

transactions_df = pd.DataFrame(transactions_data)
transactions_df.to_csv('transaction_history.csv', index=False)
print(f"✓ Generated transaction_history.csv with {len(transactions_df)} records")

# 5. GENERATE SUMMARY STATISTICS
print("\n" + "="*50)
print("DATASET SUMMARY")
print("="*50)

print(f"📊 Users: {len(users_df):,} records")
print(f"📦 Products: {len(products_df):,} records")
print(f"👆 Interactions: {len(interactions_df):,} records")
print(f"💳 Transactions: {len(transactions_df):,} records")

print(f"\n📈 Interaction Types:")
print(interactions_df['interaction_type'].value_counts())

print(f"\n🛒 Product Categories:")
print(products_df['category'].value_counts())

print(f"\n👥 Customer Segments:")
print(users_df['customer_segment'].value_counts())

print(f"\n💰 Revenue Summary:")
total_revenue = transactions_df['final_amount'].sum()
avg_order_value = transactions_df['final_amount'].mean()
print(f"Total Revenue: Rp {total_revenue:,.2f}")
print(f"Average Order Value: Rp {avg_order_value:,.2f}")

print("\n" + "="*50)
print("FILES GENERATED:")
print("="*50)
print("✓ user_profiles.csv")
print("✓ product_catalog.csv") 
print("✓ user_interactions.csv")
print("✓ transaction_history.csv")
print("\n🎉 All datasets generated successfully!")
print("📁 Files are ready for upload to AWS S3")