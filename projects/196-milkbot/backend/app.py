#!/usr/bin/env python3
"""
Simple Milk Products RAG Chatbot using FastAPI and OpenAI API.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
from dotenv import load_dotenv
from datetime import datetime
import json
import re
from pathlib import Path
from openai import OpenAI

# ChromaDB imports for RAG
import chromadb
from chromadb.config import Settings

app = Flask(__name__)
CORS(app)

# Load environment variables
load_dotenv()
GPT_API_KEY = os.getenv("OPENAI_API_KEY")

if not GPT_API_KEY:
    print("Error: OPENAI_API_KEY is not set in the .env file.")
    exit(1)

# Initialize ChromaDB for RAG
def get_chromadb_client():
    """Get ChromaDB client for milk products search"""
    try:
        chroma_path = Path("../data/chroma_db/chroma_db")
        if not chroma_path.exists():
            print("ChromaDB not found. Please run create_db.py first.")
            return None
        
        client = chromadb.PersistentClient(path=str(chroma_path))
        collection = client.get_collection("milk_products")
        return collection
    except Exception as e:
        print(f"Error initializing ChromaDB: {e}")
        return None

# Store conversation history and shopping carts
conversations = {}
shopping_carts = {}

def date():
    now = datetime.now()
    return '[' + now.strftime("%Y-%m-%d %H:%M") + ']'

def get_or_create_cart(session_id):
    """Get existing cart or create new one for session"""
    if session_id not in shopping_carts:
        shopping_carts[session_id] = {
            'items': [],
            'total': 0.0,
            'item_count': 0
        }
    return shopping_carts[session_id]

def add_to_cart(session_id, product_info, quantity=1):
    """Add product to shopping cart"""
    cart = get_or_create_cart(session_id)
    
    # Check if product already exists in cart
    existing_item = None
    for item in cart['items']:
        if item['stock_code'] == product_info.get('stock_code'):
            existing_item = item
            break
    
    if existing_item:
        # Update quantity of existing item
        existing_item['quantity'] += quantity
        existing_item['total_price'] = float(existing_item['current_price']) * existing_item['quantity']
    else:
        # Add new item to cart
        current_price = float(product_info.get('current_price', 0)) if product_info.get('current_price') != 'N/A' else 0
        new_item = {
            'stock_code': product_info.get('stock_code', ''),
            'name': product_info.get('name', 'N/A'),
            'current_price': product_info.get('current_price', 'N/A'),
            'unit_price': product_info.get('unit_price', 'N/A'),
            'image_url': product_info.get('image_url', 'N/A'),
            'product_url': product_info.get('product_url', 'N/A'),
            'quantity': quantity,
            'total_price': current_price * quantity
        }
        cart['items'].append(new_item)
    
    # Recalculate cart totals
    cart['total'] = sum(item['total_price'] for item in cart['items'])
    cart['item_count'] = sum(item['quantity'] for item in cart['items'])
    
    return cart

def remove_from_cart(session_id, stock_code, quantity=None):
    """Remove product from shopping cart"""
    cart = get_or_create_cart(session_id)
    
    for i, item in enumerate(cart['items']):
        if item['stock_code'] == stock_code:
            if quantity is None or item['quantity'] <= quantity:
                # Remove entire item
                removed_item = cart['items'].pop(i)
                print(f"Removed {removed_item['name']} from cart")
            else:
                # Reduce quantity
                item['quantity'] -= quantity
                item['total_price'] = float(item['current_price']) * item['quantity']
                print(f"Reduced quantity of {item['name']} by {quantity}")
            
            # Recalculate cart totals
            cart['total'] = sum(item['total_price'] for item in cart['items'])
            cart['item_count'] = sum(item['quantity'] for item in cart['items'])
            break
    
    return cart

def checkout_cart(session_id):
    """Process checkout and clear cart"""
    cart = get_or_create_cart(session_id)
    
    if not cart['items']:
        return {'success': False, 'message': 'Cart is empty'}
    
    # Here you would typically integrate with payment processing
    # For now, we'll just simulate a successful checkout
    
    checkout_summary = {
        'items': cart['items'].copy(),
        'total': cart['total'],
        'item_count': cart['item_count'],
        'checkout_time': datetime.now().isoformat(),
        'order_id': f"ORDER-{session_id}-{int(datetime.now().timestamp())}"
    }
    
    # Clear the cart after successful checkout
    cart['items'] = []
    cart['total'] = 0.0
    cart['item_count'] = 0
    
    return {'success': True, 'checkout_summary': checkout_summary}

def reset_cart(session_id):
    """Reset shopping cart to empty"""
    if session_id in shopping_carts:
        shopping_carts[session_id] = {
            'items': [],
            'total': 0.0,
            'item_count': 0
        }
    return get_or_create_cart(session_id)

def find_relevant_products(query: str, limit: int = 5):
    """Find relevant milk products using ChromaDB semantic search"""
    collection = get_chromadb_client()
    if not collection:
        print("ChromaDB collection not available for RAG search")
        return []
    
    try:
        # Clean and prepare the query for better search
        clean_query = query.strip()
        if len(clean_query) < 3:
            # If query is too short, try to expand it with common milk terms
            clean_query = f"milk {clean_query}"
        
        print(f"Searching ChromaDB for query: '{clean_query}'")
        
        # Perform semantic search with distance information
        results = collection.query(
            query_texts=[clean_query],
            n_results=limit,
            include=["metadatas", "distances"]
        )
        
        # Convert results to product format
        relevant_products = []
        if results['metadatas'] and results['metadatas'][0]:
            for i, metadata in enumerate(results['metadatas'][0]):
                distance = results['distances'][0][i] if results['distances'] and results['distances'][0] else None
                
                # Calculate relevance score (1.0 = perfect match, 0.0 = poor match)
                relevance_score = 1.0 - (distance if distance else 0.0)
                
                product = {
                    'name': metadata.get('name', 'N/A'),
                    'description': metadata.get('description', 'N/A'),
                    'current_price': metadata.get('current_price', 'N/A'),
                    'was_price': metadata.get('was_price', 'N/A'),
                    'savings': metadata.get('savings', 'N/A'),
                    'special_offer': metadata.get('special_offer', 'N/A'),
                    'unit_price': metadata.get('unit_price', 'N/A'),
                    'image_url': metadata.get('image_url', 'N/A'),
                    'product_url': metadata.get('product_url', 'N/A'),
                    'stock_code': metadata.get('stock_code', 'N/A'),
                    'relevance_score': relevance_score
                }
                relevant_products.append(product)
        
        # Sort by relevance score (highest first)
        relevant_products.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        print(f"Found {len(relevant_products)} relevant products")
        if relevant_products:
            print(f"Best match: {relevant_products[0].get('name')} (score: {relevant_products[0].get('relevance_score', 0):.3f})")
        
        return relevant_products
        
    except Exception as e:
        print(f"Error in ChromaDB search: {e}")
        return []

def create_context_from_products(products):
    """Create context string from relevant milk products"""
    if not products:
        return "No relevant milk products found."
    
    context_parts = []
    for product in products:
        relevance_score = product.get('relevance_score', 'N/A')
        price_info = f"${product.get('current_price', 'N/A')}"
        if product.get('was_price') and product.get('was_price') != 'N/A':
            price_info += f" (was ${product.get('was_price')})"
        
        context_parts.append(f"""
Product: {product.get('name', 'N/A')}
Price: {price_info}
Description: {product.get('description', 'N/A')}
Unit Price: {product.get('unit_price', 'N/A')}
Relevance Score: {relevance_score:.3f} (higher = more relevant)
""".strip())
    
    return "\n\n---\n\n".join(context_parts)

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        message = data.get('message', '')
        session_id = data.get('session_id', 'default')
        
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        
        print(f"\n=== New Chat Request ===")
        print(f"Session: {session_id}")
        print(f"User message: {message}")
        
        # Initialize or get conversation history
        if session_id not in conversations:
            conversations[session_id] = []
        
        # Add user message to history
        conversations[session_id].append({"role": "user", "content": message})
        
        # Find relevant milk products using RAG based on the current user message
        print(f"Searching for relevant milk products...")
        relevant_products = find_relevant_products(message)
        context = create_context_from_products(relevant_products)
        
        print(f"Found {len(relevant_products)} relevant products")
        print(f"Context length: {len(context)} characters")
        
        # Get current cart status for context
        cart = get_or_create_cart(session_id)
        cart_context = ""
        if cart['items']:
            cart_context = f"\n\nCurrent Cart Status:\nItems: {cart['item_count']}\nTotal: ${cart['total']:.2f}"
        
        # Create the base system prompt for milk products
        base_system_prompt = """You are a helpful AI assistant specializing in milk products and dairy alternatives. 
You have access to a database of milk products including:
- Regular dairy milk (full cream, lite, trim, organic)
- Plant-based alternatives (oat, almond, soy, coconut)
- Flavored milks (chocolate, strawberry, vanilla, coffee)
- Specialized products (lactose-free, high protein, A2, UHT)

You can also help with shopping cart operations:
- Add products to cart: "add [product name] to cart" or "add [product name]"
- Remove products: "remove [product name] from cart" or "remove [product name]"
- Checkout: "checkout" or "complete order"
- Reset cart: "reset cart" or "clear cart"
- View cart: "show cart" or "what's in my cart"

When relevant to the user's question, you can mention multiple milk products that might be suitable. 
Be helpful, informative, and focus on providing useful information about milk products and alternatives."""

        # Create messages for OpenAI
        messages = [
            {"role": "system", "content": base_system_prompt}
        ]
        
        # Add conversation history (last 5 messages to keep context manageable)
        # Only include user and assistant messages, skip any temporary system messages
        for msg in conversations[session_id][-5:]:
            if msg["role"] in ["user", "assistant"]:
                messages.append(msg)
        
        # Add current user message
        messages.append({"role": "user", "content": message})
        
        # NOW add the product context as a system message at the end
        product_system_message = f"Use this milk product information to answer the user's question:{context}{cart_context}"
        
        messages.append({"role": "system", "content": product_system_message})
        
        print(f"Total messages for OpenAI: {len(messages)}")
        print(f"Product context added as final system message")
        
        # Get response from OpenAI
        client = OpenAI(api_key=GPT_API_KEY)
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        
        ai_response = response.choices[0].message.content
        
        print(f"AI response received: {ai_response[:100]}...")
        
        # Update conversation history - the product lookup message is never stored
        # Just add the AI response
        conversations[session_id].append({"role": "assistant", "content": ai_response})
        
        # Keep only last 10 messages to prevent context from getting too long
        conversations[session_id] = conversations[session_id][-10:]
        
        # Log the conversation
        log_file = 'log.chat'
        with open(log_file, 'a') as log:
            log.write(f"{date()} 🙂: {message}\n")
            log.write(f"{date()} 🤖: {ai_response}\n\n")
        
        return jsonify({
            'response': ai_response,
            'session_id': session_id,
            'relevant_products': relevant_products,
            'context_used': context[:500] + "..." if len(context) > 500 else context,
            'cart_status': cart
        })
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/cart/add', methods=['POST'])
def add_to_cart_api():
    """Add product to shopping cart"""
    try:
        data = request.get_json()
        session_id = data.get('session_id', 'default')
        product_info = data.get('product', {})
        quantity = data.get('quantity', 1)
        
        if not product_info:
            return jsonify({'error': 'Product information is required'}), 400
        
        cart = add_to_cart(session_id, product_info, quantity)
        
        return jsonify({
            'success': True,
            'message': f"Added {product_info.get('name', 'product')} to cart",
            'cart': cart
        })
        
    except Exception as e:
        print(f"Error adding to cart: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/cart/remove', methods=['POST'])
def remove_from_cart_api():
    """Remove product from shopping cart"""
    try:
        data = request.get_json()
        session_id = data.get('session_id', 'default')
        stock_code = data.get('stock_code')
        quantity = data.get('quantity')  # Optional, removes all if not specified
        
        if not stock_code:
            return jsonify({'error': 'Stock code is required'}), 400
        
        cart = remove_from_cart(session_id, stock_code, quantity)
        
        return jsonify({
            'success': True,
            'message': f"Product removed from cart",
            'cart': cart
        })
        
    except Exception as e:
        print(f"Error removing from cart: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/cart/checkout', methods=['POST'])
def checkout_cart_api():
    """Checkout shopping cart"""
    try:
        data = request.get_json()
        session_id = data.get('session_id', 'default')
        
        result = checkout_cart(session_id)
        
        if result['success']:
            return jsonify({
                'success': True,
                'message': 'Checkout completed successfully!',
                'checkout_summary': result['checkout_summary']
            })
        else:
            return jsonify({
                'success': False,
                'message': result['message']
            })
        
    except Exception as e:
        print(f"Error during checkout: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/cart/reset', methods=['POST'])
def reset_cart_api():
    """Reset shopping cart"""
    try:
        data = request.get_json()
        session_id = data.get('session_id', 'default')
        
        cart = reset_cart(session_id)
        
        return jsonify({
            'success': True,
            'message': 'Cart reset successfully',
            'cart': cart
        })
        
    except Exception as e:
        print(f"Error resetting cart: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/cart/status', methods=['GET'])
def get_cart_status():
    """Get current cart status"""
    try:
        session_id = request.args.get('session_id', 'default')
        cart = get_or_create_cart(session_id)
        
        return jsonify({
            'success': True,
            'cart': cart
        })
        
    except Exception as e:
        print(f"Error getting cart status: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/reset', methods=['POST'])
def reset():
    try:
        data = request.get_json()
        session_id = data.get('session_id', 'default')
        
        if session_id in conversations:
            del conversations[session_id]
        
        if session_id in shopping_carts:
            del shopping_carts[session_id]
        
        return jsonify({'message': 'Conversation and cart reset successfully'})
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    print("Starting Flask server with Milk Products RAG and Shopping Cart...")
    print("Make sure you have OPENAI_API_KEY set in your .env file")
    print("Make sure you have run create_db.py to create the ChromaDB")
    app.run(debug=True, host='0.0.0.0', port=5008) 