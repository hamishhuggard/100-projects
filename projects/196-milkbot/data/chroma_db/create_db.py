#!/usr/bin/env python3
"""
Script to create ChromaDB vector database from milk products CSV data.
This provides semantic search capabilities for milk product content.
"""

import csv
import os
from pathlib import Path

# ChromaDB imports
import chromadb
from chromadb.config import Settings

def create_chromadb_database(products):
    """Create ChromaDB vector database from milk products data"""
    
    data_dir = Path(__file__).parent
    chroma_path = data_dir / "chroma_db"
    
    print(f"Creating ChromaDB vector database: {chroma_path}")
    
    # Load .env from parent directory
    from dotenv import load_dotenv
    load_dotenv(data_dir.parent.parent / ".env")
    
    # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found. Please set it in your environment.")
        print("   Make sure you have a .env file in the project root with:")
        print("   OPENAI_API_KEY=your_key_here")
        return False
    
    # Initialize OpenAI embeddings using ChromaDB's native function
    try:
        from chromadb.utils import embedding_functions
        embeddings = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name="text-embedding-3-small"
        )
        print("✅ Using OpenAI text-embedding-3-small model")
    except Exception as e:
        print(f"❌ Error initializing OpenAI embeddings: {e}")
        return False
    
    # Initialize ChromaDB client
    chroma_client = chromadb.PersistentClient(path=str(chroma_path))
    
    # Create or get collection
    collection_name = "milk_products"
    
    # Delete existing collection if it exists
    try:
        chroma_client.delete_collection(collection_name)
        print("Deleted existing collection")
    except:
        pass
    
    # Create new collection with embedding function
    collection = chroma_client.create_collection(
        name=collection_name,
        embedding_function=embeddings,
        metadata={"description": "Milk products for semantic search"}
    )
    
    # Prepare documents for vectorization
    documents = []
    metadatas = []
    ids = []
    
    for i, product in enumerate(products):
        # Get product name and description for embeddings
        name = product.get('name', 'N/A')
        description = product.get('description', 'N/A')
        
        # Skip products without meaningful content
        if name == 'N/A' and description == 'N/A':
            continue
        
        # Create focused content for embeddings - combine name and description
        if description != 'N/A':
            product_content = f"{name}\n\n{description}"
        else:
            product_content = name
        
        # Create comprehensive metadata
        metadata = {
            'name': name,
            'description': description,
            'current_price': product.get('current_price', 'N/A'),
            'was_price': product.get('was_price', 'N/A'),
            'savings': product.get('savings', 'N/A'),
            'special_offer': product.get('special_offer', 'N/A'),
            'unit_price': product.get('unit_price', 'N/A'),
            'image_url': product.get('image_url', 'N/A'),
            'product_url': product.get('product_url', 'N/A'),
            'stock_code': product.get('stock_code', 'N/A')
        }
        
        documents.append(product_content)
        metadatas.append(metadata)
        ids.append(f"product_{i}")
    
    # Add documents to collection
    print(f"Adding {len(documents)} milk products to ChromaDB...")
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    
    print(f"ChromaDB vector database created successfully with {len(documents)} products")
    
    # Test the collection
    print("Testing ChromaDB search...")
    results = collection.query(
        query_texts=["organic milk"],
        n_results=3
    )
    
    if results['documents']:
        print("✅ ChromaDB search test successful")
        print(f"Found {len(results['documents'][0])} results for 'organic milk'")
    else:
        print("❌ ChromaDB search test failed")
    
    return True

def create_database():
    """Create ChromaDB vector database from CSV data"""
    
    # Paths
    data_dir = Path(__file__).parent
    csv_file = data_dir.parent / "scrape-descriptions" / "milk_products.csv"
    
    if not csv_file.exists():
        print(f"CSV file not found: {csv_file}")
        return
    
    # Load CSV data
    print(f"Loading milk products from {csv_file}...")
    products = []
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            products.append(row)
    
    print(f"Loaded {len(products)} milk products")
    
    # Create ChromaDB database
    chroma_success = create_chromadb_database(products)
    
    if chroma_success:
        print("\n🎉 ChromaDB database created successfully!")
    else:
        print("\n❌ ChromaDB database creation failed.")
        print("   The app will fall back to text-based search.")

def test_databases():
    """Test ChromaDB database"""
    
    data_dir = Path(__file__).parent
    chroma_path = data_dir / "chroma_db"
    
    print("\n--- Testing ChromaDB ---")
    
    # Test ChromaDB
    if chroma_path.exists():
        try:
            chroma_client = chromadb.PersistentClient(path=str(chroma_path))
            collection = chroma_client.get_collection("milk_products")
            count = collection.count()
            print(f"✅ ChromaDB: {count} milk products")
        except Exception as e:
            print(f"❌ ChromaDB error: {e}")
    else:
        print("❌ ChromaDB not found")

if __name__ == "__main__":
    print("Creating milk products ChromaDB database...")
    create_database()
    test_databases()
