#!/usr/bin/env python3
"""
Script to create ChromaDB vector database from Simpsons episodes JSON data.
This provides semantic search capabilities for episode content.
"""

import json
import os
from pathlib import Path

# ChromaDB imports
import chromadb
from chromadb.config import Settings

# SQLite function removed - using only ChromaDB

def create_chromadb_database(episodes):
    """Create ChromaDB vector database from episodes data"""
    
    data_dir = Path(__file__).parent
    chroma_path = data_dir / "chroma_db"
    
    print(f"Creating ChromaDB vector database: {chroma_path}")
    
    # Load .env from parent directory
    from dotenv import load_dotenv
    load_dotenv(data_dir.parent / ".env")
    
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
    collection_name = "simpsons_episodes"
    
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
        metadata={"description": "Simpsons episodes for semantic search"}
    )
    
    # Prepare documents for vectorization
    documents = []
    metadatas = []
    ids = []
    
    for i, episode in enumerate(episodes):
        # Create episode content for embeddings
        title = episode.get('episode_title', 'N/A')
        description = episode.get('description', 'N/A')
        
        # Skip episodes without meaningful content
        if title == 'N/A' and description == 'N/A':
            continue
        
        # Create focused content for embeddings
        episode_content = f"{title}\n\n{description}" if description != 'N/A' else title
        
        # Create comprehensive metadata
        metadata = {
            'season': episode.get('season', 'N/A'),
            'episode_number': episode.get('episode_number_in_season', 'N/A'),
            'title': title,
            'air_date': episode.get('air_date', 'N/A'),
            'description': description,
            'imdb_rating': episode.get('imdb_rating', 'N/A'),
            'vote_count': episode.get('vote_count', 'N/A'),
            'url': episode.get('episode_url', 'N/A')
        }
        
        documents.append(episode_content)
        metadatas.append(metadata)
        ids.append(f"episode_{i}")
    
    # Add documents to collection
    print(f"Adding {len(documents)} episodes to ChromaDB...")
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    
    print(f"ChromaDB vector database created successfully with {len(documents)} episodes")
    
    # Test the collection
    print("Testing ChromaDB search...")
    results = collection.query(
        query_texts=["Homer Simpson"],
        n_results=3
    )
    
    if results['documents']:
        print("✅ ChromaDB search test successful")
        print(f"Found {len(results['documents'][0])} results for 'Homer Simpson'")
    else:
        print("❌ ChromaDB search test failed")
    
    return True

def create_database():
    """Create ChromaDB vector database from JSON data"""
    
    # Paths
    data_dir = Path(__file__).parent
    json_file = data_dir / "simpsons_episodes.json"
    
    if not json_file.exists():
        print(f"JSON file not found: {json_file}")
        return
    
    # Load JSON data
    print(f"Loading episodes from {json_file}...")
    with open(json_file, 'r', encoding='utf-8') as f:
        episodes = json.load(f)
    
    print(f"Loaded {len(episodes)} episodes")
    
    # Create ChromaDB database
    chroma_success = create_chromadb_database(episodes)
    
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
            collection = chroma_client.get_collection("simpsons_episodes")
            count = collection.count()
            print(f"✅ ChromaDB: {count} episodes")
        except Exception as e:
            print(f"❌ ChromaDB error: {e}")
    else:
        print("❌ ChromaDB not found")

if __name__ == "__main__":
    print("Creating Simpsons episodes ChromaDB database...")
    create_database()
    test_databases()
