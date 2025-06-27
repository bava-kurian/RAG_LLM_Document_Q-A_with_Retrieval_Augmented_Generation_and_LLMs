from pinecone import Pinecone
from config import Config

def test_pinecone():
    try:
        pc = Pinecone(api_key=Config.PINECONE_API_KEY)  # Use a test key or None
        print("✅ Pinecone imported successfully")
        print(f"Pinecone version: {pc.__version__}")
        return True
    except Exception as e:
        print(f"❌ Pinecone import failed: {e}")
        return False

if __name__ == "__main__":
    test_pinecone()