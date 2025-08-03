import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List
from urllib.parse import urljoin, urlparse
import time
import hashlib

# === CONFIG ===
START_URL = "https://myrealestate.in"
MAX_PAGES = 2500
REQUEST_DELAY = 0.5  # seconds between requests
MAX_CHARS = 5000

# === STEP 1: SMART CRAWLER ===
def crawl_site(start_url, max_pages=2500):
    visited = set()
    to_visit = [start_url]
    docs = []
    seen_hashes = set()

    while to_visit and len(visited) < max_pages:
        url = to_visit.pop(0)
        if url in visited or not url.startswith(start_url):
            continue

        try:
            print(f"[{len(visited)+1}/{max_pages}] Crawling: {url}")
            res = requests.get(url, timeout=10)
            soup = BeautifulSoup(res.text, "html.parser")
            text = soup.get_text(separator=" ", strip=True)[:MAX_CHARS]

            content_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
            if content_hash in seen_hashes:
                print(f"⏭ Skipped duplicate: {url}")
                visited.add(url)
                continue

            docs.append({
                "url": url,
                "content": text
            })
            visited.add(url)
            seen_hashes.add(content_hash)

            # Find internal links
            for link in soup.find_all("a", href=True):
                full_url = urljoin(url, link["href"])
                parsed = urlparse(full_url)
                if parsed.netloc == urlparse(start_url).netloc and full_url not in visited:
                    to_visit.append(full_url)

            time.sleep(REQUEST_DELAY)

        except Exception as e:
            print(f"❌ Failed: {url} - {e}")

    return docs

# === STEP 2: EMBED + STORE ===
model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.Client()
collection = client.get_or_create_collection(name="myrealestate")

def embed_and_store(docs):
    print(f"\n🔢 Embedding {len(docs)} documents in batch...")
    contents = [doc["content"] for doc in docs]
    embeddings = model.encode(contents, batch_size=32)

    for i, (doc, emb) in enumerate(zip(docs, embeddings)):
        collection.add(
            documents=[doc["content"]],
            metadatas=[{"source": doc["url"], "text": doc["content"]}],
            embeddings=[emb.tolist()],
            ids=[f"doc_{i}"]
        )
        print(f"✅ Stored: {doc['url']}")

# === STEP 3: SEARCH API ===
class Query(BaseModel):
    query: str
    top_k: int = 5

app = FastAPI()

@app.post("/query")
def search(query: Query):
    results = collection.query(
        query_embeddings=[model.encode(query.query).tolist()],
        n_results=query.top_k
    )
    return {
        "query": query.query,
        "matches": [
            {"score": s, "metadata": m}
            for s, m in zip(results["distances"][0], results["metadatas"][0])
        ]
    }

# === MAIN ===
if __name__ == "__main__":
    docs = crawl_site(START_URL, max_pages=MAX_PAGES)
    embed_and_store(docs)
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)