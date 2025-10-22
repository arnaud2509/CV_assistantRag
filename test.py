import os
import json
from api_rag import InfomaniakEmbeddings, DATA_FILE, API_KEY, PRODUCT_ID, MODEL

# Instancie l'embedding
embedding_function = InfomaniakEmbeddings(
    api_key=API_KEY,
    product_id=PRODUCT_ID,
    model=MODEL
)

# Charge les documents JSON
with open(DATA_FILE, "r", encoding="utf-8") as f:
    data_raw = json.load(f)

texts = []
for item in data_raw:
    page_content = item.get('texte') if isinstance(item, dict) else (item if isinstance(item, str) else None)
    if page_content and page_content.strip():
        texts.append(page_content.strip())

print(f"{len(texts)} textes chargés pour test d'embeddings.\n")

# Test embeddings
for i, text in enumerate(texts):
    vec = embedding_function.embed_query(text)
    print(f"Texte #{i+1}: '{text[:50]}...'")
    print(f" - Longueur vecteur: {len(vec)}")
    print(f" - 5 premières valeurs: {vec[:5]}\n")
