from sentence_transformers import SentenceTransformer
model=SentenceTransformer("all-MiniLM-L6-v2")

def generate_embeddings(chunks):
    if not chunks:
        return []
    contents=[chunk['content'] for chunk in chunks]
    embeddings=model.encode(contents,batch_size=32,show_progress_bar=True)

    for chunk,embedding in zip(chunks,embeddings):
        chunk['embeddings']=embedding
    return chunks
    
        