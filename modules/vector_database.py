import chromadb
client=chromadb.PersistentClient(path="./chroma_db")
collection=client.get_or_create_collection(name="research_document")
#print(collection.name,collection.count())

def store_embeddings(embedded_chunks):
    ids=[chunk["chunk_id"] for chunk in embedded_chunks]
    embeddings=[chunk["embeddings"] for chunk in embedded_chunks]
    documents=[chunk["content"] for chunk in embedded_chunks]

    meta=[{"document":chunk["document"],
           "pages":",".join(map(str,chunk["pages"]))} for chunk in embedded_chunks]
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=meta)
    #print("stored records",collection.count())

def search_documents(query_embedding,n_results=5):
    query_embeddings=[query_embedding]
    answer=collection.query(query_embeddings=query_embeddings,n_results=n_results,include=["documents","metadatas","distances"])
    #print(answer["documents"][0],answer["metadatas"][0],answer["distances"][0],answer["ids"][0])
    return answer