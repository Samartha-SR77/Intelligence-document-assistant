from .embedding_generator import model
from .vector_database import search_documents


def format_results(results):
    formatted_results = []

    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for chunk_id, content, metadata, distance in zip(
        ids, documents, metadatas, distances
    ):
        chunk_result = {
            "id": chunk_id,
            "content": content,
            "metadata": metadata,
            "distance": distance
        }

        formatted_results.append(chunk_result)

    return formatted_results

def retrieve_information(query):
    query_embedding=model.encode(query)
    result=search_documents(query_embedding,5)
    return format_results(result)



