from .pdf_extractor import process_documents
from .day2 import process_into_chunks
from .embedding_generator import generate_embeddings
from .vector_database import store_embeddings


files = ["research3.pdf","research.pdf"]

pages = process_documents(files)
print(len(pages))

chunks = process_into_chunks(pages, 100, 20)
embbe=generate_embeddings(chunks)
print(len(chunks),len(embbe))

store_embeddings(embbe)



