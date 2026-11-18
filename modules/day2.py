from .pdf_extractor import process_documents


def clean_text(unclean):
    words = unclean.split()
    cleaned = " ".join(words)
    return cleaned


def prepare_document(pages):
    all_words = []
    word_pages = []

    if not pages:
        return [], [], ""

    document_name = pages[0]["document"]

    for page in pages:
        if page["document"] != document_name:
            raise ValueError("Pages belong to different documents")

        content = clean_text(page["content"])
        page_words = content.split()
        page_no = page["page"]

        all_words.extend(page_words)
        word_pages.extend([page_no] * len(page_words))
    if not all_words:
        return [],[],document_name

    return all_words, word_pages, document_name


def chunk_text(all_words, word_pages, chunk_size, overlap, document):

    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Invalid chunk size or overlap")

    if len(all_words) != len(word_pages):
        raise ValueError("Word and page metadata mismatch")

    chunks = []
    step = chunk_size - overlap

    for i in range(0, len(all_words), step):

        chunk = " ".join(all_words[i:i + chunk_size])

        chunk_pages = sorted(set(word_pages[i:i + chunk_size]))

        chunk_data = {
            "document": document,
            "pages": chunk_pages,
            "chunk_id": f"{document}_chunk_{len(chunks)+1}",
            "content": chunk
        }

        chunks.append(chunk_data)

        if i + chunk_size >= len(all_words):
            break

    return chunks


def process_into_chunks(pages, chunk_size=100, overlap=20):

    documents = {}

    for page in pages:
        document_name = page["document"]
        documents.setdefault(document_name, []).append(page)

    all_chunks = []

    for document_pages in documents.values():

        words, word_pages, document_name = prepare_document(document_pages)

        chunks = chunk_text(
            words,
            word_pages,
            chunk_size,
            overlap,
            document_name
        )

        all_chunks.extend(chunks)

    return all_chunks




if __name__ == "__main__":

    files = ['research2.pdf']

    pages = process_documents(files)

    chunks = process_into_chunks(pages, 100, 20)

    print("Total chunks:", len(chunks))
    print(chunks[0],"\n",chunks[-1])

    #for chunk in chunks:
     #   print("\nDocument:", chunk["document"])
      #  print("Chunk ID:", chunk["chunk_id"])
       # print("Pages:", chunk["pages"])
        #print("Content:", chunk["content"][:200])