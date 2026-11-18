import pymupdf
from pathlib import Path


def extract_pdf(filename):
    pages = []
    path = Path(filename)

    if not path.is_file():
        raise ValueError("path is not a file")

    if not path.exists():
        raise FileNotFoundError(f"{filename} not found")

    if path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported")

    with pymupdf.open(str(path)) as doc:
        for i, page in enumerate(doc, start=1):
            content = page.get_text("text")
            pages.append({
                "document": path.name,
                "page": i,
                "content": content
            })

    return pages


def process_documents(files):
    all_docs = []

    for file in files:
        try:
            extracted = extract_pdf(file)
            all_docs.extend(extracted)

        except (FileNotFoundError, ValueError,
                PermissionError, pymupdf.FileDataError) as e:
            print(f"Error processing {file}: {e}")

    return all_docs


def display_documents(documents):
    print("Total extracted pages:", len(documents))

    for item in documents:
        print("\nDocument:", item["document"])
        print("Page:", item["page"])
        print("Content:", item["content"][:300])


def main():
    files = [
        "research3.pdf"
    ]

    documents = process_documents(files)
    #display_documents(documents)


if __name__ == "__main__":
    main()

