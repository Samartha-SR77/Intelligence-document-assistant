from .retrival import retrieve_information
from .llm import generate_answer


def ask_question(question):
    results = retrieve_information(question)

    context_parts = []
    sources = set()

    for chunk in results:
        metadata = chunk["metadata"]

        document = metadata.get("document", "Unknown")
        pages = metadata.get("pages", "Unknown")

        context_parts.append(
            f"Source: {document}\n"
            f"Pages: {pages}\n"
            f"Content: {chunk['content']}"
        )

        sources.add((document, pages))

    context = "\n\n".join(context_parts)

    answer = generate_answer(question, context)

    if answer.strip().upper() == "NULL":
        return (
            "The uploaded documents don't contain enough "
            "information to answer this question."
        )

    source_details = "\n".join(
        f"- {document}, Page(s): {pages}"
        for document, pages in sorted(sources)
    )

    return f"Answer:\n{answer}\n\nSources:\n{source_details}"



    
if __name__ == "__main__":

    question = input("Ask your question: ")

    response = ask_question(question)

    print("\nAnswer:")
    print(response)