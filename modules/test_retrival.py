from retrival import retrieve_information
from llm import generate_answer

query="what were the achievements of raja wodiyar?"
result=retrieve_information(query)
print(result)