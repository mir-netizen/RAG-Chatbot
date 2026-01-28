import pdfplumber
import tiktoken
import os
from openai import OpenAI
import chromadb
import numpy as np
import re


client = OpenAI(api_key=(os.getenv("OPENAI_API_KEY")))

def text_extraction(pdf_path : str):

    pages = []

    print("Extracting Text from PDF\n")

    with pdfplumber.open(pdf_path) as pdf:
        for page_number , page in enumerate(pdf.pages , start = 1):
            text = page.extract_text()

            pages.append({
                "page_number":page_number,
                "page_content":text

            })


    return pages


def text_normalization(text : str):


    text = re.sub(r"-\s*\n\s*", "", text)

    text = re.sub(r"\n+", " ", text)

    text = re.sub(r"([.,!?;:])(?=[A-Za-z])", r"\1 ", text)

    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)

    text = re.sub(r"([\"“])(?=[A-Za-z])", r"\1 ", text)

    text = re.sub(r"(?<=[A-Za-z])([”\"])", r" \1", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenization(text : str):

    print("Converting text to tokens\n")

    encoding = tiktoken.encoding_for_model("gpt-4o-mini")

    tokenized_text = encoding.encode(text)

    return tokenized_text


def chunking(tokenized_text , chunk_size : int = 300 , chunk_overlap : int = 50):

    chunks = []
    start = 0

    encoding = tiktoken.encoding_for_model("gpt-4o-mini")

    print("Chunking\n")

    while(start < len(tokenized_text)):

        end = start + chunk_size
        chunk_tokens = tokenized_text[start:end]

        chunk_text = encoding.decode(chunk_tokens)


        chunks.append({
            "chunk_number":len(chunks),
            "chunk_content":chunk_text,
            "tokens":len(chunk_tokens)
        })

        start = start + (chunk_size - chunk_overlap)

    return chunks


def embedding_to_vectors(chunks):
    chroma = chromadb.Client()
    collection = chroma.create_collection("Vectors")

    print("Creating vectors and storing to VectorDB\n")

    for ci , chunk in enumerate(chunks):

        response = client.embeddings.create(
           model = "text-embedding-3-small",
           input = chunk["chunk_content"]
        )

        embedding = response.data[0].embedding

        collection.add(
            ids = [f"chunk_{ci}"],
            embeddings=[embedding],
            documents=[chunk["chunk_content"]]
        )


    return collection

def user_query_embeddings(query : str , collection , top_k : int = 3):

    print("Creating vectors for user queries\n")

    response = client.embeddings.create(
        model = "text-embedding-3-small",
        input = query
    )

    embedding = response.data[0].embedding

    results = collection.query(

        query_embeddings = [embedding],
        n_results = top_k
        

    )

    return results



def generate_answer(query:str , results):

    print("Generating Answers\n")

    answer = results['documents'][0]

    context = "\n\n".join(answer)


    prompt = f"""Based on the following context , please answer the question only as per that

    Context :
    {context}

    Question : {query}

    Answer : """

    response = client.chat.completions.create(
        model = "gpt-4o-mini",
        messages = [
            {"role":"system" , "content":"You are a helpful RAG Chatbot that answers questions only on the context provided to you"},
            {"role":"user","content":prompt}
        ]

    )

    return response.choices[0].message.content



pdf_path = "retn.pdf"

pages = text_extraction(pdf_path)

for page in pages:
    page["page_content"] = text_normalization(page["page_content"])


normalized_text = " ".join(page["page_content"] for page in pages)

# for page in pages:
#     print(f"Page_Numnber: ",{page["page_number"]})
#     print(page["page_content"])



tokenized_text = tokenization(normalized_text)

chunked_text = chunking(tokenized_text)

vectors = embedding_to_vectors(chunked_text)


while True:

    user_input = input("Enter your query: ")

    if user_input in ["exit","quit","q","stop"]:
        break


    results = user_query_embeddings(user_input,vectors , 3)

    response = generate_answer(user_input , results)
    print("\n")
    print(response)

