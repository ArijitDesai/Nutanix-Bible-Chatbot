import gradio as gr
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import pandas as pd
from pypdf import PdfReader
import anthropic


client = anthropic.Anthropic(
    api_key="YOUR API KEY",
)

# Load PDF and extract text
reader = PdfReader("nutanix_bible.pdf")
pdf_texts = [p.extract_text().strip() for p in reader.pages]
combined_text = ' '.join(pdf_texts)

# Split text into chunks
character_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
character_split_texts = character_splitter.split_text((combined_text))

token_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=100, tokens_per_chunk=256)
token_split_texts = []
for combined_text in character_split_texts:
    token_split_texts += token_splitter.split_text(combined_text)

embedding_function = SentenceTransformerEmbeddingFunction()

# Create ChromaDB collection
chroma_client = chromadb.Client()
chroma_collection = chroma_client.create_collection(name="nutanix_bible-1", embedding_function=embedding_function)

# Add documents to the collection
ids = [str(i) for i in range(len(token_split_texts))]
chroma_collection.add(ids=ids, documents=token_split_texts)

# Define LLM response function
def llm_response(query, model):
    results = chroma_collection.query(query_texts=[query], n_results=10)
    retrieved_documents = results['documents'][0]
    relevant_passage = "".join(retrieved_documents)
    return make_rag_prompt(model, relevant_passage, query)

# Define function to make RAG prompt
def make_rag_prompt(model, context, query):
    response = client.messages.create(
        system="You are a helpful Nutanix chat bot assistant. You will be shown data from a PDF which has the architecture and all the details of Nutanix product and you have to answer questions about the pdf",
        messages=[
            {"role": "user", "content": "Context: " + context + "\n\n Query: " + query},
        ],
        model=model,  # Choose the model you wish to use
        temperature=0,
        max_tokens=200
    )
    return response.content[0].text

# Model configuration
model = 'claude-3-5-sonnet-20240620'

def handle_query(query):
    response = llm_response(query, model)
    return {"Response": response}

# Create Gradio interface
gr.Interface(
    fn=handle_query,
    inputs=gr.Textbox(lines=2, placeholder="Enter your query here..."),
    outputs="json",
    title="Nutanix Bible Query Assistant",
    description="Ask any question about the Nutanix PDF document."
).launch(share=True)
