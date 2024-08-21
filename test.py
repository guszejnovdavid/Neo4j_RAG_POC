import os
from dotenv import load_dotenv
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import LlamaCpp
from langchain_core.runnables import RunnableSequence, RunnablePassthrough
import gradio as gr

# Load environment variables
load_dotenv()

# Set up Neo4j connection
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Set up the language model
# model_name = "MBZUAI/LaMini-T5-738M"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map = "auto", torch_dtype = torch.float32)
# # Configure the text-generation pipeline
# device = 0 if torch.cuda.is_available() else -1
# text_generation_pipeline = pipeline(
#   "text2text-generation",
#   model = model,
#   tokenizer = tokenizer,
#   max_length = 256,
#   do_sample = True,
#   temperature = 0.3,
#   top_p = 0.95
# )

# Create a HuggingFacePipeline language model
# llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
# Load the Mistral model
# Set up the language model
model_path = "./openhermes-2.5-mistral-7b.Q4_K_M.gguf"
llm = LlamaCpp(model_path=model_path, temperature=0.7, max_tokens=2000)

# llm = OpenAI(temperature=0.7)

# Set up embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Use existing vector index for Post nodes
post_index = Neo4jVector(
    embeddings,
    url=NEO4J_URI,
    username=NEO4J_USER,
    password=NEO4J_PASSWORD,
    index_name='posts',
    node_label='Post',
    text_node_property='body',
    embedding_node_property='embedding',
    
)
# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=post_index.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    verbose=True
)
def generate_answer(question, documents):
    try:
        max_context_length = 100  # Adjust this value as needed
        context = " ".join([doc.page_content[:max_context_length] for doc in documents])
        # context = "\n".join([doc.page_content for doc in documents])
        template = """Given the following context from Stack Overflow posts, 
        please provide a detailed and comprehensive answer to the question.
        
        Context:
        {context}

        Question: {question}

        Answer:"""
        
        print("Prompt:", template)
        
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        
        chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | prompt
            | llm
        )
        
        response = chain.invoke({"context": context, "question": question})
        print("LLM Response:", response)
        
        return response
        
    except Exception as e:
        print(f"Error generating answer: {e}")
        return f"Sorry, I couldn't generate an answer. Error: {str(e)}"
    
def stackoverflow_qa(question):
    try:
        # Retrieve relevant documents
        docs = post_index.similarity_search(question, k=3)
        
        # Generate answer using LLM
        answer = generate_answer(question, docs)
        
        # Prepare document display
        doc_display = "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])
        
        return answer, doc_display
    
    except Exception as e:
        error_message = f"An error occurred: {type(e).__name__}, {str(e)}"
        return error_message, ""    

# Define the Gradio interface
gr_interface = gr.Interface(
    fn=stackoverflow_qa,
    inputs=[
        gr.Textbox(lines=2, placeholder="Ask a question about programming...")
    ],
    outputs=[
        gr.Textbox(label="Generated Answer"),
        gr.Markdown(label="Retrieved Documents")
    ],
    title="Stack Overflow RAG Application",
    description="Ask questions about programming topics and get answers using a Retrieval-Augmented Generation model."
)


# Launch the Gradio interface
if __name__ == "__main__":
    gr_interface.launch()    

