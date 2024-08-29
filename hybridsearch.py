import os
import time
from dotenv import load_dotenv
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnablePassthrough
import gradio as gr
from neo4j import GraphDatabase
from langchain.llms import LlamaCpp
import concurrent.futures
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Set up Neo4j connection
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Set up Neo4j driver
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


# Set up the language model
model_path = "./openhermes-2.5-mistral-7b.Q4_K_M.gguf"
llm = LlamaCpp(model_path=model_path, temperature=0.5, max_tokens=1000)

# Set up embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def populate_embeddings(driver, embeddings, batch_size=100, max_workers=4):
    with driver.session() as session:
        # Check if there are any posts without embeddings
        count_result = session.run("MATCH (p:Post) WHERE p.embedding IS NULL RETURN count(p) AS count").single()
        posts_without_embeddings = count_result["count"]

        if posts_without_embeddings == 0:
            print("All posts already have embeddings. No action needed.")
            return

        print(f"Found {posts_without_embeddings} posts without embeddings. Populating now...")

        # Fetch all posts without embeddings
        result = session.run("MATCH (p:Post) WHERE p.embedding IS NULL RETURN p.body AS body, id(p) AS id")
        posts = list(result)

        # Function to process a batch of posts
        def process_batch(batch):
            local_embeddings = []
            for post in batch:
                embedding = embeddings.embed_query(post["body"])
                local_embeddings.append((post["id"], embedding))
            return local_embeddings

        # Process posts in batches
        with tqdm(total=len(posts)) as pbar:
            for i in range(0, len(posts), batch_size):
                batch = posts[i:i+batch_size]
                
                # Use ThreadPoolExecutor for parallel processing
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [executor.submit(process_batch, batch[j:j+batch_size//max_workers]) 
                               for j in range(0, len(batch), batch_size//max_workers)]
                    
                    batch_embeddings = []
                    for future in concurrent.futures.as_completed(futures):
                        batch_embeddings.extend(future.result())

                # Bulk update the database
                with driver.session() as update_session:
                    update_session.run("""
                    UNWIND $batch AS item
                    MATCH (p:Post) WHERE id(p) = item.id
                    SET p.embedding = item.embedding
                    """, batch=[{"id": id, "embedding": embedding} for id, embedding in batch_embeddings])

                pbar.update(len(batch))

        print(f"Embeddings populated successfully. {len(posts)} posts updated.")

# Usage
populate_embeddings(driver, embeddings)

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


def vector_similarity_search(question, top_k=3):
    start_time = time.time()
    results = post_index.similarity_search_with_score(question, k=top_k)
    end_time = time.time()
    search_time = end_time - start_time
    
    return [{"post": result[0].page_content, 
             "score": result[1],
             "tags": [],  # We don't have tag information in this search
             "author": "Unknown"  # We don't have author information in this search
            } for result in results], search_time

def graph_based_search(question, top_k=3):
    start_time = time.time()
    with driver.session() as session:
        query = """
        CALL db.index.vector.queryNodes('posts', $k, $question_embedding)
        YIELD node as post, score
        MATCH (post)-[:HAS_TAG]->(tag)
        MATCH (post)<-[:POSTED]-(user)
        RETURN post, score, collect(DISTINCT tag.name) as tags, user.name as author
        ORDER BY score DESC
        LIMIT $top_k
        """
        
        question_embedding = embeddings.embed_query(question)
        
        result = session.run(query, 
                             question_embedding=question_embedding, 
                             k=50,
                             top_k=top_k)
        
        results = [{"post": record["post"]["body"], 
                    "score": record["score"], 
                    "tags": record["tags"],
                    "author": record["author"]} for record in result]
    
    end_time = time.time()
    search_time = end_time - start_time
    
    return results, search_time

def generate_answer(question, documents):
    try:
        max_context_length = 100
        context = " ".join([doc["post"][:max_context_length] for doc in documents])
        
        template = """Given the following detailed context from Stack Overflow posts, 
        please provide a comprehensive and well-explained answer to the question. 
        Make sure to cover different aspects and provide examples if possible
        
        Context:
        {context}

        Question: {question}

        Answer:"""
        
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        
        chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | prompt
            | llm
        )
        
        response = chain.invoke({"context": context, "question": question})
        print(f"LLM Response: {response[:1000]}")  
        return response
        
    except Exception as e:
        print(f"Error generating answer: {e}")
        return f"Sorry, I couldn't generate an answer. Error: {str(e)}"
    
def hybrid_search(question, top_k=3):
    # Perform vector similarity search
    vector_results, vector_time = vector_similarity_search(question, top_k)
    
    # Perform graph-based search
    graph_results, graph_time = graph_based_search(question, top_k)
    
    # Combine and deduplicate results
    combined_results = {}
    for result in vector_results + graph_results:
        if result['post'] not in combined_results:
            combined_results[result['post']] = result
        else:
            # If the post is already in the results, update with graph information if available
            if 'tags' in result and result['tags']:
                combined_results[result['post']]['tags'] = result['tags']
            if 'author' in result and result['author'] != "Unknown":
                combined_results[result['post']]['author'] = result['author']
    
    # Sort combined results by score
    sorted_results = sorted(combined_results.values(), key=lambda x: x['score'], reverse=True)
    
    # Take top_k results
    final_results = sorted_results[:top_k]
    
    return final_results, vector_time, graph_time

def stackoverflow_qa(question):
    try:
        # Perform hybrid search
        hybrid_results, vector_time, graph_time = hybrid_search(question)
        
        # Generate answer using LLM with hybrid results
        answer = generate_answer(question, hybrid_results)
        
        # Prepare document display for hybrid results
        hybrid_doc_display = "\n\n".join([
            f"Document {i+1}:\n"
            f"Content: {doc['post'][:500]}...\n"
            f"Tags: {', '.join(doc.get('tags', []))}\n"
            f"Author: {doc.get('author', 'Unknown')}\n"
            f"Similarity Score: {doc['score']:.4f}"
            for i, doc in enumerate(hybrid_results)
        ])
        
        # Prepare timing information
        timing_info = f"Vector Search Time: {vector_time:.4f} seconds\nGraph Search Time: {graph_time:.4f} seconds"
        
        return answer, hybrid_doc_display, timing_info
    
    except Exception as e:
        error_message = f"An error occurred: {type(e).__name__}, {str(e)}"
        return error_message, "", ""

# Define the Gradio interface
gr_interface = gr.Interface(
    fn=stackoverflow_qa,
    inputs=[
        gr.Textbox(lines=2, placeholder="Ask a question about programming...")
    ],
    outputs=[
        gr.Textbox(label="Generated Answer"),
        gr.Markdown(label="Hybrid Search Results"),
        gr.Markdown(label="Search Timing Information")
    ],
    title="Stack Overflow RAG Application with Hybrid Search",
    description="Ask questions about programming topics and get answers using a Retrieval-Augmented Generation model with hybrid vector and graph-based search."
)    

# Launch the Gradio interface
if __name__ == "__main__":
    gr_interface.launch()