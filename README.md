# Neo4j RAG PoC Using StackOverflow

## Importing StackOverflow into Neo4j

Based on the guide by https://neo4j.com/blog/import-10m-stack-overflow-questions/.
Note that this part is I/O intensive due to the CSV format, so it is recommended to store the data on a fast SSD.

### Obtaining Stack Overflow Data

#### Option 1: Download and Process Up-to-Date Data
1. Download up-to-date StackOverflow data dumps from the Internet Archive (https://archive.org/download/stackexchange).
2. Extract files (e.g., with WinRaR)
3. Generate the CSV files 
	```bash
	python create_csvs.py <path to extracted xml files>
	```
	Note that there are roughly 60 million posts, so this can take more than an hour.
4. Clean the data with
	```bash
	python cleaning.py <path to csv files generated in previous step>
	```
	This can take up to 20 minutes.
	
#### Option 2: Download Ready-to-Use Older Data
Download 2015 version from https://example-data.neo4j.org/files/stackoverflow/stackoverflow-2015-08-22.csv.tar
While this is much faster than Option 1, these posts are old and truncated, so we recommend using the up-todate posts from Option 1.

### Neo4j Database Setup

#### 1. Generate Embeddings
Although embeddings can be generated directly from Neo4j, but generating them externally offers more felxibility and potentially better performance. 
Generate embeddings to each post by running 
	```bash
	python generate_embeddings.py <path to folder of posts.cvs>
	```
Note that depending on your system this can take up to 10 hours.

#### 2. Prepare Your Directory Structure

Set up the following directory structure in your project folder and copy the relevant CSV files to ```db/import```:
```
project_folder/
├── docker-compose.yml
└── db/
    ├── data/
    ├── conf/
    ├── logs/
    ├── plugins/
    └── import/
        ├── posts_with_embeddings.csv
        ├── users.csv
        ├── tags.csv
        ├── posts_rel.csv
        ├── posts_answers.csv
        ├── tags_posts_rel.csv
        └── users_posts_rel.csv
```

#### 3. Build and Start Neo4j Container

1. Ensure that you have Docker and Docker Compose installed on your system.
2. Check and configure the settings in docker-compose.yml (e.g., port numbers). Don't forget to replace the password with a secure one before deploying in a production environment
3. Build and start the Neo4j container from the project folder in detached mode:
	```bash
	docker-compose up -d
	```

#### 4. Import StackOverflow Data into Neo4j
1. Once the container is up and running, run 
	```bash
	docker-compose stop
	```
to stop the container. This is necessary since we can't import into an active database.
2. Import the StackOverflow data into the Neo4j database with
```bash
docker compose run --rm neo4j-rag neo4j-admin database import full \
	--id-type string \
	--array-delimiter='|' \
	--nodes=Post=/var/lib/neo4j/import/posts_with_embeddings.csv \
	--nodes=User=/var/lib/neo4j/import/users.csv \
	--nodes=Tag=/var/lib/neo4j/import/tags.csv \
	--relationships=PARENT_OF=/var/lib/neo4j/import/posts_rel.csv \
	--relationships=ANSWER=/var/lib/neo4j/import/posts_answers.csv \
	--relationships=HAS_TAG=/var/lib/neo4j/import/tags_posts_rel.csv \
	--relationships=POSTED=/var/lib/neo4j/import/users_posts_rel.csv \
	--overwrite-destination=true \
	--verbose
```	
	Note that this can take up to 30 minutes
3. Restart the container with
	```bash
	docker-compose restart
	```

#### 5. Access Neo4j Database
Open a web browser and navigate to http://localhost:7888 to access your Neo4j database and inspect the schema.

If you are experiencing issues, check the Neo4j logs for detailed error messages:
```bash
docker-compose logs neo4j
```

#### 6. Generate Vector Index from the Embeddings
Execute the following Cypher command to create a vector index on the post embeddings
```Cypher
CREATE VECTOR INDEX post_embedding_index IF NOT EXISTS 
FOR (p:Post)
ON p.embedding
```

### Set up the LLM

#### Using llama.cpp
1) Install Python bindings for llama.cpp to access API through Langchain
```bash
pip install llama-cpp-python
```
2) Download an LLM model from https://huggingface.co/models. It should be in the .gguf format otherwise you will need to convert it (see https://github.com/ggerganov/llama.cpp for converters)


#### Using vLLM
Run a vLLM server in the background (requires Linux), below is an example
```
python -m vllm.entrypoints.openai.api_server \
	--model microsoft/Phi-3-mini-128k-instruct \
	--max-model-len 4096 \
	--dtype bfloat16 \
	--gpu-memory-utilization 0.90 \
	--port=8002 \
	--trust-remote-code \
	--disable-log-stats
```
Depending on your system you might want to change the parameters.


## Usage
If you are using llama.cpp run:
```bash
python hybridsearch.py llamacpp <path_to_your_model_file>
```

If you are using vLLM run:
```bash
python hybridsearch.py vllm <name_of_the_model>
``` 

Once the graph RAG model has started you can use a browser to access the gradio interface it generates at http://localhost:7860 .

Once a question is submitted the script will search through the posts in the database, comparing their embeddings with that of the question with two methods:
1) Vector search for the most similar posts using Langchain's similarity_search_with_score() method for Neo4j then returns the most similar ones
2) Vector search for the most similar posts using using a native Neo4J vector search, then query the graph for the accepted answers to the posts most similar to the question.

The LLM is given two prompts: one without context and one with the context provided by the posts returned by the search.

