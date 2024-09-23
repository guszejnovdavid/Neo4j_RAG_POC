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
	docker-compose run -rm neo4j neo4j-admin database import full \
	--id-type string --array-delimiter='|' \
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


Model from huggingface
llama.cpp for API to model  (https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md)
llama-cpp-python to access API in python  (pip install llama-cpp-python)






# To Create Vector Index on Particular Node Use following Command and embedding

```bash
   post_index1 = Neo4jVector.from_existing_graph(
    embeddings,
    url=NEO4J_URI,
    username=NEO4J_USER,
    password=NEO4J_PASSWORD,
    index_name='posts',
    node_label='Post',
    # text_node_property='body',
    text_node_properties=['body'],
    embedding_node_property='embedding',
)
# To use existing index:
```bash
post_index = Neo4jVector.from_existing_index(
    embeddings,
    url=NEO4J_URI,
    username=NEO4J_USER,
    password=NEO4J_PASSWORD,
    index_name='posts',
    text_node_property='body'
    
)