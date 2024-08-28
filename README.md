Neo4j Data Import Guide
This guide explains how to set up Neo4j using Docker Compose and import data using CSV files.
Prerequisites

Docker and Docker Compose installed on your system
CSV files prepared for import:

posts.csv
users.csv
tags.csv
posts_rel.csv
posts_answers.csv
tags_posts_rel.csv
users_posts_rel.csv



Steps

Create a docker-compose.yml file:
Create a file named docker-compose.yml in your project directory with the following content:
yamlCopyversion: "3.8"
services:
  neo4j:
    image: neo4j:5.19.0
    ports:
      - 7888:7474
      - 7999:7687
    deploy:
      resources:
        limits:
          memory: 6G
        reservations:
          memory: 4G
    restart: unless-stopped
    environment:
      - NEO4J_AUTH=neo4j/password
      - NEO4J_PLUGINS=["apoc", "apoc-extended"]
      - NEO4J_apoc_export_file_enabled=true
      - NEO4J_apoc_import_file_enabled=true
      - NEO4J_apoc_import_file_use__neo4j__config=true
      - NEO4J_dbms_memory_pagecache_size=2G
      - NEO4J_dbms_memory_heap_initial__size=1G
      - NEO4J_dbms_memory_heap_max__size=3G
    volumes:
      - ./db/data:/data
      - ./db/conf:/conf
      - ./db/logs:/logs
      - ./db/plugins:/plugins
      - ./db/import:/var/lib/neo4j/import
Note: Adjust the NEO4J_AUTH value to set a secure password.
Prepare your directory structure:
Ensure you have the following directory structure in your project folder:
Copyproject_folder/
├── docker-compose.yml
└── db/
    ├── data/
    ├── conf/
    ├── logs/
    ├── plugins/
    └── import/
        ├── posts.csv
        ├── users.csv
        ├── tags.csv
        ├── posts_rel.csv
        ├── posts_answers.csv
        ├── tags_posts_rel.csv
        └── users_posts_rel.csv

Start the Neo4j container:
Run the following command in your project directory:
bashCopydocker-compose up -d
This will start the Neo4j container in detached mode.
Run the import command:
Once the container is up and running, execute the import command:
bashCopydocker-compose exec neo4j neo4j-admin database import full \
  --id-type string \
  --nodes=Post=/var/lib/neo4j/import/posts.csv \
  --nodes=User=/var/lib/neo4j/import/users.csv \
  --nodes=Tag=/var/lib/neo4j/import/tags.csv \
  --relationships=PARENT_OF=/var/lib/neo4j/import/posts_rel.csv \
  --relationships=ANSWER=/var/lib/neo4j/import/posts_answers.csv \
  --relationships=HAS_TAG=/var/lib/neo4j/import/tags_posts_rel.csv \
  --relationships=POSTED=/var/lib/neo4j/import/users_posts_rel.csv \
  --overwrite-destination=true \
  --verbose

Restart the Neo4j container:
bashCopydocker-compose restart neo4j

Access Neo4j:
Open a web browser and navigate to http://localhost:7888. Log in using the username neo4j and the password you set in the docker-compose.yml file.

Configuration Details

The Neo4j browser is accessible on port 7888, and the Bolt protocol on port 7999.
Memory is limited to 6GB, with 4GB reserved.
APOC plugins are enabled for extended functionality.
File import and export are enabled via APOC.
Memory settings:

Page cache: 2GB
Initial heap: 1GB
Max heap: 3GB



Troubleshooting

If you encounter permission issues, ensure that the Neo4j user has read access to the CSV files.
Verify that your CSV files are correctly formatted according to Neo4j's requirements.
Check the Neo4j logs for detailed error messages:
bashCopydocker-compose logs neo4j


Additional Notes

The --id-type string option specifies that node IDs are strings. Adjust if your data uses different ID types.
The --overwrite-destination=true flag allows overwriting an existing database. Use with caution in production environments.
The --verbose flag provides detailed output during the import process, which can be helpful for debugging.

Remember to replace the password in the docker-compose.yml file with a secure one before deploying in a production environment.