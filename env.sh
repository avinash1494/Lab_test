export VECTOR_DB_STORAGE_PATH="/home/avinash_dataneuron_ai/jupy/rag-env/rag-product-code/vector-stores/Langchain"
export VECTOR_DB_STORAGE_PATH_LLAMMA_INDEX="/home/avinash_dataneuron_ai/jupy/rag-env/rag-product-code/vector-stores/Llamaindex"
export AUTH_SERVER_URL="https://alp-aw-auth-server.azurewebsites.net/user/me"
export MONGO_DB_CONNECTION_STRING="mongodb+srv://dn-admin:uB5SlJo9JS9DIQ4X@rag-test-db.8hrgm9a.mongodb.net/"
export NETAPP_CVS_MOUNT_PATH="/mnt/netapp-poc-data/data"
export NETAPP_CVS_STORE_PATH="/mnt/netapp-poc-data/vectorstores"
export NETAPP_CVS_VECTORDB_MOUNT_PATH="/mnt/netapp-poc-data/vectorstores"
export GOOGLE_CREDINTIALS_JSON="prj-prod-bucket-access.json"
export  SQL_SAVE_DB_PATH="/home/avinash_dataneuron_ai/jupy/rag-env/rag-product-code/sql_store/"
export  NO_OF_PROMPTS_PER_BATCH_IN_PROMPT_GEN=20
export NO_OF_PAGES_TO_GENERATE_PROMTPS_FOR_A_FILE=10 (


docker pull qdrant/qdrant
docker run -d -p 6333:6333 -p 6334:6334 \
    --ulimit nofile=10000:10000 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
