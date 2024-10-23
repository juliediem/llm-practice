# from dotenv import load_dotenv
# import os
# # Enable Logging
# import logging
# import sys
# from llama_index.readers.wikipedia import WikipediaReader
# from llama_index.core.node_parser import SimpleNodeParser
# from llama_index.vector_stores.deeplake import DeepLakeVectorStore
# from llama_index.core.storage import StorageContext
# from llama_index.core.storage.docstore import SimpleDocumentStore
# from llama_index.core import VectorStoreIndex, load_index_from_storage
#
# load_dotenv()
#
# # You can set the logging level to DEBUG for more verbose output,
# # or use level=logging.INFO for less detailed information.
# # Sets logging messages to sys.stdout
# # Logs all messages severity levels of INFO and higher
# # logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# # logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
#
# if not os.path.exists("./storage"):
#     loader = WikipediaReader()
#     documents = loader.load_data(pages=['Natural Language Processing', 'Artificial Intelligence'])
#     print(len(documents))
#
#     # Initialize parser
#     parser = SimpleNodeParser.from_defaults(chunk_size=512, chunk_overlap=20)
#
#     # Parse documents into nodes
#     nodes = parser.get_nodes_from_documents(documents)
#     print(len(nodes))
#
#     # This is the identifier of my organization, found in Active Loop
#     my_activeloop_org_id = os.getenv("ACTIVELOOP_ORG_ID")
#     # This is just the name of the dataset that you can freely assign
#     my_activeloop_dataset_name = "LlamaIndex_intro"
#     # This aggregates the above into the path
#     dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
#     # Create an index over the documents
#     # A Vector Store index generates embeddings during index construction to identify
#     # the top-k most similar nodes in response to a query Use this to connect to ActiveLoop platform
#     vector_store = DeepLakeVectorStore(dataset_path=dataset_path, overwrite=True)
#     # Setting the storage settings
#     storage_context = StorageContext.from_defaults(vector_store=vector_store)
#     # Store the Wikipedia information you grabbed earlier in the storage space you created
#     index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
#     index.storage_context.persist()
#
# else:
#     # If the index already exists, we'll just load it:
#     storage_context = StorageContext.from_defaults(persist_dir="./storage")
#     # storage_context = StorageContext.from_defaults(persist_dir="./storage")
#     index = load_index_from_storage(storage_context)
#
# # Query Engine
# # This enables you to ask the document questions
# query_engine = index.as_query_engine()
# # Input your question here
# response = query_engine.query("What does NLP stand for?")
# print(response.response)

from dotenv import load_dotenv
import os
import logging
import sys
from llama_index.readers.wikipedia import WikipediaReader
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.vector_stores.deeplake import DeepLakeVectorStore
from llama_index.core.storage import StorageContext
from llama_index.core import VectorStoreIndex, load_index_from_storage

load_dotenv()

# Configure logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


def create_new_index():
    logger.info("Creating new index")
    loader = WikipediaReader()
    documents = loader.load_data(pages=['Natural Language Processing', 'Artificial Intelligence'])
    logger.info(f"Number of documents: {len(documents)}")

    parser = SimpleNodeParser.from_defaults(chunk_size=1024, chunk_overlap=100)
    nodes = parser.get_nodes_from_documents(documents)
    logger.info(f"Number of nodes: {len(nodes)}")

    return documents


def save_to_local_and_deeplake(documents):
    # Save to local storage
    storage_context = StorageContext.from_defaults()
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    index.storage_context.persist(persist_dir="./storage")
    logger.info("Index saved to local storage")

    # Save to DeepLake
    my_activeloop_org_id = os.getenv("ACTIVELOOP_ORG_ID")
    my_activeloop_dataset_name = "LlamaIndex_intro"
    dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"

    vector_store = DeepLakeVectorStore(dataset_path=dataset_path, overwrite=True)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    index.storage_context.persist()
    logger.info("Index saved to DeepLake")

    return index


# Main execution
if os.path.exists("./storage"):
    try:
        logger.info("Attempting to load index from local storage")
        storage_context = StorageContext.from_defaults(persist_dir="./storage")
        index = load_index_from_storage(storage_context)
        logger.info("Successfully loaded index from local storage")
    except Exception as e:
        logger.error(f"Failed to load from local storage: {e}")
        logger.info("Falling back to creating new index")
        documents = create_new_index()
        index = save_to_local_and_deeplake(documents)
else:
    logger.info("Local storage not found")
    documents = create_new_index()
    index = save_to_local_and_deeplake(documents)

# Query Engine
query_engine = index.as_query_engine()
response = query_engine.query("What does NLP stand for?")
print(response.response)