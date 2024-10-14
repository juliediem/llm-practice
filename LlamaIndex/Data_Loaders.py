from dotenv import load_dotenv
import os
# Enable Logging
import logging
import sys
from llama_index.readers.wikipedia import WikipediaReader
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.vector_stores.deeplake import DeepLakeVectorStore
from llama_index.core.storage import StorageContext
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core import VectorStoreIndex, load_index_from_storage

load_dotenv()

# You can set the logging level to DEBUG for more verbose output,
# or use level=logging.INFO for less detailed information.
# Sets logging messages to sys.stdout
# Logs all messages severity levels of INFO and higher
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

if not os.path.exists("./storage"):
    loader = WikipediaReader()
    documents = loader.load_data(pages=['Natural Language Processing','Artificial Intelligence'])
    print(len(documents))

    # Initialize parser
    parser = SimpleNodeParser.from_defaults(chunk_size=512, chunk_overlap=20)

    # Parse documents into nodes
    nodes = parser.get_nodes_from_documents(documents)
    print(len(nodes))

    # This is the identifier of my organization, found in Active Loop
    my_activeloop_org_id = "jdiem"
    # This is just the name of the dataset that you can freely assign
    my_activeloop_dataset_name = "LlamaIndex_intro"
    # This aggregates the above into the path
    dataset_path =f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
    # Create an index over the documents
    vector_store = DeepLakeVectorStore(dataset_path=dataset_path,overwrite=True)
    # Setting the storage settings
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    # Store the Wikipedia information you grabbed earlier in the storage space you created
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    index.storage_context.persist()

else:
    # If the index already exists, we'll just load it:
    storage_context = StorageContext.from_defaults(
        docstore=SimpleDocumentStore.from_persist_dir(persist_dir="./storage"),
        vector_store=SimpleVectorStore.from_persist_dir(
            persist_dir="./storage"
        ),
        index_store=SimpleIndexStore.from_persist_dir(persist_dir="./storage"),
    )
    # storage_context = StorageContext.from_defaults(persist_dir="./storage")
    index = load_index_from_storage(storage_context)

# Query Engine
# This enables you to ask the document questions
query_engine = index.as_query_engine()
# Input your question here
response = query_engine.query("What does NLP stand for?")
print(response.response)