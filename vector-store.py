from openai import AzureOpenAI
import json
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
import os

AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
MODEL = os.getenv("MODEL_NAME2")
API_VERSION = os.getenv("AZURE_API_VERSION")

token_provider = get_bearer_token_provider(
    DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
)

client = AzureOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    azure_ad_token_provider=token_provider,
    api_version=API_VERSION,
)

# # This is to create a vector store in OpenAI. Uncomment and run this if you want to create a new vector store.
# vector_store = client.vector_stores.create(
#   name="BusinessRules"
# )
# print(vector_store)


# Ready the files for upload to OpenAI. Run this if you want to upload files (i.e. business rules for anomaly detection) to the vector store.
file_paths = ["data_files/p2p-rules.txt"]
file_streams = [open(path, "rb") for path in file_paths]

vector_store = client.vector_stores.retrieve(
    vector_store_id=os.getenv("vector_store_id")
)
# Use the upload and poll SDK helper to upload the files, add them to the vector store,
# and poll the status of the file batch for completion.
file_batch = client.vector_stores.file_batches.upload_and_poll(
    vector_store_id=vector_store.id, files=file_streams
)

# # You can print the status and the file counts of the batch to see the result of this operation.
# print(file_batch.status)
# print(file_batch.file_counts)

# test the vector store
response = client.responses.create(
    model="gpt-4o",
    tools=[
        {
            "type": "file_search",
            "vector_store_ids": [vector_store.id],
            "max_num_results": 20,
        }
    ],
    input="What are business rules in procure to pay process?",
)

print(json.dumps(response, default=lambda o: o.__dict__, indent=4))
