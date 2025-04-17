import config
from openai import OpenAI
from openai import AzureOpenAI
import json
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
import os
# client = OpenAI(api_key=config.api_key)


AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
MODEL = os.getenv("MODEL_NAME2")
API_VERSION = "2025-03-01-preview"


client = AzureOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    api_key=config.az_api_key,
    api_version=API_VERSION
)

# # This is to create a vector store in OpenAI
# vector_store = client.vector_stores.create(
#   name="BusinessRules"
# )
# print(vector_store)


# Ready the files for upload to OpenAI
file_paths = ["data_files/p2p-rules.txt"]
file_streams = [open(path, "rb") for path in file_paths]

vector_store = client.vector_stores.retrieve(
  vector_store_id=config.az_vector_store_id
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
    tools=[{
      "type": "file_search",
      "vector_store_ids": [vector_store.id],
      "max_num_results": 20
    }],
    input="What are business rules in procure to pay process?",
)

print(json.dumps(response, default=lambda o: o.__dict__, indent=4))
