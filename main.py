from langchain.document_loaders.base import Document
from langchain.indexes import VectorstoreIndexCreator
from langchain.utilities import ApifyWrapper
from langchain.document_loaders import ApifyDatasetLoader

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
apify_api_token = os.getenv("APIFY_API_TOKEN")

# Create an instance of the ApifyWrapper class.
apify = ApifyWrapper()

print("Loading dataset...")

# Run an Apify actor to scrape the data you need.
loader = apify.call_actor(
    actor_id="perci/apify-blog-scraper",
    run_input={"filter": "all"},
    dataset_mapping_function=lambda item: Document(
        page_content=item["Text"] or "", metadata={"source": item["Link"]}
    ),
)

# Fetch data from an existing Apify dataset.
# loader = ApifyDatasetLoader(
#     dataset_id="you datasetID",
#     dataset_mapping_function=lambda item: Document(
#         page_content=item["Text"] or "", metadata={"source": item["Link"]}
#     ),
# )

index = VectorstoreIndexCreator().from_loaders([loader])


query = "Is Apify running a video tutorial contest? If so, what are the rules and the rewards for the winners?"
result = index.query_with_sources(query)

# Print the query, answer and sources to the console.
print(f"Query: {query}")
print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")