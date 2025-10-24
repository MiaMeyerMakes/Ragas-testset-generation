from langchain_community.document_loaders import DirectoryLoader
from dotenv import load_dotenv, find_dotenv
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from ragas.embeddings import OpenAIEmbeddings
from ragas.testset import TestsetGenerator
import openai
import os
load_dotenv(find_dotenv(usecwd=True), override=False)


# Load the sample documemys
path = "Sample_Docs_Markdown/"
loader = DirectoryLoader(path, glob="**/*.md")
docs = loader.load()

# Choose your LLM
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
openai_client = openai.OpenAI()
generator_embeddings = OpenAIEmbeddings(client=openai_client)

# Generate the testset
generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)
dataset = generator.generate_with_langchain_docs(docs, testset_size=10)
dataset.to_pandas()
print(dataset.head())
