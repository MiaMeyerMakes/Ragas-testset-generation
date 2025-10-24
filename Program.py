import os
import logging
import openai
import glob
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import OpenAIEmbeddings
from ragas.testset import TestsetGenerator

# --- 1. Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def main():
    """Main function to generate the testset."""
    try:
        # --- 2. Load Environment Variables ---
        logging.info("Loading environment variables...")
        load_dotenv(find_dotenv(usecwd=True), override=False)
        if not os.environ.get("OPENAI_API_KEY"):
            logging.error("OPENAI_API_KEY not found. Please check your .env file.")
            return
        logging.info("Environment variables loaded successfully.")

        # --- 3. Load Documents Manually (The Reliable Method) ---
        logging.info("Starting to load documents...")
        path = "Sample_Docs_Markdown/"
        markdown_files = glob.glob(os.path.join(path, "**/*.md"), recursive=True)

        if not markdown_files:
            logging.warning("No markdown files found to load.")
            return

        logging.info(f"Found {len(markdown_files)} files to load.")

        all_docs = []
        for file_path in markdown_files:
            logging.info(f"--> Loading file: {file_path}")
            loader = UnstructuredMarkdownLoader(file_path, mode="single")
            all_docs.extend(loader.load())

        logging.info(f"All {len(all_docs)} document sections loaded successfully.")

        # --- 4. Initialize Models ---
        logging.info("Initializing LLM and embedding models...")
        generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
        openai_client = openai.OpenAI()
        generator_embeddings = OpenAIEmbeddings(client=openai_client)
        logging.info("Models initialized.")

        # --- 5. Generate the Testset ---
        logging.info("Starting testset generation... This may take several minutes.")
        generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)
        dataset = generator.generate_with_langchain_docs(all_docs, testset_size=10)
        logging.info("Testset generation complete!")

        # --- 6. Display Results ---
        logging.info("Converting dataset to pandas DataFrame and displaying the first 5 rows.")
        df = dataset.to_pandas()
        print("\n--- Generated Testset (First 5 Rows) ---")
        print(df.head())
        print("-----------------------------------------")

    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    main()