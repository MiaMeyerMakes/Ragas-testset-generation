---
license: apache-2.0
---

# RAG Testset Generation with Ragas

This project is a learning exercise to understand and implement testset generation for a RAG (Retrieval-Augmented Generation) application using the `ragas` library.

The implementation follows the official Ragas documentation tutorial: [RAG Testset Generation](https://docs.ragas.io/en/stable/getstarted/rag_testset_generation/).

## Setup

To get started, install the required Python packages:

```sh
pip install -r requirements.txt
```
## Process summary

### Loading and preparation

The manual loading process mentioned above is used. 11 Markdown files from the `Sample_Docs_Markdown` directory are loaded.

### Document Analysis

Ragas' `TestsetGenerator` performs a deep analysis of the documents to understand them. The documents are then split into smaller chunks based on their headings. Thereafter, the `_Extractor` tools are used to identify main topics and themes that are present across the documents. The `NERExtractor` method is used to find specific keywords by performing Named Entity Recognition.

### Persona and Scenario Generation

Ragas uses the previous analysis to create different user personas who might be interested in the documents. It then creates scenarios where these personas would ask questions (e.g., "A new employee wants to know how to be a good ally"). This ensures the generated questions are realistic and varied.

### Generating the questions and answers

Finally, the script uses an OpenAI model to write questions with ground-truth answers. 

```text                                   user_input                                 reference_contexts                                          reference                       synthesizer_name
0  How can Zoom meetings be used effectively to p...  [Skills and Behaviors of allies\n\nTo be an ef...  Zoom meetings can be used effectively to promo...  single_hop_specific_query_synthesizer
1  What is the purpose of the ALLG in promoting a...  [Tips on being an ally Identifying your power ...  The ALLG, or Ally Lab Learning Group, is an in...  single_hop_specific_query_synthesizer
2  How can the Diversity, Inclusion & Belonging S...  [What it means to be an ally\n\nTake on the st...  The Diversity, Inclusion & Belonging Sharing P...  single_hop_specific_query_synthesizer
3  How does GitLab use CultureAmp to support dive...  [title: "Building an Inclusive Remote Culture"...  The DIB team at GitLab runs an annual survey v...  single_hop_specific_query_synthesizer
4  How do DIB roundtables facilitate empowerment ...  [<1-hop>\n\ntitle: "Roundtables" description: ...  DIB roundtables facilitate empowerment and inc...   multi_hop_abstract_query_synthesizer
```
You can see from the `synthesizer_name` column that the final output uses different strategies, from simple, direct questions (`single_hop_specific_query`) to more complex questions using multiple documents (`multi_hop_abstract_query`).

## Tuning & controls

1. ### Tune the size of the testset
   1. This can be done with the `testset_size=10` argument in `generate_with_langchain_docs()`
   2. The recommended number of Q&A sets is typically **between 50 and 150**.
2. ### Controlling the complexity of the questions
   1. You have control over the mix of question types.
      2. The main types of questions are:
         1. `simple`
         2. `reasoning`
         3. `multi_hop`
         4. `conditional`
      3. You can pass `question_distributions` in to the `TestsetGenerator`:
          ```    
         question_distributions = {
                  "simple": 0.4,      # 40% simple questions
                  "reasoning": 0.3,   # 30% reasoning questions
                  "multi_hop": 0.3    # 30% multi-hop questions
              }
         ```

## Things to note

### Possible problems & solutions
Initially there was a persistent silent failure within the `langchain_community.document_loaders.DirectoryLoader` class when running on a Windows machine. It could be established that the problem was a deep-seated incompatibility between how `DirectoryLoader` handles file processing in parallel to the specific environment.

Therefore, since `DirectoryLoader` remained unstable, it was replaced by a manual loading process using `UnstructuredMarkdownLoader`.

### Privacy & security

As RAGAS itself is a library that runs locally on your machine, it won't send your data anywhere. The `TestsetGenerator` uses an external service: an OpenAI API call.

#### OpenAI's API Data Policy
- OpenAI does NOT use data submitted via their API to train or improve their models.
- Your data is retained for a limited period (e.g., 30 days) for abuse and misuse monitoring, after which it is deleted.
- Data is encrypted in transit (using TLS) and at rest.

Something to note, though: _"While this is a strong privacy promise, the fact remains that your confidential data is being processed on servers you do not control. For highly sensitive information, many organizations consider this an unacceptable risk."_

Other alternatives are to use Microsoft Azure OpenAI Service or to run local, open-source models.
