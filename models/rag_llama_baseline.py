import os
import re
from abc import ABC
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import ray
import torch
import vllm
from blingfire import text_to_sentences_and_offsets
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from models.main_content_extractor import MainContentExtractor
from models.bm25_retriever import Bm25Retriever
from models.faiss_retriever import FaissRetriever
from models.rerank_model import reRankLLM
from models.utils import trim_predictions_to_max_token_length

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoModel,
    pipeline,
    AutoModelForSequenceClassification
)
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.schema.embeddings import Embeddings


base = "/root/autodl-tmp/codes/meta-comphrehensive-rag-benchmark-starter-kit/"
RERANKER_MODEL_PATH = base + "models/pretrained_model/bge-rerank"
EMBEDDING_MODEL_PATH = base + "models/pretrained_model/gte-large-en"
LLAMA3_MODEL_PATH = base + "models/pretrained_model/llama3-8b"
#### CONFIG PARAMETERS ---

# Define the number of context sentences to consider for generating an answer.
NUM_CONTEXT_SENTENCES = 20
# Set the maximum length for each context sentence (in characters).
MAX_CONTEXT_SENTENCE_LENGTH = 1000
# Set the maximum context references length (in characters).
MAX_CONTEXT_REFERENCES_LENGTH = 4000

# Batch size you wish the evaluators will use to call the `batch_generate_answer` function
AICROWD_SUBMISSION_BATCH_SIZE = 4 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.

# VLLM Parameters 
VLLM_TENSOR_PARALLEL_SIZE = 1 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.
VLLM_GPU_MEMORY_UTILIZATION = 0.75 #0.85  # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.

# Sentence Transformer Parameters
SENTENTENCE_TRANSFORMER_BATCH_SIZE = 128 # TUNE THIS VARIABLE depending on the size of your embedding model and GPU mem available

class ChunkExtractor:
    
    @ray.remote
    def _extract_chunks(self, interaction_id, html_source):
        """
        Extracts and returns chunks from given HTML source.

        Note: This function is for demonstration purposes only.
        We are treating an independent sentence as a chunk here,
        but you could choose to chunk your text more cleverly than this.

        Parameters:
            interaction_id (str): Interaction ID that this HTML source belongs to.
            html_source (str): HTML content from which to extract text.

        Returns:
            Tuple[str, List[str]]: A tuple containing the interaction ID and a list of sentences extracted from the HTML content.
        """
        # Parse the HTML content using BeautifulSoup
        chunks = []
        extracted_markdown = MainContentExtractor.extract(html_source, include_links = False, output_format="markdown")
        if(extracted_markdown == None):
            soup = BeautifulSoup(html_source, "lxml")
            text = soup.get_text(" ", strip=True)  # Use space as a separator, strip whitespaces
            if not text:
            # Return a list with empty string when no text is extracted
                return interaction_id, [""]
            if(len(text)< 512):
                chunks.append(text)
            else:
                text_split = text.split("\n\n")
                chunks.extend(self.SlidingWindow(text_split))
        else:
            try:
                sentences = re.split(r'\#+',extracted_markdown)
                for txt in sentences:
                   if(len(txt) < 10):
                       continue
                   if(len(txt) > 1000):
                       text_split = txt.split("\n\n")
                       chunks.extend(self.SlidingWindow(text_split))
                   else:
                       sentence = txt[:MAX_CONTEXT_SENTENCE_LENGTH]
                       chunks.append(sentence)
            except:
                if(len(extracted_markdown)< 512):
                    chunks.append(extracted_markdown)
                else:
                    text_split = extracted_markdown.split("\n\n")
                    chunks.extend(self.SlidingWindow(text_split))
        return interaction_id, chunks

    def extract_chunks(self, batch_interaction_ids, batch_search_results):
        """
        Extracts chunks from given batch search results using parallel processing with Ray.

        Parameters:
            batch_interaction_ids (List[str]): List of interaction IDs.
            batch_search_results (List[List[Dict]]): List of search results batches, each containing HTML text.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing an array of chunks and an array of corresponding interaction IDs.
        """
        # Setup parallel chunk extraction using ray remote
        ray_response_refs = [
            self._extract_chunks.remote(
                self,
                interaction_id=batch_interaction_ids[idx],
                html_source=html_text["page_result"]
            )
            for idx, search_results in enumerate(batch_search_results)
            for html_text in search_results
        ]

        # Wait until all sentence extractions are complete
        # and collect chunks for every interaction_id separately
        chunk_dictionary = defaultdict(list)

        for response_ref in ray_response_refs:
            interaction_id, _chunks = ray.get(response_ref)  # Blocking call until parallel execution is complete
            chunk_dictionary[interaction_id].extend(_chunks)

        # Flatten chunks and keep a map of corresponding interaction_ids
        chunks, chunk_interaction_ids = self._flatten_chunks(chunk_dictionary)

        return chunks, chunk_interaction_ids

    def _flatten_chunks(self, chunk_dictionary):
        """
        Flattens the chunk dictionary into separate lists for chunks and their corresponding interaction IDs.

        Parameters:
            chunk_dictionary (defaultdict): Dictionary with interaction IDs as keys and lists of chunks as values.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing an array of chunks and an array of corresponding interaction IDs.
        """
        chunks = []
        chunk_interaction_ids = []

        for interaction_id, _chunks in chunk_dictionary.items():
            # De-duplicate chunks within the scope of an interaction ID
            unique_chunks = list(set(_chunks))
            chunks.extend(unique_chunks)
            chunk_interaction_ids.extend([interaction_id] * len(unique_chunks))

        # Convert to numpy arrays for convenient slicing/masking operations later
        chunks = np.array(chunks)
        chunk_interaction_ids = np.array(chunk_interaction_ids)

        return chunks, chunk_interaction_ids

    def SlidingWindow(self, sentences, max_len = 512):
        cur = ""
        idx = 0
        sentence_tmp = []
        ans = []
        for sentence in sentences:
            if(len(sentence) < max_len):
                sentence_tmp.append(sentence)
            else:
                sentence_split = sentence.split(". ")
                sentence_tmp.extend(sentence_split)
    
        while(idx < len(sentence_tmp)):
        
            sentence = sentence_tmp[idx]
            idx = idx + 1
            if(len(cur + sentence) > max_len):
                ans.append((cur + " " + sentence)[:MAX_CONTEXT_SENTENCE_LENGTH])
                cur = sentence
            else:
                cur = cur + sentence
        return ans

class RAGModel:
    """
    An example RAGModel for the KDDCup 2024 Meta CRAG Challenge
    which includes all the key components of a RAG lifecycle.
    """
    def __init__(self):
        self.initialize_models()
        self.chunk_extractor = ChunkExtractor()
        self.bm25 = Bm25Retriever()
        self.reranker = reRankLLM(RERANKER_MODEL_PATH)
        self.faiss = FaissRetriever(EMBEDDING_MODEL_PATH)
        self.emb_top_k = 7  # 4
        self.bm25_top_k = 4  # 2
        self.rerank_top_k = 6
        self.max_ctx_sentence_length = 200

        self.overlap_length = 200
        self.window_size = 500

        self.sim_threshold = 0.75

    def initialize_models(self):
        # Initialize Meta Llama 3 - 8B Instruct Model
        self.model_name = LLAMA3_MODEL_PATH

        if not os.path.exists(self.model_name):
            raise Exception(
                f"""
            The evaluators expect the model weights to be checked into the repository,
            but we could not find the model weights at {self.model_name}
            
            Please follow the instructions in the docs below to download and check in the model weights.
            
            https://gitlab.aicrowd.com/aicrowd/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024/meta-comphrehensive-rag-benchmark-starter-kit/-/blob/master/docs/dataset.md
            """
            )

        # Initialize the model with vllm
        self.llm = vllm.LLM(
            self.model_name,
            tensor_parallel_size=VLLM_TENSOR_PARALLEL_SIZE, 
            gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION, 
            trust_remote_code=True,
            dtype="half", # note: bfloat16 is not supported on nvidia-T4 GPUs
            enforce_eager=True
        )
        self.tokenizer = self.llm.get_tokenizer()

    def post_process(self, answer):
        if "i don't know" in answer.lower():
            return "i don't know"
        else:
            return answer

    def get_batch_size(self) -> int:
        """
        Determines the batch size that is used by the evaluator when calling the `batch_generate_answer` function.
        
        The evaluation timeouts linearly scale with the batch size. 
            i.e.: time out for the `batch_generate_answer` call = batch_size * per_sample_timeout 
        

        Returns:
            int: The batch size, an integer between 1 and 16. It can be dynamic
                 across different batch_generate_answer calls, or stay a static value.
        """
        self.batch_size = AICROWD_SUBMISSION_BATCH_SIZE  
        return self.batch_size


    def batch_generate_answer(self, batch: Dict[str, Any]) -> List[str]:
        """
        Generates answers for a batch of queries using associated (pre-cached) search results and query times.

        Parameters:
            batch (Dict[str, Any]): A dictionary containing a batch of input queries with the following keys:
                - 'interaction_id;  (List[str]): List of interaction_ids for the associated queries
                - 'query' (List[str]): List of user queries.
                - 'search_results' (List[List[Dict]]): List of search result lists, each corresponding
                                                      to a query. Please refer to the following link for
                                                      more details about the individual search objects:
                                                      https://gitlab.aicrowd.com/aicrowd/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024/meta-comphrehensive-rag-benchmark-starter-kit/-/blob/master/docs/dataset.md#search-results-detail
                - 'query_time' (List[str]): List of timestamps (represented as a string), each corresponding to when a query was made.

        Returns:
            List[str]: A list of plain text responses for each query in the batch. Each response is limited to 75 tokens.
            If the generated response exceeds 75 tokens, it will be truncated to fit within this limit.

        Notes:
        - If the correct answer is uncertain, it's preferable to respond with "I don't know" to avoid
          the penalty for hallucination.
        - Response Time: Ensure that your model processes and responds to each query within 30 seconds.
          Failing to adhere to this time constraint **will** result in a timeout during evaluation.
        """
        batch_interaction_ids = batch["interaction_id"]
        queries = batch["query"]
        batch_search_results = batch["search_results"]
        query_times = batch["query_time"]

        # Chunk all search results using ChunkExtractor
        chunks, chunk_interaction_ids = self.chunk_extractor.extract_chunks(
            batch_interaction_ids, batch_search_results
        )

        # Retrieve top matches for the whole batch
        batch_retrieval_results = []
        for _idx, interaction_id in enumerate(batch_interaction_ids):
            query = queries[_idx]
            query_time = query_times[_idx]

            relevant_chunks_mask = chunk_interaction_ids == interaction_id
            retrieval_results = []
            # Filter out the said chunks and corresponding embeddings
            relevant_chunks = chunks[relevant_chunks_mask]
            
            self.faiss.GetvectorStore(relevant_chunks)
            retrieval_emb_ans = self.faiss.GetTopK(query, k=self.emb_top_k, score_threshold = self.sim_threshold)
            retrieval_content_ans = [doc.page_content for doc, score in retrieval_emb_ans]
            retrieval_results.extend(retrieval_content_ans)
            self.bm25.init_bm25(relevant_chunks)
            bm25_docs = self.bm25.GetBM25TopK(query, self.bm25_top_k)
            bm25_text = [doc.page_content for doc in bm25_docs]
            retrieval_results.extend(bm25_text)
            rerank_res = self.reranker.predict(query, list(set(retrieval_results)))[:self.rerank_top_k]
            batch_retrieval_results.append(rerank_res)
            
        # Prepare formatted prompts from the LLM        
        # formatted_prompts = self.merge_format_prompts(queries, query_times, batch_retrieval_results)
        # formatted_prompts = self.format_prompts(queries, query_times, batch_retrieval_results)
        formatted_prompts = self.five_shot_template(queries, query_times, batch_retrieval_results)
        # Generate responses via vllm
        responses = self.llm.generate(
            formatted_prompts,
            vllm.SamplingParams(
                n=2,  # Number of output sequences to return for each prompt.
                top_p=0.6,  # Float that controls the cumulative probability of the top tokens to consider.
                temperature=0.1,  # Randomness of the sampling
                skip_special_tokens=True,  # Whether to skip special tokens in the output.
                max_tokens=300,  # Maximum number of tokens to generate per output sequence.
                
                # Note: We are using 50 max new tokens instead of 75,
                # because the 75 max token limit for the competition is checked using the Llama2 tokenizer.
                # Llama3 instead uses a different tokenizer with a larger vocabulary
                # This allows the Llama3 tokenizer to represent the same content more efficiently, 
                # while using fewer tokens.
            ),
            use_tqdm=False # you might consider setting this to True during local development
        )

        # Aggregate answers into List[str]
        answers = []
        for response in responses:
            trimmed_answer = self.post_process(trim_predictions_to_max_token_length(response.outputs[0].text))
            print("answer: " + trimmed_answer + "\n")
            answers.append(trimmed_answer)
        
        return answers
    def five_shot_template(self, queries, query_times, batch_retrieval_results=[]):
        """
        Formats queries, corresponding query_times and retrieval results using the chat_template of the model.
            
        Parameters:
        - queries (List[str]): A list of queries to be formatted into prompts.
        - query_times (List[str]): A list of query_time strings corresponding to each query.
        - batch_retrieval_results (List[str])
        """        
        system_prompt = "You are given a quesition and references which may or may not help answer the question." \
                      "Your goal is to answer the question in as few words as possible and still accurate." \
                      "If the information in the references does not include an answer, you must say 'I don't know'. " \
                      "You must not say any incorrect information. Especially for the time, location, name, and number, you confirm them repeatedly, otherwise you must say 'I don't know'." \
                      "If you cannot be 100% certain that you are right, please answer that I do not know." \
                      "There is no need to explain the reasoning behind your answers.\n"
        
        formatted_prompts = []

        for _idx, query in enumerate(queries):
            query_time = query_times[_idx]
            retrieval_results = batch_retrieval_results[_idx]

            user_message = ""
            references = ""
            
            if len(retrieval_results) > 0:
                references += "# References \n"
                # Format the top sentences as references in the model's prompt template.
                for _snippet_idx, snippet in enumerate(retrieval_results):
                    snippet = snippet.replace("\n", " ")
                    references += f"- {snippet.strip()}\n"
            
            references = references[:MAX_CONTEXT_REFERENCES_LENGTH]
           # Limit the length of references to fit the model's input size.
            user_message += f"{references}\n------\n\n"
            user_message += f"Using only the references listed above, answer the following question: \n"
            user_message += f"Current Time: {query_time}\n"
            user_message += f"Question: {query}\n"
            
            formatted_prompts.append(
                self.tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        return formatted_prompts
    def merge_format_prompts(self, queries, query_times, batch_retrieval_results=[]):
        """
        Formats queries, corresponding query_times and retrieval results using the chat_template of the model.
            
        Parameters:
        - queries (List[str]): A list of queries to be formatted into prompts.
        - query_times (List[str]): A list of query_time strings corresponding to each query.
        - batch_retrieval_results (List[str])
        """        
        system_prompt = "You are provided with a question and various references. Your task is to answer the question succinctly, using the fewest words possible. If the references do not contain the necessary information to answer the question, respond with 'I don't know'. You must not say any incorrect information, Especially for the times, location, names, number, and what happened today. If you are not 100% sure please answer 'I don't know'. There is no need to explain the reasoning behind your answers."
        formatted_prompts = []

        for _idx, query in enumerate(queries):
            query_time = query_times[_idx]
            retrieval_results = batch_retrieval_results[_idx]

            user_message = ""
            references = ""
            
            if len(retrieval_results) > 0:
                references += "# References \n"
                # Format the top sentences as references in the model's prompt template.
                for _snippet_idx, snippet in enumerate(retrieval_results):
                    snippet = snippet.replace("\n"," ")
                    references += f"- {snippet.strip()}\n"
            
            references = references[:MAX_CONTEXT_REFERENCES_LENGTH]
            # Limit the length of references to fit the model's input size.

            user_message += f"{references}\n------\n\n"
            user_message += f"Using only the references listed above, answer the following question: \n"
            user_message += f"Current Time: {query_time}\n"
            user_message += f"Question: {query}\n"
            
            formatted_prompts.append(
                self.tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

        return formatted_prompts
    def format_prompts(self, queries, query_times, batch_retrieval_results=[]):
        """
        Formats queries, corresponding query_times and retrieval results using the chat_template of the model.
            
        Parameters:
        - queries (List[str]): A list of queries to be formatted into prompts.
        - query_times (List[str]): A list of query_time strings corresponding to each query.
        - batch_retrieval_results (List[str])
        """        
        system_prompt = "You are provided with a question and various references. Your task is to answer the question succinctly, using the fewest words possible. If the references do not contain the necessary information to answer the question, respond with 'I don't know'. There is no need to explain the reasoning behind your answers."
        formatted_prompts = []

        for _idx, query in enumerate(queries):
            query_time = query_times[_idx]
            retrieval_results = batch_retrieval_results[_idx]

            user_message = ""
            references = ""
            
            if len(retrieval_results) > 0:
                references += "# References \n"
                # Format the top sentences as references in the model's prompt template.
                for _snippet_idx, snippet in enumerate(retrieval_results):
                    references += f"- {snippet.strip()}\n"
            
            references = references[:MAX_CONTEXT_REFERENCES_LENGTH]
            # Limit the length of references to fit the model's input size.

            user_message += f"{references}\n------\n\n"
            user_message += f"Using only the references listed above, answer the following question: \n"
            user_message += f"Current Time: {query_time}\n"
            user_message += f"Question: {query}\n"
            
            formatted_prompts.append(
                self.tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

        return formatted_prompts

if __name__=='__main__':
    dataset_path = "/root/autodl-tmp/meta-comprehensive-rag-benchmark-kdd-cup-2024/data/dev_data.jsonl.bz2"
    
    rag = RAGModel()

    with bz2.open(dataset_path, "rt") as file: 
        for idx, line in enumerate(file): 
            jdata = json.loads(line)
            pages = jdata["search_results"]
            query = jdata["query"]
            print("########################  " + str(idx))
            print("query:   ")
            print(query)


