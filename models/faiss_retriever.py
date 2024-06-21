#!/usr/bin/env python
# coding: utf-8

import json
import torch
import numpy as np
from typing import List
from abc import ABC
from langchain.schema import Document
from langchain.schema.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModel

class TextEmbedding(Embeddings, ABC):
    def __init__(self, emb_model_name_or_path, batch_size=2, max_len=512, device = "cuda", **kwargs):
        super().__init__(**kwargs)
        self.model = AutoModel.from_pretrained(
                    emb_model_name_or_path,
                    trust_remote_code = True,
                    device_map = device
                ).half()
        self.tokenizer = AutoTokenizer.from_pretrained(emb_model_name_or_path, trust_remote_code=True)
        if 'bge' in emb_model_name_or_path:
            self.DEFAULT_QUERY_BGE_INSTRUCTION_ZH = "为这个句子生成表示以用于检索相关文章："
        else:
            self.DEFAULT_QUERY_BGE_INSTRUCTION_ZH = ""
        self.emb_model_name_or_path = emb_model_name_or_path
        self.device = self.model.device
        self.batch_size = batch_size
        self.max_len = max_len
        print("successful load embedding model")

    def compute_kernel_bias(self, vecs, n_components=384):
        """
            bertWhitening: https://spaces.ac.cn/archives/8069
            计算kernel和bias
            vecs.shape = [num_samples, embedding_size]，
            最后的变换：y = (x + bias).dot(kernel)
        """
        mu = vecs.mean(axis=0, keepdims=True)
        cov = np.cov(vecs.T)
        u, s, vh = np.linalg.svd(cov)
        W = np.dot(u, np.diag(1 / np.sqrt(s)))
        return W[:, :n_components], -mu

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
            Compute corpus embeddings using a HuggingFace transformer model.
        Args:
            texts: The list of texts to embed.
        Returns:
            List of embeddings, one for each text.
        """
        num_texts = len(texts)
        texts = [t.replace("\n", " ") for t in texts]
        sentence_embeddings = []

        for start in range(0, num_texts, self.batch_size):
            end = min(start + self.batch_size, num_texts)
            batch_texts = texts[start:end]
            encoded_input = self.tokenizer(
                        batch_texts, 
                        max_length=512, 
                        padding=True, 
                        truncation=True,
                        return_tensors='pt'
                    ).to(self.device)

            with torch.no_grad():
                model_output = self.model(**encoded_input)
                # Perform pooling. In this case, cls pooling.
                if 'gte' in self.emb_model_name_or_path:
                    batch_embeddings = model_output.last_hidden_state[:, 0]
                else:
                    batch_embeddings = model_output[0][:, 0]

                batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
                sentence_embeddings.extend(batch_embeddings.tolist())

        # sentence_embeddings = np.array(sentence_embeddings)
        # self.W, self.mu = self.compute_kernel_bias(sentence_embeddings)
        # sentence_embeddings = (sentence_embeddings+self.mu) @ self.W
        # self.W, self.mu = torch.from_numpy(self.W).cuda(), torch.from_numpy(self.mu).cuda()
        return sentence_embeddings

    def embed_query(self, text: str) -> List[float]:
        """
            Compute query embeddings using a HuggingFace transformer model.
        Args:
            text: The text to embed.
        Returns:
            Embeddings for the text.
        """
        text = text.replace("\n", " ")
        if 'bge' in self.emb_model_name_or_path:
            encoded_input = self.tokenizer([self.DEFAULT_QUERY_BGE_INSTRUCTION_ZH + text], padding=True,
                                           truncation=True, return_tensors='pt').to(self.device)
        else:
            encoded_input = self.tokenizer([text], padding=True,
                                           truncation=True, return_tensors='pt').to(self.device)
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            # Perform pooling. In this case, cls pooling.
            sentence_embeddings = model_output[0][:, 0]
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        # sentence_embeddings = (sentence_embeddings + self.mu) @ self.W
        return sentence_embeddings[0].tolist()


class FaissRetriever(object):
    def __init__(self, model_path):
        self.embeddings = TextEmbedding(
                          emb_model_name_or_path=model_path)
        self.vector_store = None
        torch.cuda.empty_cache()
    
    def GetTopK(self, query, k=10, score_threshold = 0.75):
        if(self.vector_store == None):
            return None
        context = self.vector_store.similarity_search_with_score(query, k=k, score_threshold=score_threshold)
        return context
    def GetvectorStore(self, data):
        docs = []
        for idx, line in enumerate(data):
            line = line.strip("\n").strip()
            docs.append(Document(page_content=line, metadata={"id": idx}))
        self.vector_store = FAISS.from_documents(docs, self.embeddings)

if __name__ == '__main__':
    
    import bz2
    import re
    from main_content_extractor import MainContentExtractor
    
    base = "/root/autodl-tmp/codes/meta-comphrehensive-rag-benchmark-starter-kit/"
    model_name=base + "models/pretrained_model/gte-large-en" #text2vec-large-chinese
    faissretriever = FaissRetriever(model_name)
    
    dataset_path = "/root/autodl-tmp/meta-comprehensive-rag-benchmark-kdd-cup-2024/data/dev_data.jsonl.bz2"
    
    with bz2.open(dataset_path, "rt") as file: 
        for idx, line in enumerate(file): 
            jdata = json.loads(line)
            pages = jdata["search_results"]
            query = jdata["query"]
            print("########################  " + str(idx))
            print("query:   ")
            print(query)
            pages_sentences = []
            for page in pages:
                # print(page["page_url"])
                html_str = page["page_result"]
                
                extracted_markdown = MainContentExtractor.extract(html_str, include_links = False, output_format="markdown")
                if(extracted_markdown):
                    sentences = re.split(r'\#+',extracted_markdown)
                    pages_sentences.extend(sentences)
            faissretriever.GetvectorStore(pages_sentences)
            faiss_ans = faissretriever.GetTopK(query, 1)
            print(faiss_ans)
