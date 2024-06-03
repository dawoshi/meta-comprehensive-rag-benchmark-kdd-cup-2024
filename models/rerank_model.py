import os
import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
class reRankLLM(object):
    def __init__(self, model_path, max_length = 512, device='cuda:0', batch_size = 2):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).half().to(device).eval()
        self.max_length = max_length
        self.device = device
        self.batch_size = batch_size
        print("load rerank model ok")
    def predict(self, query, docs):
        docs_ = []
        for item in docs:
            if isinstance(item, str):
                docs_.append(item)
            else:
                docs_.append(item.page_content)
        docs = list(set(docs_))
        pairs = []
        scores = []
        for d in docs:
            pairs.append([query, d])
        num_pairs = len(pairs)
        for start in range(0, num_pairs, self.batch_size):
            end = min(start + self.batch_size, num_pairs)
            batch_pairs = pairs[start:end] 
            with torch.no_grad():
                batch_inputs = self.tokenizer(batch_pairs, padding=True, truncation=True, return_tensors='pt', max_length=512).to(self.device)
                batch_score = self.model(**batch_inputs, return_dict=True).logits.view(-1, ).float().cpu().tolist()
                scores.extend(batch_score)
        docs = [(docs[i], scores[i]) for i in range(len(docs))]
        docs = sorted(docs, key = lambda x: x[1], reverse = True)
        docs_ = []
        for item in docs:
            docs_.append(item[0])
        return docs_

if __name__ == "__main__":
    import re
    import bz2
    from main_content_extractor import MainContentExtractor
    from bm25_retriever import Bm25Retriever
    bm25 = Bm25Retriever() 
    
    dataset_path = "/root/autodl-tmp/meta-comprehensive-rag-benchmark-kdd-cup-2024/data/dev_data.jsonl.bz2"
    bge_reranker_large = "/root/autodl-tmp/codes/meta-comphrehensive-rag-benchmark-starter-kit/models/pretrained_model/bge-rerank"
    rerank = reRankLLM(bge_reranker_large)
    
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
                html_str = page["page_result"]
                extracted_markdown = MainContentExtractor.extract(html_str, include_links = False, output_format="markdown")
                if(extracted_markdown):
                    sentences = re.split(r'\#+',extracted_markdown)
                    pages_sentences.extend(sentences)
            bm25.init_bm25(pages_sentences)
            bm25_response = bm25.GetBM25TopK(query, 10)
            texts = [doc.page_content for doc in bm25_response]
            rerank_res = rerank.predict(query, texts)
            print(rerank_res[:1])
