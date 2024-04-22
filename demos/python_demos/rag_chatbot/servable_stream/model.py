#*****************************************************************************
# Copyright 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#*****************************************************************************

########### Workaround: https://docs.trychroma.com/troubleshooting#sqlite
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
############
import os
import re
#from pprintpp import pprint as pp
from pyovms import Tensor
import time
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from threading import Thread

from tritonclient.utils import deserialize_bytes_tensor, serialize_byte_tensor
from langchain_community.llms import HuggingFacePipeline
from optimum.intel.openvino import OVModelForCausalLM
import torch
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from typing import Any, List, Mapping, Optional
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from transformers import (
    AutoModelForCausalLM,
    AutoModel,
    AutoTokenizer,
    AutoConfig,
    TextIteratorStreamer,
    pipeline,
    StoppingCriteria,
    StoppingCriteriaList,
    set_seed,
)

from config import SUPPORTED_EMBEDDING_MODELS, SUPPORTED_LLM_MODELS
from ov_embedding_model import OVEmbeddings

SEED = os.environ.get('SEED', 10)

SELECTED_MODEL = os.environ.get('SELECTED_MODEL', 'tiny-llama-1b-chat')
LANGUAGE = os.environ.get('LANGUAGE', 'English')
llm_model_configuration = SUPPORTED_LLM_MODELS[LANGUAGE][SELECTED_MODEL]

EMBEDDING_MODEL = 'all-mpnet-base-v2'
embedding_model_configuration = SUPPORTED_EMBEDDING_MODELS[EMBEDDING_MODEL]

llm_model_dir = "/llm_model"
model_name = llm_model_configuration["model_id"]
stop_tokens = llm_model_configuration.get("stop_tokens")
class_key = SELECTED_MODEL.split("-")[0]
tok = AutoTokenizer.from_pretrained(llm_model_dir, trust_remote_code=True)
LocalRun=0
embedding_model_dir = "/embed_model"

class StopOnTokens(StoppingCriteria):
    def __init__(self, token_ids):
        self.token_ids = token_ids

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        for stop_id in self.token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

if stop_tokens is not None:
    if isinstance(stop_tokens[0], str):
        stop_tokens = tok.convert_tokens_to_ids(stop_tokens)
    stop_tokens = [StopOnTokens(stop_tokens)]

from ov_llm_model import model_classes
model_class = (
    AutoModelForCausalLM
    if not llm_model_configuration["remote"]
    else model_classes[class_key]
)
ov_config = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1", "CACHE_DIR": ""}

print(f"Loading LLM model {SELECTED_MODEL}...")
gov_model = model_class.from_pretrained(
    llm_model_dir, torch_dtype=torch.float32, device_map='auto', trust_remote_code=True,
)
#gov_model = model_class.from_pretrained(
#    llm_model_dir,
#    device="AUTO",
#    ov_config=ov_config,
#    compile=True,
#    config=AutoConfig.from_pretrained(llm_model_dir, trust_remote_code=True),
#    trust_remote_code=True)        
print("LLM model loaded")




# Document Splitter
from typing import List
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter, MarkdownTextSplitter
from langchain_community.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PDFMinerLoader,
    TextLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader, )
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
#from langchain.chains import RetrievalQA
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.docstore.document import Document

class ChineseTextSplitter(CharacterTextSplitter):
    def __init__(self, pdf: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.pdf = pdf

    def split_text(self, text: str) -> List[str]:
        if self.pdf:
            text = re.sub(r"\n{3,}", "\n", text)
            text = text.replace("\n\n", "")
        sent_sep_pattern = re.compile(
            '([﹒﹔﹖﹗．。！？]["’”」』]{0,2}|(?=["‘“「『]{1,2}|$))')
        sent_list = []
        for ele in sent_sep_pattern.split(text):
            if sent_sep_pattern.match(ele) and sent_list:
                sent_list[-1] += ele
            elif ele:
                sent_list.append(ele)
        return sent_list


TEXT_SPLITERS = {
    "Character": CharacterTextSplitter,
    "RecursiveCharacter": RecursiveCharacterTextSplitter,
    "Markdown": MarkdownTextSplitter,
    "Chinese": ChineseTextSplitter,
}


LOADERS = {
    ".csv": (CSVLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PDFMinerLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
}

import os

def read_generated_text():
    _dir = '/documents/descriptions/'
    files = [file for file in os.listdir(_dir) if file.endswith('.txt')]
    results = []
    for file in files:
        path = _dir + file
        with open(path, 'r') as file:
            content = file.read()

        results.append((file.name, content))
    return results

def read_images():
    _dir = '/documents/videos/'
    files = os.listdir(_dir)
    return [file for file in files]

def read_embeddings(file_name):
    file_path = f'/documents/embeddings/{file_name}'
    _data = torch.load(file_path, map_location='cpu')
    return _data[0]

def read_all_embeddings():
    _dir = '/documents/embeddings/'
    files = os.listdir(_dir)
    all_embeddings = {}
    for file in files:
        all_embeddings[file] = read_embeddings(file)
    return all_embeddings

def get_list_of_embeddings():
    _dir = '/documents/embeddings/'
    files = os.listdir(_dir)
    return files



def load_single_document(file_path: str) -> List[Document]:
    """
    helper for loading a single document

    Params:
      file_path: document path
    Returns:
      documents loaded

    """
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADERS:
        loader_class, loader_args = LOADERS[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()

    raise ValueError(f"File does not exist '{ext}'")


def default_partial_text_processor(partial_text: str, new_text: str):
    """
    helper for updating partially generated answer, used by default

    Params:
      partial_text: text buffer for storing previosly generated text
      new_text: text update for the current step
    Returns:
      updated text string

    """
    partial_text += new_text
    return partial_text


text_processor = llm_model_configuration.get(
    "partial_text_processor", default_partial_text_processor
)


def deserialize_prompts(batch_size, input_tensor):
    if batch_size == 1:
        return [bytes(input_tensor).decode()]
    np_arr = deserialize_bytes_tensor(bytes(input_tensor))
    return [arr.decode() for arr in np_arr]


def serialize_completions(batch_size, result):
    if batch_size == 1:
        return [Tensor("completion", result.encode())]
    return [Tensor("completion", serialize_byte_tensor(
        np.array(result, dtype=np.object_)).item())]

from threading import Thread
from typing import Optional
from transformers import TextIteratorStreamer

class CustomLLM(LLM):
    n: int  
    streamer: Optional[TextIteratorStreamer] = None

    def wrap_image_embeddings(self, x):        
        pattern = r"'embedding_path': '([^']*)'"

        # Using re.search to find the pattern in the string
        match = re.search(pattern, x)
        
        if match:
            embedding_path = match.group(1)
            #print(embedding_path)
        else:
            print("Embedding path not found in the string.")

        _user_input = x.split('Question:')[1]
        #print ('User input ', _user_input)

        embeddings = read_embeddings(embedding_path)

        #print (embeddings.shape)
        text_summary = x[x.find('page_content') : x.find('metadata')]
        #print ('TEXT summmary:', text_summary)
    
        # Wrapping embeddings between prompt question
        img_list = [embeddings[:, 56:88, :]]
        # for img in img_list: print (f'\t{img.shape}')
        
        # prompt = "You are helpful assistant who understands visual context and can answer user's questions in details. Visual Context: \n<Video><ImageHere></Video>###" + _user_input
        
        # Prompt template
        DEFAULT_RAG_PROMPT = """\
        You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\
        """
        
        prompt = f"""[INST]Human: <<SYS>> {DEFAULT_RAG_PROMPT }<</SYS>>
                    Text Summary: {text_summary},
                    Question: {_user_input}
                    Context: <Video><ImageHere></Video>                    
                    Answer: [/INST]"""

        print ('PROMPT = ', prompt, flush=True)

        prompt_segs = prompt.split('<ImageHere>')
        
        seg_tokens = [
            tok(
                seg, return_tensors="pt", add_special_tokens=i == 0).to('cpu').input_ids
            # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]
        # print (f'seg_tokens type {type(seg_tokens)} size {len(seg_tokens)}')
        #print(dir(gov_model))
        
        
        #print(dir(gov_model.model))
        seg_embs = [gov_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
        # print (f'seg_embs type {type(seg_embs)} size {len(seg_embs)}')
        #for seg_t in seg_embs: print (f'\t type {type(seg_t)} size {seg_t.shape}')
        
        mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
        # print (f'mixed_embs type {type(mixed_embs)} size {len(mixed_embs)}')
        #for x in mixed_embs: print (f'\t type {type(x)} size {x.shape}')
        
        mixed_embs = torch.cat(mixed_embs, dim=1)        
        return mixed_embs, embedding_path
    
    def _call(
            self, 
            prompt: str, 
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
        ) -> str:
        #print("CustomLLM: In _call")
        prompt_length = len(prompt)
        #print (prompt_length, prompt_length)
        
        mixed_embs, epath = self.wrap_image_embeddings (prompt)        
        #prompt = "Who is first president of USA?"
        
        # only return newly generated tokens
        self.streamer = TextIteratorStreamer(tok, skip_prompt=True, Timeout=5)
        inputs = tok(prompt, return_tensors="pt")
        output = gov_model.generate(inputs_embeds = mixed_embs, 
                                    max_new_tokens = 512,
                                    num_return_sequences = 1,
                                    num_beams = 1,
                                    min_length = 1,
                                    top_p = 0.9,
                                    top_k = 50,
                                    repetition_penalty = 1.0,
                                    length_penalty = 1,
                                    temperature = 1.0,
                                    pad_token_id=tok.eos_token_id,
                                    do_sample=True,
                                    streamer=self.streamer)
        res = tok.decode(output[0], skip_special_tokens=True)
        #print("Resp1:", res, flush=True)
        response = res + '####' + epath
        #print("Resp2:", response, flush=True)
        return response
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        #print("CustomLLM: In _identifying_params")
        return {"n": self.n}
    
    @property
    def _llm_type(self) -> str:
        #print("CustomLLM: In _llm_type")
        return "custom"

class OvmsPythonModel:
    def __init__(self): #self is the current instance
        self.name="Test"
        
    def initialize(self, kwargs: dict):
        '''
        print(f"Loading LLM model {SELECTED_MODEL}...")
        gov_model = model_class.from_pretrained(
            llm_model_dir,
            device="AUTO",
            ov_config=ov_config,
            compile=True,
            config=AutoConfig.from_pretrained(llm_model_dir, trust_remote_code=True),
            trust_remote_code=True)        
        print("LLM model loaded")
        '''
        print(f"Loading embedding model {EMBEDDING_MODEL}...")
        self.embedding = OVEmbeddings.from_model_id(
            embedding_model_dir,
            do_norm=embedding_model_configuration["do_norm"],
            ov_config={
                "device_name": "CPU",
                "config": {"PERFORMANCE_HINT": "THROUGHPUT"},
            },
            model_kwargs={
                "model_max_length": 512,
            },
        )
        print("Embedding model loaded")
        print("Building document database...")

        # following code (till from_documents()) can be commented
        if False:
            # if we store db in persistence memory
            documents = []
            for file_path in os.listdir("/documents/descriptions"):
                abs_path = f"/documents/descriptions/{file_path}"
                print(f"Reading document {abs_path}...")
                documents.extend(load_single_document(abs_path))
            spliter_name = "RecursiveCharacter"  # TODO: Param?
            chunk_size=1000  # TODO: Param?
            chunk_overlap=200  # TODO: Param?
            text_splitter = TEXT_SPLITERS[spliter_name](chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            self.texts = text_splitter.split_documents(documents)
            persist_directory='/workspace/db'
            self.db = Chroma.from_documents(self.texts, self.embedding, persist_directory=persist_directory)
        else:
            # create embeddings from new model
            from sentence_transformers import SentenceTransformer
            #from sentence_transformers.util import cos_sim

            # 1. load model
            model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

            # For retrieval you need to pass this prompt.
            """
            query = 'Represent this sentence for searching relevant passages: A man is eating a piece of bread'

            docs = [
                query,
                "A man is eating food.",
                "A man is eating pasta.",
                "The girl is carrying a baby.",
                "A man is riding a horse.",
            ]
            """

            # 2. Encode
            #embeddings = model.encode(docs)
            # till here

            list_of_embeddings = read_all_embeddings()
            print("# Embeddings:", len(list_of_embeddings))
            list_of_generated_text = read_generated_text(); 
            print ("# Text documents:", len(list_of_generated_text))
            list_of_image_paths = read_images();
            print ("# image paths:", len(list_of_image_paths))
            list_of_embedding_paths = get_list_of_embeddings(); 
            print ("# Embedding paths:", len(list_of_embedding_paths))
            data = {}
            for text_file, content in list_of_generated_text:
                #print(text_file)
                id = text_file.split('/')[3].replace('.mp4', '').replace('.txt', '')
                #print(id)
                data[id] = {
                    'text': content,
                }
                #print(id, content)
            #print(data.keys())
            for file in list_of_embedding_paths:
                x = file.replace('.mp4', '').replace('.pt', '')
                #print(file, x)
                data[x]['embedding_path'] = f'{file}'
            #print(data['0320241915_002_300_601'])
            # Preparing documents
            docs = []
            text = []

            for chunk, content in data.items():
                if 'embedding_path' in content.keys():

                    if 'image_path' not in content.keys():
                        content['image_path'] = 'NA'
                    
                    doc_object = Document(
                        page_content = content['text'],
                        metadata = {
                            'embedding_path': content['embedding_path'],
                            'image_path': content['image_path'],
                        },
                    )
                
                    docs.append(doc_object)
                    text.append(content['text'])

            print ("#Docs: ", len(docs))
            self.embeddings = model.encode(text[:len(text)])
            
            #spliter_name = "RecursiveCharacter"  # TODO: Param?
            #chunk_size=1000  # TODO: Param?
            #chunk_overlap=200  # TODO: Param?
            #text_splitter = TEXT_SPLITERS[spliter_name](chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            #self.texts = text_splitter.split_documents(documents)
            persist_directory='/workspace/db1'
            self.db = Chroma.from_documents(docs, self.embeddings, persist_directory=persist_directory)
        #x = docs[0]
        #print("First doc:", x)
        #print("First doc:", x.metadata)
        # till here can be commented if db stored in persistence memory
        # persist_directory='/workspace/db'
        #self.db = Chroma(persist_directory=persist_directory, embedding_function=self.embedding)
        vector_search_top_k = 1  #4  # TODO: Param?
        self.retriever = self.db.as_retriever(search_kwargs={"k": vector_search_top_k})
        print("Document database loaded", flush=True)
    def execute(self, inputs: list):
        print("Executing", flush=True)
        if SEED is not None: set_seed(int(SEED))
        if (LocalRun==0):
            batch_size = inputs[0].shape[0]
            if batch_size != 1:
                print("Error !!")
                raise ValueError("Batch size must be 1")
        
            prompts = deserialize_prompts(batch_size, inputs[0])
        else:
            print("In local run", flush=True)
            prompts = inputs[0]
        llm = CustomLLM(n=10)
        template = 'Context: {context} Question: {question}'
        prompt = ChatPromptTemplate.from_template(template)
        
        rag_chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}            
            | prompt
            | llm            
        )
        question = prompts[0]
        result = rag_chain.invoke(question)
        result = result.split('####')
        
        # Need to comment these 3 yield lines for local execution !
        yield [Tensor("completion", result[0].encode())]
        yield [Tensor("completion", '#'.encode())]
        yield [Tensor("completion", result[1].encode())]
        yield [Tensor("end_signal", "".encode())]            
        print(result)            
        return

if __name__ == "__main__":
    ovmsObj = OvmsPythonModel()
    ovmsObj.initialize({'1':'2'})
    LocalRun=1    
    inputs=[["Is a person with blue shirt seen in the videos?"]]    
    inputs=[["Describe in detail the video and its contents"]]    
    ovmsObj.execute(inputs)    
    print("Done executon")
    ovmsObj.execute(inputs)    
    print("Done executon")
    ovmsObj.execute(inputs)    
    print("Done executon")
    ovmsObj.execute(inputs)    
    print("Done executon")
