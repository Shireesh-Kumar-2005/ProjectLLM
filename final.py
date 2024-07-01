#!/usr/bin/env python
# coding: utf-8

# In[1]:


pdf_path = "report.pdf"


# In[2]:


import fitz
from tqdm.auto import tqdm
def text_formatter(text:str)->str:
    cleaned_text = text.replace("\n"," ").strip()
    return cleaned_text 

def open_and_read_pdf(pdf_path:str)->list[dict]:
    doc = fitz.open(pdf_path)
    pages_and_text = []
    for page_numbers, page in tqdm(enumerate(doc)):
        text = page.get_text() 
        text = text_formatter(text = text)
        pages_and_text.append({"page_number":page_numbers - 9,
                               "page_char_count":len(text),
                               "page_word_count":len(text.split(" ")),
                               "page_sentence_count_raw":len(text.split(". ")),
                               "page_token_count":len(text)/4,
                               "text":text
                               })
    return pages_and_text
pages_and_texts = open_and_read_pdf(pdf_path = pdf_path)
pages_and_texts[:2]


# In[3]:


import random
random.sample(pages_and_texts, k = 3 )


# In[4]:


import pandas as pd 
df = pd.DataFrame(pages_and_texts)
df.head()


# In[5]:


df.describe().round(2)


# In[6]:


from spacy.lang.en import English
nlp = English()
nlp.add_pipe('sentencizer')
doc = nlp("this is a sentence. This another sentence. I like elephants.")
assert len(list(doc.sents)) == 3
list(doc.sents)


# In[7]:


for item in tqdm(pages_and_texts):
    item['sentences'] = list(nlp(item["text"]).sents)
    item['sentences'] = [str(sentences) for sentences in item['sentences']]
    item['page_sentences_count_spacy'] = len(item["sentences"])


# In[8]:


random.sample(pages_and_texts, k = 1)


# In[9]:


df = pd.DataFrame(pages_and_texts)
df.describe().round(2)


# In[10]:


num_sentence_chunk_size = 10
def split_list(input_list : list[str],
               slice_size: int = num_sentence_chunk_size)-> list[list[str]]:
    return [input_list[i:i+slice_size] for i in range(0,len(input_list),slice_size)]

test_list = list(range(25))
split_list(test_list)
    
    


# In[11]:


for item in tqdm(pages_and_texts):
    item["sentence_chunks"] = split_list(input_list = item['sentences'],slice_size = num_sentence_chunk_size)
    item['num_chunks'] = len(item["sentence_chunks"])


# In[12]:


random.sample(pages_and_texts, k = 3)


# In[13]:


df =pd.DataFrame(pages_and_texts)
df.describe().round(2)


# In[14]:


import re 
pages_and_chunks = []
for item in tqdm(pages_and_texts):
    for sentence_chunk in item['sentence_chunks']:
        chunk_dict = {}
        chunk_dict["page_number"] = item['page_number']
        
        joined_sentence_chunks = "".join(sentence_chunk).replace(" "," ").strip()
        joined_sentence_chunks = re.sub(r'\.([A-Z])',r'. \1',joined_sentence_chunks)
        
        chunk_dict["sentence_chunks"] = joined_sentence_chunks
        
        chunk_dict["chunk_char_count"] = len(joined_sentence_chunks)
        chunk_dict["chunk_word_count"] = len([word for word in joined_sentence_chunks.split(" ")])
        chunk_dict["chunk_token_count"] = len(joined_sentence_chunks) / 4
        
        pages_and_chunks.append(chunk_dict)
        
len(pages_and_chunks)


# In[15]:


random.sample(pages_and_chunks,k = 1)


# In[16]:


df = pd.DataFrame(pages_and_chunks)
df.describe().round(2)


# In[17]:


min_token_length = 30
for row in df[df["chunk_token_count"] <= min_token_length].sample(5).iterrows():
    print(f'chunk token count: {row[1]["chunk_token_count"]} | Text : {row[1]["sentence_chunks"]}')


# In[18]:


pages_and_chunks_over_min_token_len = df[df["chunk_token_count"] > min_token_length].to_dict(orient = "records")
pages_and_chunks_over_min_token_len[:2]


# In[19]:


random.sample(pages_and_chunks_over_min_token_len, k = 1)


# In[20]:


from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer(model_name_or_path="all-MiniLM-L6-v2",device="cuda")
sentences = ["the sentence transformer library creates an easy way to create embeddings",
             "sentence can be embedded one by one in a list",
             "I like horses!"]
embeddings = embedding_model.encode(sentences)
embeddings_dict = dict(zip(sentences,embeddings))
for sentence , embedding in embeddings_dict.items():
    print(f"Sentence: {sentences}")
    print(f"Embedding: {embedding}")
    print(" ")


# In[21]:


embeddings[0].shape


# In[22]:


get_ipython().run_cell_magic('time', '', '\nembedding_model.to("cpu")\nfor item in tqdm(pages_and_chunks_over_min_token_len):\n    item[\'embedding\'] = embedding_model.encode(item["sentence_chunks"])\n')


# In[23]:


get_ipython().run_cell_magic('time', '', '\nembedding_model.to("cuda")\nfor item in tqdm(pages_and_chunks_over_min_token_len):\n    item[\'embedding\'] = embedding_model.encode(item["sentence_chunks"])\n')


# In[24]:


get_ipython().run_cell_magic('time', '', '\ntext_chunks = [item["sentence_chunks"] for item in pages_and_chunks_over_min_token_len]\ntext_chunks[40]\n')


# In[25]:


len(text_chunks)


# In[26]:


get_ipython().run_cell_magic('time', '', '\ntext_chunks_embeddings = embedding_model.encode(text_chunks, batch_size = 16,convert_to_tensor=True).to("cuda")\ntext_chunks_embeddings\n')


# In[27]:


text_chunks_and_embeddings_df = pd.DataFrame(pages_and_chunks_over_min_token_len)
embeddings_df_save_path = "text_chunks_and_embeddings_df.csv"
text_chunks_and_embeddings_df.to_csv(embeddings_df_save_path, index = False)


# In[28]:


text_chunks_and_embeddings_df_load = pd.read_csv(embeddings_df_save_path)
text_chunks_and_embeddings_df_load.head()


# In[29]:


import random
import torch 
import numpy as np 
import pandas as pd

text_chunks_and_embeddings_df = pd.read_csv("text_chunks_and_embeddings_df.csv")

text_chunks_and_embeddings_df["embedding"] = text_chunks_and_embeddings_df['embedding'].apply(lambda x: np.fromstring(x.strip("[]"),sep =" "))

embeddings = torch.tensor(np.stack(text_chunks_and_embeddings_df['embedding'].tolist(),axis = 0),dtype = torch.float32).to("cuda")

pages_and_chunks = text_chunks_and_embeddings_df.to_dict(orient = "records")
text_chunks_and_embeddings_df


# In[30]:


embeddings.shape


# In[31]:


from sentence_transformers import util , SentenceTransformer

embedding_model = SentenceTransformer(model_name_or_path="all-MiniLM-L6-v2",device = "cuda")


# In[32]:


query = " Climate  Change  Research  Complex "
print(f"Query: {query}")

query_embedding = embedding_model.encode(query, convert_to_tensor = True).to("cuda")

from time import perf_counter as timer
start_time = timer()
dot_score = util.dot_score(a = query_embedding, b = embeddings)[0]
end_time = timer()

print(f"[INFO] time taken to get scores on {len(embedding)} embedding: {end_time - start_time: .5f}seconds.")

top_results_dot_product = torch.topk(dot_score, k = 5)
top_results_dot_product
 


# In[33]:


pages_and_chunks[24]


# In[34]:


import textwrap 

def print_wrapped(text,wrap_length = 80):
    wrapped_text = textwrap.fill(text,wrap_length)
    print(wrapped_text)


# In[35]:


print(f"Query: '{query}'\n")
print("results:")

for score, idx in zip(top_results_dot_product[0],top_results_dot_product[1]):
    print(f"Score: {score:.4f}")
    print("text:")
    print(pages_and_chunks[idx]["sentence_chunks"])
    print(f"page number : {pages_and_chunks[idx]['page_number']}")
    print("\n")


# In[36]:


import fitz

pdf_path = "report.pdf"
doc = fitz.open(pdf_path)
page = doc.load_page(3+9)

img = page.get_pixmap(dpi = 300)

doc.close()

img_array = np.frombuffer(img.samples_mv,dtype = np.uint8).reshape((img.h, img.w, img.n))

import matplotlib.pyplot as plt 
plt.figure(figsize=(13,10))
plt.imshow(img_array)
plt.title(f"Query: '{query}' | most relevant page:")
plt.axis("off")
plt.show()


# In[37]:


embeddings[0]


# In[38]:


import torch 
def dot_product(vectro1,vector2):
    return torch.dot(vectro1,vector2)

def cosine_similarity(vector1,vector2):
    dot_product = torch.dot(vector1,vector2)
    
    norm_vector1 = torch.sqrt(torch.sum(vector1**2))
    norm_vector2 = torch.sqrt(torch.sum(vector2**2))
    
    return dot_product/(norm_vector1 * norm_vector2)

vector1 = torch.tensor([1,2,3] , dtype = torch.float32)
vector2 = torch.tensor([1,2,3] , dtype = torch.float32)
vector3 = torch.tensor([4,5,6] , dtype = torch.float32)
vector4 = torch.tensor([-1,-2,-3] , dtype = torch.float32)

print("dot product between vector1 and vector2: ",dot_product(vector1,vector2))
print("dot product between vector1 and vector3: ",dot_product(vector1,vector3))
print("dot product between vector1 and vector4: ",dot_product(vector1,vector4))

print(" ")

print("cosine similarity between vector 1 and vector 2: ",cosine_similarity(vector1, vector2))
print("cosine similarity between vector 1 and vector 3: ",cosine_similarity(vector1, vector3))
print("cosine similarity between vector 1 and vector 4: ",cosine_similarity(vector1, vector4))


# In[39]:


def retrieve_relavant_resource(query: str,
                               embeddings : torch.tensor,
                               model : SentenceTransformer = embedding_model,
                               n_resource_to_return: int = 5,
                               print_time: bool = True):
    
    query_embedding = model.encode(query,convert_to_tensor = True)
    start_time = timer()
    dot_scores = util.dot_score(query_embedding,embeddings)[0]
    end_timer = timer()

    if print_time:
        print(f"[INFO] Time taken to get scores on {len(embeddings)} embeddings: {end_time-start_time:5f} seconds." )
    
    scores , indices = torch.topk(input=dot_scores,k = n_resource_to_return )

    return scores,indices

def print_top_results_and_scores(query: str,
                                 embeddings: torch.tensor,
                                 pages_and_chunks : list[dict]=pages_and_chunks,
                                 n_resource_to_return: int = 5):
    
    scores, indicies = retrieve_relavant_resource(query = query,
                                                  embeddings = embeddings,
                                                  n_resource_to_return = n_resource_to_return)
    
    for score, idx in zip(scores,indicies):
        print(f"Score: {score:.4f}")
        print("text:")
        print(pages_and_chunks[idx]["sentence_chunks"])
        print(f"page number : {pages_and_chunks[idx]['page_number']}")
        print("\n")
    


# In[40]:


query = "foods high in fiber"
retrieve_relavant_resource(query = query , embeddings = embeddings)
print_top_results_and_scores(query = query, embeddings = embeddings)


# In[41]:


import torch 
gpu_memory_bytes = torch.cuda.get_device_properties(0).total_memory
gpu_memory_gb = round(gpu_memory_bytes/(2**30))
print(f"Available GPU memory : {gpu_memory_gb} GB")


# In[42]:


model_id= "google/gemma-2b-it"
# Note: the following is Gemma focused, however, there are more and more LLMs of the 2B and 7B size appearing for local use.
use_quantization_config = True
if gpu_memory_gb < 5.1:
    print(f"Your available GPU memory is {gpu_memory_gb}GB, you may not have enough memory to run a Gemma LLM locally without quantization.")
elif gpu_memory_gb < 8.1:
    print(f"GPU memory: {gpu_memory_gb} | Recommended model: Gemma 2B in 4-bit precision.")
    use_quantization_config = True 
    model_id = "google/gemma-2b-it"
elif gpu_memory_gb < 19.0:
    print(f"GPU memory: {gpu_memory_gb} | Recommended model: Gemma 2B in float16 or Gemma 7B in 4-bit precision.")
    use_quantization_config = False 
    model_id = "google/gemma-2b-it"
elif gpu_memory_gb > 19.0:
    print(f"GPU memory: {gpu_memory_gb} | Recommend model: Gemma 7B in 4-bit or float16 precision.")
    use_quantization_config = False 
    model_id = "google/gemma-7b-it"

print(f"use_quantization_config set to: {use_quantization_config}")
print(f"model_id set to: {model_id}")


# In[43]:


import torch 
from transformers import AutoTokenizer , AutoModelForCausalLM
from transformers.utils import is_flash_attn_2_available

from transformers import BitsAndBytesConfig
quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                         bnb_4bit_compute_dtype=torch.float16)

if(is_flash_attn_2_available() and (torch.cuda.get_device_capability(0)[0] >= 8)):
    attn_implementation = "flash_attention_2"
else:
    attn_implementation = "sdpa"
model_id = "google/gemma-2b-it"
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_id)
llm_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_id,
                                                torch_dtype = torch.float16,
                                                quantization_config = quantization_config if use_quantization_config else None,
                                                low_cpu_mem_usage = False,
                                                attn_implementation=attn_implementation)

if not use_quantization_config:
    llm_model.to("cuda")


# In[44]:


llm_model


# In[45]:


def get_model_num_params(model: torch.nn.Module):
    return sum([param.numel() for param in model.parameters()])
get_model_num_params(llm_model)


# In[46]:


def get_model_mem_size(model: torch.nn.Module):
    mem_params = sum([param.nelement() * param.element_size() for param in model.parameters()])
    mem_buffers = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])
    
    model_mem_bytes = mem_params + mem_buffers
    model_mem_mb = model_mem_bytes / (1024**2)
    model_mem_gb = model_mem_bytes / (1024**3)
    
    return {"model_mem_bytes" : model_mem_bytes,
            "model_mem_mb": round(model_mem_mb,2),
            "model_mem_gb": round(model_mem_gb,2)}
    
get_model_mem_size(llm_model)
    


# ### generating text with our LLM 

# In[47]:


gpt4_questions = ["What is the definition of rainfed agriculture, and why is it considered the predominant form of agriculture in India?",
                  "Which organizations, such as FAO, ICARDA, and CRIDA, have been mentioned in the context of rainfed agriculture, and what are their contributions?",
                  "What was the significance of the establishment of the All India Coordinated Research Project on Agrometeorology by ICAR in 1983?",
                  "How might climate change affect rainfed agriculture, and what measures can be taken to mitigate its effects?"]


# In[48]:


def prompt_formatter(query:str,
                     context_items: list[dict]) -> str:
    
    context = "- " + "\n- ".join([item["sentence_chunks"] for item in context_items])
    base_prompt = """Based on the following context items, please answer the query.
Give yourself room to think by extracting relevant passages from the context before answering the query.
Don't return the thinking, only return the answer.
Make sure your answers are as explanatory as possible.
Use the following examples as reference for the ideal answer style.
\nExample 1:
Query: What are the fat-soluble vitamins?
Answer: The fat-soluble vitamins include Vitamin A, Vitamin D, Vitamin E, and Vitamin K. These vitamins are absorbed along with fats in the diet and can be stored in the body's fatty tissue and liver for later use. Vitamin A is important for vision, immune function, and skin health. Vitamin D plays a critical role in calcium absorption and bone health. Vitamin E acts as an antioxidant, protecting cells from damage. Vitamin K is essential for blood clotting and bone metabolism.
\nExample 2:
Query: What are the causes of type 2 diabetes?
Answer: Type 2 diabetes is often associated with overnutrition, particularly the overconsumption of calories leading to obesity. Factors include a diet high in refined sugars and saturated fats, which can lead to insulin resistance, a condition where the body's cells do not respond effectively to insulin. Over time, the pancreas cannot produce enough insulin to manage blood sugar levels, resulting in type 2 diabetes. Additionally, excessive caloric intake without sufficient physical activity exacerbates the risk by promoting weight gain and fat accumulation, particularly around the abdomen, further contributing to insulin resistance.
\nExample 3:
Query: What is the importance of hydration for physical performance?
Answer: Hydration is crucial for physical performance because water plays key roles in maintaining blood volume, regulating body temperature, and ensuring the transport of nutrients and oxygen to cells. Adequate hydration is essential for optimal muscle function, endurance, and recovery. Dehydration can lead to decreased performance, fatigue, and increased risk of heat-related illnesses, such as heat stroke. Drinking sufficient water before, during, and after exercise helps ensure peak physical performance and recovery.
\nNow use the following context items to answer the user query:
{context}
\nRelevant passages: <extract relevant passages from the context here>
User query: {query}
Answer:"""
    base_prompt = base_prompt.format(context=context, query=query)

    # Create prompt template for instruction-tuned model
    dialogue_template = [
        {"role": "user",
        "content": base_prompt}
    ]

    # Apply the chat template
    prompt = tokenizer.apply_chat_template(conversation=dialogue_template,
                                          tokenize=False,
                                          add_generation_prompt=True)
    return prompt


# In[49]:


query = random.choice(gpt4_questions)
print(f"Query: {query}")

# Get relevant resources
scores, indices = retrieve_relavant_resource(query=query,embeddings=embeddings)
    
# Create a list of context items
context_items = [pages_and_chunks[i] for i in indices]

# Format prompt with context items
prompt = prompt_formatter(query=query,
                          context_items=context_items)
print(prompt)


# In[50]:


get_ipython().run_cell_magic('time', '', '\ninput_ids = tokenizer(prompt,return_tensors="pt").to("cuda")\n\noutputs = llm_model.generate(**input_ids,\n                             temperature = 0.7,\n                             do_sample = True,\n                             max_new_tokens=300)\n\noutput_text = tokenizer.decode(outputs[0])\nprint(f"Query: {query}")\nprint(f"GEMMA\'s Answer:\\m{output_text.replace(prompt, \'\')}")\n')


# ### Functionize our LLM answering feature 

# In[51]:


def ask(query, 
        temperature=0.7,
        max_new_tokens=512,
        format_answer_text=True, 
        return_answer_only=True):
    """
    Takes a query, finds relevant resources/context and generates an answer to the query based on the relevant resources.
    """
    
    # Get just the scores and indices of top related results
    scores, indices = retrieve_relavant_resource(query=query,embeddings=embeddings)
    
    # Create a list of context items
    context_items = [pages_and_chunks[i] for i in indices]

    # Add score to context item
    for i, item in enumerate(context_items):
        item["score"] = scores[i].cpu() # return score back to CPU 
        
    # Format the prompt with context items
    prompt = prompt_formatter(query=query,
                              context_items=context_items)
    
    # Tokenize the prompt
    input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")

    # Generate an output of tokens
    outputs = llm_model.generate(**input_ids,
                                 temperature=temperature,
                                 do_sample=True,
                                 max_new_tokens=max_new_tokens)
    
    # Turn the output tokens into text
    output_text = tokenizer.decode(outputs[0])

    if format_answer_text:
        # Replace special tokens and unnecessary help message
        output_text = output_text.replace(prompt, "").replace("<bos>", "").replace("<eos>", "").replace("Sure, here is the answer to the user query:\n\n", "")

    # Only return the answer without the context items
    if return_answer_only:
        return output_text
    
    return output_text, context_items


# In[52]:


print(ask(query = "what is rainfed agriculture"))


# In[53]:


print(ask(query = "who is john nicholas"))


# In[54]:


print(ask(query = "explain about ICAR CRIDA"))


# In[55]:


print(ask(query = "what is the average salary of farmer"))


# In[58]:


print(ask(query = "what is rainfed agriculture"))


# In[ ]:




