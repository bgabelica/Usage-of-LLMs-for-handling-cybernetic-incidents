import os
from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import AutoTokenizer
from huggingface_hub import login

#login(token="") # potrebno stvoriti besplatan huggingface token i ovdje ga umetnuti za koristenje embedder modela


loader = PyPDFLoader("./files/ccs_actions-attack-only-descr.pdf")
all_splits = loader.load_and_split()

for d in all_splits:
    d.page_content = d.page_content.replace("\n", "")
    d.page_content = d.page_content.replace("#", ";;;  Action name: ")
    d.page_content = d.page_content.replace("  ", " ")
    d.page_content = d.page_content.replace("   ", " ")
    d.page_content = d.page_content.replace("Description:", "; Description:")
    d.page_content = d.page_content.replace("Keywords:", "; Keywords:")

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")

#model_name = "BAAI/bge-large-en-v1.5"
#model_kwargs = {"device": "cuda"}
#encode_kwargs = {"normalize_embeddings": True}
#hf = HuggingFaceBgeEmbeddings(
#    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
#)

#vectorstore = Chroma.from_documents(documents=all_splits, embedding=hf)

file_path = os.path.join(os.path.dirname(__file__), "mixtral-8x7b-instruct-v0.1.Q6_K.gguf")

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

llm = LlamaCpp(
    model_path=file_path,
    temperature=0.1,
    max_tokens=300,
    top_p=1,
    n_gpu_layers=19,
    n_batch=2048,
    n_ctx=4096,
    callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager


)


sysprompt = """You are assisting a user that is training as an attacker in a cyberattack simulator, you have to suggest to the user the next action to perform based on his needs and possibilities. 
A list of possible actions with their description is given here: {context}. 
This is the summary of what happened so far: {summary}. 
The user's situation is: {q}. Respond in one sentence by only giving the next possible action from the list of actions."""

summarys = ""

summaryPrompt = """Please write a summary of what was discussed so far. You should only describe what happened to the user so far, what the user wants and which actions were taken by the user. This is the previous summary: {prevsum}. This is newly asked question: {q}. This is the answer to the question: {answer}.
                 Only list actions that are mentioned in this prompt. Write the minimum number of sentences. Your anser should be in this format: ''Firstly the user said A happened. The LLM suggested to do action B. The user did action B and said C is now the problem. The LLM suggested doing action D.''"""

def saveSummary(sum):
    global summarys
    
    summarys= sum
    


while True: 
    question = input("What is your next question? ")

    #docs = vectorstore.similarity_search(question)
    #cont = docs[0].page_content + ";;;" + docs[1].page_content + ";;;"
    cont = ""

    for d in all_splits:
        cont = cont + ";;;" + d.page_content

    syspromptconcrete = sysprompt.format(context = cont, summary = summarys, q = question)

    chat = [{"role": "user", "content": syspromptconcrete}]
   
    #print(chat)

    realprompt = tokenizer.apply_chat_template(chat, tokenize=False) # prompt za trazenje iduce akcije
    realprompt = realprompt + " " + "The next action to take would be "

    prompt = PromptTemplate.from_template("{historyprompt}")
    final = prompt.partial(historyprompt=realprompt)
    

    #print(final)

    chain = final | llm | StrOutputParser()

    response = chain.invoke({})

    fullresponse = "The next action to take would be " + response
    
    new_response = summaryPrompt.format(prevsum = summarys, q = question, answer = fullresponse) # prompt za stvaranje novog sazetka

    chat = [{"role":"user", "content":new_response}]
     
    realprompt = tokenizer.apply_chat_template(chat, tokenize=False)
    realprompt = realprompt + " " + "Firstly, the user said that"

    prompt = PromptTemplate.from_template("{historyprompt}")
    final = prompt.partial(historyprompt=realprompt) 

    #print(final)

    chain = final | llm | saveSummary

    chain.invoke({})

    #print("Summary:" + summarys)

    

  

