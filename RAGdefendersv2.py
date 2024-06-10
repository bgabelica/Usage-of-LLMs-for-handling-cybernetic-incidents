from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate


import os




loader = PyPDFLoader("./files/ccs_actions-defense-only-descr.pdf")
all_splits = loader.load_and_split() # stvaranje chunksa iz dokumenta

for d in all_splits:
    d.page_content = d.page_content.replace("\n", "")
    d.page_content = d.page_content.replace("#", ";;;  Action name: ")
    d.page_content = d.page_content.replace("  ", " ")
    d.page_content = d.page_content.replace("   ", " ")
    d.page_content = d.page_content.replace("Description:", "; Description:")
    d.page_content = d.page_content.replace("Keywords:", "; Keywords:")

model_name = "BAAI/bge-large-en-v1.5" # embedder model
model_kwargs = {"device": "cuda"}
encode_kwargs = {"normalize_embeddings": True}
hf = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

vectorstore = Chroma.from_documents(documents=all_splits, embedding=hf) # stvaranje vectorstorea

file_path = os.path.join(os.path.dirname(__file__), "llama-2-70b-chat.Q5_K_M.gguf") # ucitavanje modela

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

llm = LlamaCpp(
    model_path=file_path,
    temperature=0.1,
    max_tokens=500,
    n_gpu_layers=34,
    n_batch=2048,
    n_ctx=4096,
    callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
)



base = """<s>[INST] <<SYS>>\n\nYou are are a helpful assistant in a cybersecurity simulator. 
You have to assist a person training as an defender in the simulator and by listening to what they are saying you should suggest what action they should perform next from the list of actions. 
The person you are helping is a manager in an entreprise. Use the documentation of the simulator that has all possible actions the defender can take if needed: {context}. 
This is the summary of the conversation until now: {summary}\n\n
Respond by thinking step by step and then giving the name of the action. For example: "Let's think step by step. Since the user already performed action A and their goal is to do B, the next action to take would be C. Final answer: C."<</SYS>>
\n"""

createsummarybase = """<s>[INST] <<SYS>>\n\nYou are are a helpful assistant that creates a summary of the conversation between an LLM and a person based on a previous summary and the newly added question and answer to the question.\n\n<</SYS>>
\n"""

promptcreatesummary = """ This is the previous summary: {summary}. This is the newly asked question: {questionq}. This is the answer the LLM gave to the question: {responser}. 
Please create a short summary of the conversation by starting with "This is what the human and the LLM have discussed so far:". 
Only include a short description of what happened so far and which actions were taken, do not explain why these actions were chosen. 
For example: "This is what the human and the LLM have discussed so far. The user said A happened. The LLM suggested to do action B. The user did action B and said C is now the problem. The LLM suggested doing action D."[/INST] """

newquestion = ""
summarys = ""

def saveSummary(sum):
    global summarys
    summarys= sum


while True: 
    question = input("What is your next question? ")
    newquestion = question + " [/INST] "

    docs = vectorstore.similarity_search(question) # izvadi najslicnije dijelove dokumentacije iz vectorstorea

    pagecontext = docs[0].page_content + ";;;" + docs[1].page_content + ";;;"

    for d in all_splits:
        pagecontext = pagecontext + " " +d.page_content
    #print(pagecontext)

    temp = base.format(context = pagecontext, summary=summarys)
    temp2 = temp + newquestion

    prompt = PromptTemplate.from_template("{historyprompt}")
    partial_prompt = prompt.partial(historyprompt=temp2) # stvori prompt za upit llmu


    chain = partial_prompt | llm | StrOutputParser()

    response = chain.invoke({}) # pokreni lanac

    temp = promptcreatesummary.format(summary=summarys, questionq=question, responser=response)
    temp2 = createsummarybase + temp

    prompt = PromptTemplate.from_template("{historyprompt}") # stvori prompt za generiranje sazetka
    partial_prompt = prompt.partial(historyprompt=temp2)


    chain = partial_prompt | llm | saveSummary

    chain.invoke({})

    #print(summarys)



