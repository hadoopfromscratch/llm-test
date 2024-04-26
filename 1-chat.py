from llama_cpp import Llama

def qna_cycle(llm, chat):
    print("User:", end=" ",flush=True)
    user_input = input()
    chat.append({"role": "user", "content": user_input})
    print("LLM:", end=" ",flush=True)
    response = llm.create_chat_completion(chat, stream=True)
    tokens = ""
    for r in response:
        try:
            token = r["choices"][0]["delta"]["content"]
            print(token, end="", flush=True)
            tokens += token
        except KeyError: pass
    chat.append({"role": "assistant", "content": f"{tokens}\n"})
    if len(chat) > 6:
        chat = chat[2:]
    return chat

llm = Llama(model_path="models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf", 
            chat_format="chatml", stream=True, n_gpu_layers=32,
            n_ctx=1024)
chat = []
while True:
    chat = qna_cycle(llm, chat)