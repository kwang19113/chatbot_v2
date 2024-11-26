import requests
import json
import time
import prompt_layout
def generate_response(prompt):
    url = 'http://localhost:11434/api/generate'
    headers = {'Content-Type': 'application/json'}
    payload = {
        'model': 'gemma2:latest',  # Replace with your specific model name
        'prompt': prompt,
        'stream': False,  # Set to True if you want a streaming response
        'system_prompt': prompt_layout.SYSTEM_PROMPT,
        'temperature': TEMPERATURE
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()  # Raise an error for bad status codes
        result = response.json()
        return result.get('response', 'No response received.')
    except requests.exceptions.RequestException as e:
        return f'An error occurred: {e}'
def generate_query(prompt):
    url = 'http://localhost:11434/api/generate'
    headers = {'Content-Type': 'application/json'}
    payload = {
        'model': 'qwen2.5-coder:3b',  # Replace with your specific model name
        'prompt': prompt,
        'num_ctx': 2048,
        'stream': False,  # Set to True if you want a streaming response
        'system_prompt': prompt_layout.SYSTEM_PROMPT_QUERY,
        'temperature': TEMPERATURE
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()  # Raise an error for bad status codes
        result = response.json()
        return result.get('response', 'No response received.')
    except requests.exceptions.RequestException as e:
        return f'An error occurred: {e}'

import argparse
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
import json
from get_embedding_function import get_embedding_function
import warnings
import sqlite3
warnings.filterwarnings("ignore")

CHROMA_PATH = "chroma"

TEMPERATURE = "0.3"

user_info = None
previous_question_type = None
def main():
    # Create CLI.

    #print("\n✨✨✨✨USING MODEL:",model_name,"✨✨✨✨\n")
    # Initialize the conversation history as an empty list.
    conversation_history = []
    global user_info 
    while True:
        question = input("❓❓❓❓CÂU HỎI❓❓❓❓\n")
        if question.lower() == "exit":
            print("Chào tạm biệt!")
            break

        try:
            # Adding an exit check before sending the question to model
            if not question.strip():
                continue

            # Query model with conversation history
            response = query_rag(question, conversation_history)
            print(f"🤖: {response}")

            # Append the question and response to the conversation history.
            conversation_history.append(f"Người dùng: {question}")
            conversation_history.append(f"Trợ lý: {response}")

        except Exception as e:
            print(f"Đã có lỗi xảy ra: {e}")

def query_rag(query_text: str, conversation_history):
    global user_info
    global previous_question_type
    # Prepare the DB.
    start_time = time.time()
    question_type = question_classifier(query_text=query_text)
    print("QUESTION CLASSIFIER time:")
    print("--- %s seconds ---" % (time.time() - start_time))
    context_path = CHROMA_PATH+"/policy"
    match question_type:
        case 1:
            context_path = CHROMA_PATH+"/policy"
        case 2:
            context_path = CHROMA_PATH+"/products"
        case 5:
            context_path = CHROMA_PATH+"/contact"
    #check here
    start_time = time.time()
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=context_path, embedding_function=embedding_function)
    # Search the DB for similar context.
    results = db.similarity_search_with_score(query_text, k=2)
    print("EMBEDDING time:")
    print("--- %s seconds ---" % (time.time() - start_time))
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    #print("=====================CONTEXT",context_text)
    
    # print("+++++++++++++USER_INFO",user_info)
    # Create prompt with the entire history and current question
    prompt_template = ChatPromptTemplate.from_template(prompt_layout.PROMPT_TEMPLATE)
    if user_info:
        context_text += "\nthông tin người hỏi"+ json.dumps(user_info)
    # Invoke the model with the full prompt
    start_time = time.time()
    # Build the conversation history.
    if previous_question_type is None or previous_question_type != question_type:
        history_text = ""
    else:
        history_text = "\n".join(conversation_history)
    previous_question_type = question_type
    prompt = prompt_template.format(history=history_text, question=query_text, context= context_text)

    #print("QUESTION CLASSIFIER time:")
    #print("--- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    match question_type:
        case 4:
            response_text = "Câu hỏi của bạn không liên quan tới những gì công ty hay quy định công ty"
        case 1:
            response_text = generate_response(prompt)
        case 2:
            response_text = generate_response(prompt)
        case 3:
            if user_info is None:
                print("Hãy cung cấp ID và tên đầy đủ của bạn để tôi có thể hỗ trợ thêm")
                input_info = input()
                start_time = time.time()
                user_info = authentication(input_info)
                print("AUTHENTICATION time:")
                print("--- %s seconds ---" % (time.time() - start_time))
            if user_info:
                prompt = prompt_template.format(history=history_text, question=query_text, context= user_info)
                response_text = generate_response(prompt)
            else:
                print("Không tìm thấy thông tin trên cơ sở dữ liệu")
        case 5:
            response_text = generate_response(prompt)
        case 6:
            response_text = get_par(query_text)
    print("PROMPT time:")
    print("--- %s seconds ---" % (time.time() - start_time))    


    return response_text


def get_par(query_text):
    prompt_template_query = ChatPromptTemplate.from_template(prompt_layout.QUERY_PRMPT)
    prompt_query = prompt_template_query.format(question= query_text)

    conn = sqlite3.connect("database/chat.db")
    cursor = conn.cursor()

    select_query = generate_query(prompt_query).replace("sql\n", "").replace("\n", "").replace("\n","")
    print(select_query)

    cursor.execute(select_query)
    result_string = ""
    # Fetch and print the matching rows
    rows = cursor.fetchall()
    for row in rows:
        print(row)
        result_string = '\n'.join([str(row) for row in rows])

    # Close the connection
    conn.close()
    print("KẾT QUẢ TỪ DATABASE:",result_string)
    prompt_template = ChatPromptTemplate.from_template(prompt_layout.PAR_ANSWER_PROMPT)

    prompt = prompt_template.format(question= query_text, data = result_string )
    response_text = generate_response(prompt)


    
    return response_text


def authentication(input_information, db_path = "database/chat.db"):
    prompt_template = ChatPromptTemplate.from_template(prompt_layout.AUTHENTICATION_PROMPT)
    prompt = prompt_template.format(question= input_information)
    data = generate_response(prompt).replace("json\n", "").replace("\n", "").strip().replace("[","{").replace("]","}")

    print(data)
    export_info = json.loads(str(data))
    
    print("NAME", export_info["name"])
    print("ID", export_info["ID"])
    user_info = get_user_information(export_info["name"], export_info["ID"], db_path)


    return user_info

def question_classifier(query_text: str):
    CLASSIFIER_PROMPT = """

    Người dùng hỏi: "{question}"
    
    phân loại câu hỏi đó theo các chủ đề như sau: 
    1-câu hỏi về quy trình của công ty 
    2-câu hỏi về sản phẩm công nghệ của công ty 
    3-Câu hỏi về thông tin cá nhân 
    4-Câu hỏi không liên quan 
    5-Tìm thông tin liên lạc 
    6-Truy vết người với đặc điểm được cho
    trả lời bằng một số duy nhất:
    """
    prompt_template = ChatPromptTemplate.from_template(CLASSIFIER_PROMPT)
    prompt = prompt_template.format(question = query_text)
    classify_result = generate_response(prompt)
    print("THIS QUESTION IS CLASSIFY AS:")
    print(classify_result)
    return int(classify_result)

def get_user_information(name, ID, db_path):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Execute a query to get all information of a user with the given name and ID
    query = "SELECT * FROM users WHERE name = ? AND ID = ?"
    cursor.execute(query, (name, ID))
    result = cursor.fetchone()

    # Close the database connection
    conn.close()

    # If a result is found, return it; otherwise, return None
    if result:
        # You can also fetch the column names and return as a dictionary
        column_names = [description[0] for description in cursor.description]
        user_info = dict(zip(column_names, result))
        return user_info
    return None
if __name__ == "__main__":
    main()
