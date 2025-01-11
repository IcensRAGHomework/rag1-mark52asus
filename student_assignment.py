from functools import partial
import json
import traceback
import re
import requests
from rich import print as pprint
from uuid import uuid4


from model_configurations import get_model_configuration

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from typing import List, Dict
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain.output_parsers import (ResponseSchema, StructuredOutputParser)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

gpt_chat_version = 'gpt-4o'
gpt_config = get_model_configuration(gpt_chat_version)


# 初始化 Azure OpenAI (替換為你的設定)
def initialize_llm():
    return AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )
# 定義 Few-Shot Examples
def define_few_shot_examples():
    examples = [
        {
            "input": "2024年台灣10月有哪些紀念日？",
            "output": {
                "Result": [
                    {"date": "2024-10-10", "name": "國慶日"},
                    {"date": "2024-10-09", "name": "重陽節"}
                ]
            }
        },
        {
            "input": "2024年台灣2月有哪些紀念日？",
            "output": {
                "Result": []
            }
        }
    ]
    return examples

# 定義 Few-Shot Template
def define_few_shot_prompt():
    examples = define_few_shot_examples()
    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("ai", "{output}")
    ])
    return FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples
    )

# 定義主要 Prompt
def define_main_prompt(few_shot_prompt):
    return ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. "
                   "請回答台灣特定月份的紀念日有哪些? "
                   "請將結果存入 Result 欄位，紀念日日期存放在 date，紀念日名稱存在 name。"
                   "請直接輸出符合 JSON 格式的結果，且無多餘的解釋文字。"),
        *few_shot_prompt.format_prompt().to_messages(),
        ("human", "{query}")
    ])

def escape_json(json_obj):
    """自動將 JSON 物件轉換為雙大括號格式"""
    return json.dumps(json_obj, ensure_ascii=False).replace("{", "{{").replace("}", "}}")
# 使用 langchain 和 Few-Shot Examples 回答問題
def generate_hw01(question):
    # 初始化模型
    llm = initialize_llm()

    # ✅ 定義範例數據 (使用雙大括號避免變數解析衝突)
    examples = [
        {
            "input": "2024年台灣10月紀念日有哪些?",
            "output": """{{
        "Result": [
            {{
                "date": "2024-10-10",
                "name": "國慶日"
            }},
            {{
                "date": "2024-10-25",
                "name": "台灣光復節"
            }}
        ]
    }}"""
        },
        {
            "input": "2024年台灣12月紀念日有哪些?",
            "output": """{{
        "Result": [
            {{
                "date": "2024-12-25",
                "name": "聖誕節"
            }}
        ]
    }}"""
        }
    ]

    # ✅ 定義範例模板 (使用 ChatPromptTemplate 並確保正確轉義)
    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("ai", "{output}")
    ])

    # ✅ 使用 FewShotChatMessagePromptTemplate
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        examples=examples,
        example_prompt=example_prompt
    )

    # ✅ 正確提取範例訊息 (直接使用雙大括號)
    messages = []
    for example in examples:
        messages.append(HumanMessage(content=example["input"]))
        messages.append(AIMessage(content=example["output"]))

    # ✅ 建立最終 PromptTemplate (已處理 JSON 格式衝突)
    final_prompt = ChatPromptTemplate.from_messages(
        [("system", "你是一個能夠以 JSON 格式回應的 AI 助手，請參考以下範例並生成符合範例的輸出。")]
        + [(message.type, message.content) for message in messages]
        + [("human", "{question}")]
    )

    # ✅ 使用 RunnableSequence
    chain = final_prompt | llm

    response = chain.invoke({"question": question})
    result = json.loads(response.content.strip())
    formatted_result = json.dumps(result, indent=4, ensure_ascii=False)
    return formatted_result
    #return response.content

    
# 定義函數：透過 Calendarific API 取得台灣的假日資料
def getHolidayData(month):
    Calendarific_api_key = "GnZ2XwPFTIZIlqVgDdPZMT5nHOp5NCYv"
    url = "https://calendarific.com/api/v2/holidays/json"
    params = {
        "api_key": Calendarific_api_key,
        "country": "TW",  # 台灣
        "year": 2024,
        "month": month
    }

    # 向 Calendarific 發送請求
    response = requests.get(url, params=params)

    # 檢查 API 回應狀態
    if response.status_code == 200:
        data = response.json().get("response", {}).get("holidays", [])
        if data:
            # 轉換為符合格式的 JSON 格式
            return [{"date": holiday["date"]["iso"], "name": holiday["name"]} for holiday in data]
        else:
            return [{"date": "N/A", "name": "無紀念日"}]
    else:
        return [{"date": "N/A", "name": "API查詢失敗"}]
# 定義函數：從使用者查詢中提取月份
def extract_month_from_query(query):
    # 使用正則表達式抓取數字前面帶有"月"的格式
    match = re.search(r"(\d{1,2})月", query)
    if match:
        return match.group(1)  # 回傳數字月份
    return None

def generate_hw02(question):
    llm = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )
    json_parser = JsonOutputParser()
    json_format_instructions = json_parser.get_format_instructions()

    # 建立 ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages([
        ("system", "請根據以下資訊回答台灣{month}月的紀念日: {holidays}"
                    "如果只有一項資料，不需要用list方式"
                    "假日名稱轉成繁體中文"
                    """{{
                    "Result": 
                        {{
                            "date": "2024-10-10",
                            "name": "國慶日"
                        }}
                    }}
                    """               
                   "{format_intructions}"),
        ("human", "{query}")
    ])
    # 使用者查詢示例
    user_query = question

    # 從使用者查詢中提取月份
    month = extract_month_from_query(user_query)
    if month:
        # 使用 partial 傳遞假日資料
        partial_prompt = prompt.partial(holidays=getHolidayData(month), month=month, format_intructions=json_format_instructions)
        user_prompt = partial_prompt.invoke({"query": user_query})
        #pprint("生成的 Prompt:", partial_prompt)
    else:
        print("無法識別月份，請輸入有效的查詢格式。")
    response = llm.invoke(user_prompt)
    return response.content.replace("```json\n", "").replace("\n```", "")
        
def generate_hw03(question2, question3):
    llm = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )
    # 建立 Prompt Template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("human", "{query}")
    ])
    # 初始化對話記憶體
    memory = ConversationBufferMemory(return_messages=True)
    # 定義 get_session_history 函數，用於獲取對話歷史
    def get_session_history(session_id):
        # 在這裡直接回傳 ConversationBufferMemory，作為簡單的示範
        return memory
    # 使用 RunnableWithMessageHistory 將記憶體與模型綁定
    runnable_with_memory = RunnableWithMessageHistory(
        runnable=llm, 
        memory=memory,
        input_key="query",
        get_session_history=get_session_history
    )
    # 模擬一個唯一的對話 ID (UUID)
    session_id = str(uuid4())

    # 第一次對話
    user_input_1 = question2
    response_1 = runnable_with_memory.invoke({"query": user_input_1}, config={"configurable": {"session_id": session_id}})
    print("AI 回應:", response_1)

    return response_1
    
def generate_hw04(question):
    pass
    
def demo(question):
    llm = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )
    message = HumanMessage(
            content=[
                {"type": "text", "text": question},
            ]
    )
    response = llm.invoke([message])
    
    return response
if __name__ == '__main__':
    response = generate_hw01('2024年台灣10月紀念日有哪些?')
    #response = generate_hw02('2024年台灣12月紀念日有哪些?')
    #question2 = "2024年台灣10月紀念日有哪些?"
    #question3 = "根據先前的節日清單，這個節日{\"date\": \"10-31\", \"name\": \"蔣公誕辰紀念日\"}是否有在該月份清單？"
    #response = generate_hw03(question2, question3)
    #print(response)
    #response.pretty_print()
    pprint(response)

