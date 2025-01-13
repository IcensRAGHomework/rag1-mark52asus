from functools import partial
import json
import traceback
import re
from rich import print as pprint


from model_configurations import get_model_configuration

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Dict
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain.output_parsers import (ResponseSchema, StructuredOutputParser)
import requests
import json

gpt_chat_version = 'gpt-4o'
gpt_config = get_model_configuration(gpt_chat_version)

def generate_hw01(question):
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

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."
                    "請回答台灣特定月份的紀念日有哪些(請用以下JSON格式呈現)?"
                    "如果只有一項資料，不需要用list方式"
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

    new_prompt = prompt.partial(format_intructions=json_format_instructions)

    user_query = question
    user_prompt = new_prompt.invoke({"query": user_query})
    response = llm.invoke(user_prompt)
    
    #pprint(json_parser.invoke(response)) #變成字典輸出
    return response.content.replace("```json\n", "").replace("\n```", "")
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
    pass
    
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
    #response = generate_hw01('2024年台灣10月紀念日有哪些?')
    response = generate_hw02('2024年台灣10月紀念日有哪些?')
    #print(response)
    #response.pretty_print()
    print(response)
