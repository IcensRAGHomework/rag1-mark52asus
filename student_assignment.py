from functools import partial
import json
import traceback
import re
import requests
from rich import print as pprint
from uuid import uuid4


from model_configurations import get_model_configuration

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
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

# 使用 langchain 和 Few-Shot Examples 回答問題
def generate_hw01(question):
    # 初始化模型
    llm = initialize_llm()

    # 定義 Few-Shot Prompt 和主要 Prompt
    few_shot_prompt = define_few_shot_prompt()
    main_prompt = define_main_prompt(few_shot_prompt)

    # 格式化使用者查詢
    user_prompt = main_prompt.invoke({"query": question})

    # 呼叫模型並取得回應
    response = llm.invoke(user_prompt)
    #pprint(response.content);
    return response.content



    
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

# 初始化消息歷史記錄
message_history = ChatMessageHistory()

# 儲存和檢索記憶的函數
class MemoryManager:
    def __init__(self):
        self.storage = {}

    def store(self, key, value):
        self.storage[key] = value

    def retrieve(self, key, default=None):
        return self.storage.get(key, default)

# 初始化記憶管理器
memory = MemoryManager()

# 創建一個消息歷史函數
def get_session_history():
    return memory.retrieve("message_history", [])

# 節日清單作為前一次對話結果
holiday_list = {
    "Result": [
        {"date": "2024-10-10", "name": "國慶日"},
        {"date": "2024-10-09", "name": "重陽節"},
        {"date": "2024-10-21", "name": "華僑節"},
        {"date": "2024-10-25", "name": "台灣光復節"},
        {"date": "2024-10-31", "name": "萬聖節"}
    ]
}


# 檢查給定節日是否存在的函數
def check_holiday_existence(stored_holiday_list, target_holiday):
    for holiday in stored_holiday_list["Result"]:
        if holiday["date"].endswith(target_holiday["date"]) and holiday["name"] == target_holiday["name"]:
            return True
    return False

def generate_hw03(question2, question3):
    #dialogue = f'{question2}{question3}'
    response = generate_hw02(question2)
    dialogue = f'{question3}'

    # 將節日清單儲存到記憶中
    #memory.store("holiday_list", holiday_list)
    # 將 JSON 字符串轉換為 Python 字典
    response_dict = json.loads(response)
    memory.store("holiday_list", response_dict)


    # 使用正則表達式提取 JSON 字符串
    match = re.search(r'\{.*\}', dialogue)

    if match:
        holiday_json_str = match.group()
        try:
            # 修正單引號為雙引號，確保 JSON 格式正確
            holiday_json_str = holiday_json_str.replace("'", '"')
            
            # 將 JSON 字符串轉換為 Python 字典
            target_holiday = json.loads(holiday_json_str)
            
            # 從記憶中取出節日清單
            stored_holiday_list = memory.retrieve("holiday_list")
            
            # 檢查是否存在
            exists = check_holiday_existence(stored_holiday_list, target_holiday)
            
            # 根據檢查結果生成回應
            if not exists:
                reason = (
                    f"{target_holiday['name']}並未包含在十月的節日清單中。"
                    f"目前十月的現有節日包括"
                    f"{', '.join([holiday['name'] for holiday in stored_holiday_list['Result']])}。"
                    f"因此，如果該日被認定為節日，應該將其新增至清單中。"
                )
                return json.dumps({
                    "Result": {
                        "add": True,
                        "reason": reason
                    }
                }, ensure_ascii=False)
            else:
                reason = (
                    f"{target_holiday['name']}已經存在於十月的節日清單中。"
                    f"目前十月的現有節日包括"
                    f"{', '.join([holiday['name'] for holiday in stored_holiday_list['Result']])}。"
                )
                return json.dumps({
                    "Result": {
                        "add": False,
                        "reason": reason
                    }
                }, ensure_ascii=False)
        except json.JSONDecodeError:
            return "解析 JSON 時出錯，請檢查輸入格式是否正確。"
    else:
        return "未找到節日信息"


if __name__ == '__main__':
    #response = generate_hw01("2024年台灣10月紀念日有哪些?")
    question2 = "2024年台灣10月紀念日有哪些?"
    question3 = {"date": "10-31", "name": "蔣公誕辰紀念日"}
    response = generate_hw03(question2, question3)
    pprint(response)
