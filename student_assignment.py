import json
from model_configurations import get_model_configuration
#from rich import print as pprint

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.few_shot import FewShotChatMessagePromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage

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

# 定義 few-shot examples
examples = [
    {
        "input": "2024年台灣10月紀念日有哪些?",
        "output": "{\n    \"Result\": [\n        {\n            \"date\": \"2024-10-10\",\n            \"name\": \"國慶日\"\n        }\n    ]\n}"
    },
    {
        "input": "2024年台灣12月紀念日有哪些?",
        "output": "{\n    \"Result\": [\n        {\n            \"date\": \"2024-12-25\",\n            \"name\": \"行憲紀念日\"\n        }\n    ]\n}"
    }
]


# 建立範例模板
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}")
])

# 使用 FewShotChatMessagePromptTemplate
few_shot_prompt = FewShotChatMessagePromptTemplate(
    examples=examples,
    example_prompt=example_prompt
)

# 定義最終的提示模板
final_prompt = ChatPromptTemplate.from_messages([
    ("system", "請用 JSON 格式回答台灣特定月份的紀念日，並依照範例輸出，確保 JSON 格式完全符合範例格式，包括括號與縮排，去除'''jason及 '''。"),
    few_shot_prompt,
    ("human", "{input}")
])

# 定義 generate_hw01 函數
def generate_hw01(question):
    llm = initialize_llm()

    # 建立語言模型鏈
    chain = final_prompt | llm

    # 呼叫模型並解析結果
    response = chain.invoke({"input": question})
    #pprint(response.content)
    # 嘗試解析為 JSON 格式，移除不必要的標記
    try:
        if isinstance(response.content, str):
            response.content = response.content.replace('```json', '').replace('```', '').strip()
        result = json.loads(response.content)

        # 如果 "Result" 鍵不存在，則視為無效輸出
        if "Result" not in result or not isinstance(result["Result"], list):
            return json.dumps({"Result": []}, ensure_ascii=False, indent=4)

        return json.dumps(result, ensure_ascii=False, indent=4)

    except json.JSONDecodeError:
        # 如果解析失敗，返回預設格式
        return json.dumps({"Result": []}, ensure_ascii=False, indent=4)
    


def generate_hw02(question):
    pass
        
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
    #response = generate_hw01('2023台灣紀念日有哪些?')
    response = generate_hw01('2024年台灣10月紀念日有哪些?')
    #question2 = "2024年台灣10月紀念日有哪些?"
    #question3 = "根據先前的節日清單，這個節日{\"date\": \"10-31\", \"name\": \"蔣公誕辰紀念日\"}是否有在該月份清單？"
    #response = generate_hw03(question2, question3)
    #print(response)
    #response.pretty_print()
    print(response)
