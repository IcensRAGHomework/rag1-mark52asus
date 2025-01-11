import json
from model_configurations import get_model_configuration
from rich import print as pprint

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
        "output": '{\n    "Result": [\n        {\n            "date": "2024-10-10",\n            "name": "國慶日"\n        }\n    ]\n}'
    },
    {
        "input": "2024年台灣12月紀念日有哪些?",
        "output": '{\n    "Result": [\n        {\n            "date": "2024-12-25",\n            "name": "行憲紀念日"\n        }\n    ]\n}'
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
    ("system", "請用 JSON 格式回答台灣特定月份的紀念日，並依照範例輸出。"),
    few_shot_prompt,
    ("human", "{input}")
])

# 定義 generate_hw01 函數
def generate_hw01(question):
    llm = initialize_llm()
    # 建立訊息
    message = HumanMessage(content=[{"type": "text", "text": question}])

    # 建立語言模型鏈
    chain = final_prompt | llm

    # 呼叫模型並回傳結果
    response = chain.invoke({"input": question})
    return response.content

    


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
    # 嘗試解析為 JSON 格式，確保格式正確
    try:
        result = json.loads(response.content)
        formatted_result = json.dumps(result, ensure_ascii=False, indent=4)
    except json.JSONDecodeError:
        formatted_result = "無法解析為正確的 JSON 格式"

    return formatted_result

    return result    
    #return response
if __name__ == '__main__':
    response = generate_hw01('2024年台灣10月紀念日有哪些?')
    #response = generate_hw02('2024年台灣12月紀念日有哪些?')
    #question2 = "2024年台灣10月紀念日有哪些?"
    #question3 = "根據先前的節日清單，這個節日{\"date\": \"10-31\", \"name\": \"蔣公誕辰紀念日\"}是否有在該月份清單？"
    #response = generate_hw03(question2, question3)
    #print(response)
    #response.pretty_print()
    pprint(response)

