"""
提示词模板模块
包含各种场景的提示词模板
"""
from typing import Dict, Any
from langchain.prompts import ChatPromptTemplate, PromptTemplate


class PromptTemplates:
    """提示词模板类"""
    
    # 客服系统提示词
    CUSTOMER_SERVICE_SYSTEM_PROMPT = """
你是一个专业的电商客服助手，名叫小助手。请遵循以下准则：

1. 身份与态度：
   - 始终保持礼貌、专业、友好的态度
   - 以客户为中心，积极解决问题
   - 语言简洁明了，避免冗长

2. 回答规范：
   - 基于提供的知识库信息回答问题
   - 如果知识库中没有相关信息，请说"很抱歉，我需要为您转接人工客服"
   - 不要编造或猜测信息
   - 确保回答准确可靠

3. 语言风格：
   - 使用"您"来称呼客户
   - 开头使用"您好"问候
   - 结尾使用"还有其他问题吗？"或"如有其他问题请随时咨询"

4. 特殊情况处理：
   - 遇到投诉问题时，表示理解并引导联系人工客服
   - 涉及退换货时，提供明确的操作步骤
   - 关于优惠活动，提供准确的活动信息

知识库内容：
{context}

用户问题：{question}

请基于以上信息为用户提供专业的客服回复：
"""

    # RAG检索提示词
    RAG_RETRIEVAL_PROMPT = """
基于以下上下文信息，回答用户的问题。

上下文信息：
{context}

用户问题：{input}

请提供准确、有帮助的回答。如果上下文中没有相关信息，请说明无法基于现有信息回答。

回答：
"""

    # 对话历史提示词
    CONVERSATION_PROMPT = """
以下是之前的对话历史：
{chat_history}

当前用户问题：{question}

基于对话历史和当前问题，请提供合适的回答：
"""

    # 安全审核提示词
    SECURITY_REVIEW_PROMPT = """
请审核以下客服回复内容是否符合规范：

内容：{text}

审核标准：
1. 不能包含歧视性、攻击性语言
2. 不能透露用户隐私信息
3. 不能有虚假承诺或误导信息
4. 语言要礼貌、专业
5. 不能包含政治敏感内容

如果内容完全符合规范，请回复"通过"。
如果不符合规范，请回复"不通过：[具体原因]"。

审核结果：
"""

    # 格式化提示词
    FORMAT_RESPONSE_PROMPT = """
请将以下客服回复格式化为标准的客服用语：

原始内容：{text}

格式化要求：
1. 开头使用"您好"问候
2. 内容要礼貌、专业
3. 结尾使用"如有其他问题请随时咨询"
4. 保持原意不变，只调整表达方式

格式化后的回复：
"""

    # 意图识别提示词
    INTENT_CLASSIFICATION_PROMPT = """
请分析用户问题的意图类型，从以下类别中选择最合适的一个：

类别：
- 咨询：询问产品信息、服务流程等
- 投诉：对产品或服务不满意
- 退换货：申请退货、换货
- 物流：查询订单、物流状态
- 支付：支付相关问题
- 优惠：咨询优惠活动、折扣信息
- 其他：不属于以上类别的问题

用户问题：{question}

请只回复意图类别，不需要其他解释。

意图类别：
"""

    # 情感分析提示词
    SENTIMENT_ANALYSIS_PROMPT = """
请分析用户问题的情感倾向：

用户问题：{question}

情感选项：
- 积极：用户态度友好、满意
- 中性：用户态度平和、客观
- 消极：用户态度不满、愤怒或失望

请只回复情感倾向，不需要其他解释。

情感倾向：
"""

    @classmethod
    def get_rag_prompt(cls) -> ChatPromptTemplate:
        """获取RAG检索提示词模板"""
        return ChatPromptTemplate.from_template(cls.RAG_RETRIEVAL_PROMPT)

    @classmethod
    def get_customer_service_prompt(cls) -> ChatPromptTemplate:
        """获取客服系统提示词模板"""
        return ChatPromptTemplate.from_template(cls.CUSTOMER_SERVICE_SYSTEM_PROMPT)

    @classmethod
    def get_conversation_prompt(cls) -> ChatPromptTemplate:
        """获取对话历史提示词模板"""
        return ChatPromptTemplate.from_template(cls.CONVERSATION_PROMPT)

    @classmethod
    def get_security_review_prompt(cls) -> PromptTemplate:
        """获取安全审核提示词模板"""
        return PromptTemplate(
            input_variables=["text"],
            template=cls.SECURITY_REVIEW_PROMPT
        )

    @classmethod
    def get_format_response_prompt(cls) -> PromptTemplate:
        """获取格式化提示词模板"""
        return PromptTemplate(
            input_variables=["text"],
            template=cls.FORMAT_RESPONSE_PROMPT
        )

    @classmethod
    def get_intent_classification_prompt(cls) -> PromptTemplate:
        """获取意图识别提示词模板"""
        return PromptTemplate(
            input_variables=["question"],
            template=cls.INTENT_CLASSIFICATION_PROMPT
        )

    @classmethod
    def get_sentiment_analysis_prompt(cls) -> PromptTemplate:
        """获取情感分析提示词模板"""
        return PromptTemplate(
            input_variables=["question"],
            template=cls.SENTIMENT_ANALYSIS_PROMPT
        )

    @classmethod
    def get_custom_prompt(cls, template: str, input_variables: list) -> PromptTemplate:
        """创建自定义提示词模板"""
        return PromptTemplate(
            input_variables=input_variables,
            template=template
        )


# 常用提示词实例
prompt_templates = PromptTemplates() 