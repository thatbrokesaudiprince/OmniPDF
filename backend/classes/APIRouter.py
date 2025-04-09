import base64
import json
import os
import time

from openai import OpenAI

# For LM Studio models
LM_API_URL = os.getenv("LM_API_URL")
LM_API_KEY = os.getenv("LM_API_KEY")

# Point to the local LM Studio server
CLIENT = OpenAI(base_url=LM_API_URL, api_key=LM_API_KEY)


def caption_image(image_path: str, client: OpenAI) -> str:
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

    start_time = time.time()
    response = client.chat.completions.create(
        model="llava-v1.5-7b",
        messages=[
            {
                "role": "system",
                "content": "You are an assistant specializing in generating captions for images.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe this image in a concise caption.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    },
                ],
            },
        ],
        temperature=0.7,
    )
    inference_time = time.time() - start_time
    print(f"Inference time: {inference_time} seconds")
    return response.choices[0].message.content


def translate_text(text: str, client: OpenAI) -> str:
    start_time = time.time()
    response = client.chat.completions.create(
        model="aya-expanse-8b",
        messages=[
            {
                "role": "system",
                "content": "You are an assistant specializing in translating content into english, the user will provide strings and your job is to translate them into english. Only reply with the translated english text with no further inputs and explanation while keeping the original formatting the same",
            },
            {"role": "user", "content": text},
        ],
        temperature=0,
    )
    inference_time = time.time() - start_time
    print(f"Inference time: {inference_time} seconds")
    return response.choices[0].message.content


def summarize_text(text: str, client: OpenAI) -> str:
    start_time = time.time()
    response = client.chat.completions.create(
        model="aya-expanse-8b",
        messages=[
            {
                "role": "system",
                "content": "You are an assistant specializing in summarizing content. Your task is to provide a concise summary of the given text. Focus on the main points and key information, while avoiding unnecessary details.",
            },
            {"role": "user", "content": text},
        ],
        temperature=0.2,
    )
    inference_time = time.time() - start_time
    print(f"Inference time: {inference_time} seconds")
    return response.choices[0].message.content


def summarize_table(data: list, client: OpenAI) -> list:
    start_time = time.time()
    prompt = f"""
        <|START_OF_TURN_TOKEN|><|USER_TOKEN|>Provide a text summary in English of the following table provided:

        {json.dumps(data)}

        Provide only the text summary, without any additional explanations.<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|ASSISTANT_TOKEN|>
    """

    response = client.chat.completions.create(
        model="aya-expanse-8b",
        messages=[
            {
                "role": "system",
                "content": """
                    You are a helpful assistant tasked with summarizing the contents of the following table. Your summary should focus on extracting the most important information, identifying key trends, and providing a concise, easy-to-understand overview. When summarizing, consider the following:
                    - Identify and highlight key data points.
                    - Mention any trends or patterns that emerge from the table.
                    - Keep the summary brief and to the point, avoiding unnecessary details.
                    - Make sure the summary is clear and understandable even for someone unfamiliar with the table’s context.
                    The summaries will also be used for retrieval augmented generation purposes.
                """,
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    inference_time = time.time() - start_time
    print(f"Inference time: {inference_time} seconds")
    return response.choices[0].message.content


def translate_table(table, client: OpenAI) -> str:
    data = {"table_data": table}
    start_time = time.time()
    print(json.dumps(data["table_data"]))
    prompt = f"""
    <|START_OF_TURN_TOKEN|><|USER_TOKEN|>Translate the following table into English, maintaining the original format:

    {json.dumps(data['table_data'])}

    Provide only the translated table, without any additional explanations.<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|ASSISTANT_TOKEN|>
        """

    response = client.chat.completions.create(
        model="aya-expanse-8b",
        messages=[
            {
                "role": "system",
                "content": """
                    You are a helpful assistant tasked with translating the tables into English. The user will give u inputs in the form of nested lists,
                    Translate each list contents and return with a nested list of the translated contents.
                """,
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    inference_time = time.time() - start_time
    print(f"Inference time: {inference_time} seconds")

    return json.loads(response.choices[0].message.content)


def rag_prompt(prompt, docs, pages_data, client: OpenAI) -> str:
    start_time = time.time()

    helping = ""
    for i, doc in enumerate(docs):
        if doc.metadata.get("type") == "text":
            helping += f"- {doc.page_content}\n"
        elif doc.metadata.get("type") == "table":
            found = False
            for page in pages_data:
                for trans_table_summary in page.get("translated_tables_summary", []):
                    if trans_table_summary["key"] == doc.metadata.get(
                        "trans_table_summary_key"
                    ):
                        helping += f"- {trans_table_summary['translated_table']}\n"
                        found = True
                        break
                if found:
                    break
        elif doc.metadata.get("type") == "image":
            found = False
            for page in pages_data:
                for image in page.get("images", []):
                    if image["key"] == doc.metadata.get("image_caption_key"):
                        helping += f"- {image['caption']}\n"
                        found = True
                        break
                if found:
                    break

        # Keep number of tokens within limit
        if len(helping) > 8192:
            helping = helping[: 8000 - len(prompt)]
            docs = docs[: i + 1]
            break

    prompt_with_help = f"""Answer this prompt:
        {prompt}
        Use the following pieces of context (if needed) to support and answer the question.
        {helping}
    """

    response = client.chat.completions.create(
        model="aya-expanse-8b",
        messages=[
            {
                "role": "system",
                "content": """
                    You are a helpful assistant for answering user questions with detailed information. You also have RAG capabilities, thus you will be given retrieved documents that are relevant to the question.
                """,
            },
            {"role": "user", "content": prompt_with_help},
        ],
        temperature=0.2,
    )
    inference_time = time.time() - start_time
    print(f"Inference time: {inference_time} seconds")

    return response.choices[0].message.content, docs
