# Example: reuse your existing OpenAI setup
from openai import OpenAI
import base64
import json
import time
from frontend.classes.TableDataProcessor import TableDataProcessor

# Point to the local server
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")


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
                "content": "You are an assistant specializing in translating content into english, the user will provide strings and your job is to translate them into english. Only reply with the translated english text with no further inputs while keeping the original formatting the same",
            },
            {"role": "user", "content": text},
        ],
        temperature=0,
    )
    inference_time = time.time() - start_time
    print(f"Inference time: {inference_time} seconds")
    return response.choices[0].message.content


def summarize_table(data: list, client: OpenAI) -> list:
    start_time = time.time()
    prompt = f"""
        <|START_OF_TURN_TOKEN|><|USER_TOKEN|>Provide a text summary in English of the following table provided:

        {json.dumps(data.table_data)}

        Provide only the text summary, without any additional explanations.<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|ASSISTANT_TOKEN|>
    """

    completion = client.chat.completions.create(
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

    response_text = completion.choices[0].message.content

    inference_time = time.time() - start_time
    print(f"Inference time: {inference_time} seconds")
    print(f"Response: {response_text}")

    return response_text


def translate_table(data: list, client: OpenAI) -> str:

    start_time = time.time()
    prompt = f"""
        <|START_OF_TURN_TOKEN|><|USER_TOKEN|>Translate the following table into English, maintaining the original format:

        {json.dumps(data.table_data)}

        Provide only the translated table, without any additional explanations.<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|ASSISTANT_TOKEN|>
    """

    completion = client.chat.completions.create(
        model="aya-expanse-8b",
        messages=[
            {
                "role": "system",
                "content": """
                    You are a helpful assistant tasked with translating the contents of the following table into English. Your translation should focus on preserving the original formatting and structure of the table while ensuring that the text is accurately translated. When translating, consider the following:
                    - Maintain the original formatting and structure of the table.
                    - Ensure that all text is accurately translated into English.
                    - Avoid adding any additional explanations or comments.
                """,
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    response_text = completion.choices[0].message.content

    inference_time = time.time() - start_time
    print(f"Inference time: {inference_time} seconds")
    print(f"Response: {response_text}")

    return response_text


def embed_text():
    return


tdp = TableDataProcessor()
