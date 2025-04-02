import json
import time

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
from pydantic import BaseModel

from classes.rag import RAGHelper


class textData(BaseModel):
    prompt: str


class tableData(BaseModel):
    table_data: list


class docData(BaseModel):
    doc_data: list[dict[str, str]]


app = FastAPI()


@app.post("/api/translate-table")
async def translate_table(data: tableData):
    start_time = time.time()

    try:
        client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

        prompt = f"""
            <|START_OF_TURN_TOKEN|><|USER_TOKEN|>Translate the following table into English, maintaining the original format:

            {json.dumps(data.table_data)}

            Provide only the translated table, without any additional explanations.<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|ASSISTANT_TOKEN|>
        """

        completion = client.chat.completions.create(
            model="aya-expanse-8b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        response_text = completion.choices[0].message.content

        # Process response_text to return to json format.
        try:
            translated_table = json.loads(response_text)
        except json.JSONDecodeError as e:
            return JSONResponse(
                content={
                    "error": "Translation failed, invalid json response",
                    "message": str(e),
                },
                status_code=500,
            )

        inference_time = time.time() - start_time
        print(f"Inference time: {inference_time} seconds")
        print(f"Response: {translated_table}")

        return JSONResponse(
            content={
                "translated_table": translated_table,
                "inference_time": inference_time,
            }
        )

    except Exception as e:
        print(f"Error during inference: {str(e)}")
        return JSONResponse(
            content={"error": "Translation failed", "message": str(e)}, status_code=500
        )


@app.post("/api/summarize-table")
async def summarize_table(data: tableData):
    start_time = time.time()

    try:
        client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

        prompt = f"""
            <|START_OF_TURN_TOKEN|><|USER_TOKEN|>Summarize :

            {json.dumps(data.table_data)}

            Provide only the translated table, without any additional explanations.<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|ASSISTANT_TOKEN|>
        """

        completion = client.chat.completions.create(
            model="aya-expanse-8b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        response_text = completion.choices[0].message.content

        # Process response_text to return to json format.
        try:
            translated_table = json.loads(response_text)
        except json.JSONDecodeError as e:
            return JSONResponse(
                content={
                    "error": "Translation failed, invalid json response",
                    "message": str(e),
                },
                status_code=500,
            )

        inference_time = time.time() - start_time
        print(f"Inference time: {inference_time} seconds")
        print(f"Response: {translated_table}")

        return JSONResponse(
            content={
                "translated_table": translated_table,
                "inference_time": inference_time,
            }
        )

    except Exception as e:
        print(f"Error during inference: {str(e)}")
        return JSONResponse(
            content={"error": "Translation failed", "message": str(e)}, status_code=500
        )


@app.post("/api/translate-text")
async def translate_text(data: textData):
    start_time = time.time()
