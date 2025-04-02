from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from llama_cpp import Llama
from pydantic import BaseModel
import json
import time

aya_expanse_8b_path = r"C:\Users\admin\Desktop\OmniPDF\backend\models\bartowski\aya-expanse-8b-GGUF\aya-expanse-8b-Q4_K_S.gguf"

aya_expanse_8b_llm = Llama(
    model_path=aya_expanse_8b_path,
    gpu_layers=-1,
    seed=1337,
    n_ctx=2048,
)

class textData(BaseModel):
    prompt: str


class tableData(BaseModel):
    table_data: list

app = FastAPI()

@app.post("/api/translate-table")
async def translate_table(data: tableData):
    start_time = time.time()

    prompt = f"""
<|START_OF_TURN_TOKEN|><|USER_TOKEN|>Translate the following table into English, maintaining the original format:

{json.dumps(data.table_data)}

Provide only the translated table, without any additional explanations.<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|ASSISTANT_TOKEN|>
    """

    print("Received request:", prompt)

    try:
        output = aya_expanse_8b_llm(
            prompt=prompt,
            max_tokens=2048,  # Adjust based on table length
            temperature=0.2,
            top_p=0.95,
            repeat_penalty=1.1,
        )

        response_text = output.get("choices", [{}])[0].get("text", "").strip()

        # Process response_text to return to json format.
        try:
            translated_table = json.loads(response_text)
        except json.JSONDecodeError as e:
            return JSONResponse(content={"error": "Translation failed, invalid json response", "message": str(e)}, status_code=500)

        inference_time = time.time() - start_time
        print(f"Inference time: {inference_time} seconds")
        print(f"Response: {translated_table}")

        return JSONResponse(content={"translated_table": translated_table, "inference_time": inference_time})

    except Exception as e:
        print(f"Error during inference: {str(e)}")
        return JSONResponse(content={"error": "Translation failed", "message": str(e)}, status_code=500)


@app.post("/api/summarize-table")
async def summarize_table(data: tableData):
    return

@app.post("/api/translate-text")
async def translate_text(data: textData):
    start_time = time.time()

    prompt = f"""
<|START_OF_TURN_TOKEN|><|USER_TOKEN|>Translate the following text to English: 

{data.prompt}

Provide only the translated english text, without any additional explanations.<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|ASSISTANT_TOKEN|>

    """

    print("Received request:", prompt)

    try:
        output = aya_expanse_8b_llm(
            prompt=prompt,
            max_tokens=512,  # Increased max_tokens
            temperature=0.2, # lowered temperature further.
            top_p=0.95,
            repeat_penalty=1.1,
        )

        response_text = output.get("choices", [{}])[0].get("text", "").strip()

        inference_time = time.time() - start_time
        print(f"Inference time: {inference_time} seconds")
        print(f"Response: {response_text}")

        return JSONResponse(content={"translation": response_text, "inference_time": inference_time})

    except Exception as e:
        print(f"Error during inference: {str(e)}")
        return JSONResponse(content={"error": "Translation failed", "message": str(e)}, status_code=500)

@app.post("/api/summarize-text")
async def summarize_text(data: textData):
    return

@app.post("/api/caption-image")
async def caption_image():
    return

@app.post("/api/embed-text")
async def embed_text():
    return

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)