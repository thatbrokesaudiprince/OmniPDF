import os
import shutil
import tempfile

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from classes.PDFProcessor import PDFProcessor
from classes.RAGHelper import RAGHelper
from classes.APIRouter import translate_text, rag_prompt, CLIENT

app = FastAPI()

# Enable CORS for frontend dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_helper = RAGHelper()


@app.get("/")
def hello_world():
    exampleClass = HelloWorld()
    return {"Message": exampleClass.get()}


@app.post("/pdf_pages")
async def retrieve_pdf_pages(file: UploadFile = File(...)):
    """Retrieve PDF pages.

    Pages can be individually processed and updated on the progress bar.
    """

    extension = os.path.splitext(file.filename)[1] or ".pdf"
    if extension not in [".pdf"]:
        raise HTTPException(status_code=400, detail="Unsupported file format.")

    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp:
        tmp_path = tmp.name
        file.file.seek(0)
        with open(tmp_path, "wb") as out_file:
            shutil.copyfileobj(file.file, out_file)

    try:
        global pdf_processor
        pdf_processor = PDFProcessor(tmp_path)
        return {"num_pages": pdf_processor.get_pages()}
    except Exception as e:
        print(f"[ERROR] Getting PDF pages failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to get PDF pages.")


@app.post("/process_pdf_page")
async def process_pdf(payload: dict):
    """Process PDFs by extracting images, text, and tables per page."""

    try:
        page_number = payload["page_number"]
        pages_data, documents = pdf_processor.process_pdf_page(page_number)
        # print(pages_data, documents)
        return {"pages_data": pages_data, "documents": documents}
    except Exception as e:
        print(f"[ERROR] PDF processing failed: {e}")
        # print(payload)
        raise HTTPException(status_code=500, detail="Failed to process PDF.")


@app.post("/ingest")
async def ingest_documents(payload: dict):
    """Ingest Documents into the vector database."""

    try:
        docs = payload.get("documents")

        if not docs:
            raise HTTPException(status_code=400, detail="No Documents found.")

        rag_helper.add_docs_to_chromadb(docs)

        return {"message": "Documents ingested successfully."}
    except Exception as e:
        print(f"[ERROR] Document ingestion failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to ingest documents.")


@app.post("/translate/")
async def translate(payload: dict):
    """Translate vernacular text to English."""

    try:
        text = payload["text"]

        if not text:
            raise HTTPException(status_code=400, detail="No text found.")

        translation = translate_text(text, CLIENT)

        return {"translation": translation}
    except Exception as e:
        print(f"[ERROR] Text translation failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to translate text.")


@app.post("/rag_prompt/")
async def rag(payload: dict):
    """Create a new prompt with RAG and return enhanced answer."""

    try:
        prompt = payload["prompt"]
        num_docs = payload["num_docs"]
        pages_data = payload["pages_data"]

        if not prompt:
            raise HTTPException(status_code=400, detail="No prompt found.")

        rel_docs = rag_helper.retrieve_relevant_docs(prompt, num_docs)
        ans, docs = rag_prompt(prompt, rel_docs, pages_data, CLIENT)

        return {"ans": ans, "docs": docs}
    except Exception as e:
        print(f"[ERROR] RAG prompt failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to RAG prompt.")
