
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import json
from process import process_pdf, process_image

# New imports for assessment
import os
from dotenv import load_dotenv
from openai import OpenAI

app = FastAPI()

# Enable CORS for all origins (for both REST API and WebSockets)
origins = [
    "http://localhost:5173",  # Your React app's frontend URL
    "http://localhost:3000",  # If you use another port, you can add that too
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows CORS from these origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allow all headers
)

# WebSocket endpoint to process the file
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        json_file_path = "invoice_result.json"
        with open(json_file_path, "r", encoding="utf-8") as f:
            file_data = json.load(f)
        # Receive the file from the frontend (simulated by the uploaded file path here)
        file_name = await websocket.receive_text() 
        uploaded_file = await websocket.receive_bytes()  # Get file in bytes
        

        # # Save the uploaded file
        #  # Generate a unique file name
        # file_path = os.path.join("uploads", file_name)
        # os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # with open(file_path, "wb") as f:
        #     f.write(uploaded_file)

        # Send the file format
        file_extension = file_name.split('.')[-1].lower()
        await websocket.send_text(json.dumps({"message" : f"File format: {file_extension.upper()}", "type" : "success", "finished" : True} ))

        # Send the file size
        # file_size = os.path.getsize(file_path)
        # await websocket.send_text(f"File size: {file_size} bytes")

        if file_extension == "pdf":
            await websocket.send_text("📄 **PDF file detected**")
            # await websocket.send_json({"result" : {"text" : file_data} })
            processing_result = await process_pdf(uploaded_file, websocket, file_name)
            
        else:
            #  await process_image(uploaded_file, websocket, file_name)
            await websocket.send_text("📄 **Image detected**")
            await process_image(uploaded_file, websocket, file_name)     
            # await websocket.send_json({"result" : {"text" : file_data} })
        # Simulate processing completion
        await websocket.send_text(json.dumps({"message" : "File uploaded and processed successfully!", "type" : "success"}))
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(e)

# New REST endpoint: Assess whether purchase request and OCR-extracted invoice match
@app.post("/assess")
async def assess_invoice_match(payload: dict = Body(...)):
    """
    Expects JSON body with:
    {
      "purchase": { ... purchase request fields incl. items ... },
      "invoice": { ... OCR extracted JSON ... }
    }
    Returns structured assessment with overall_match, score, discrepancies, and recommendation.
    """
    try:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        if not api_key:
            return JSONResponse(status_code=500, content={"message": "OPENAI_API_KEY not configured"})

        client = OpenAI(api_key=api_key)

        purchase = payload.get("purchase", {})
        invoice = payload.get("invoice", {})

        system_prompt = (
            "You are a strict auditor that compares an employee purchase request against an OCR-extracted supplier invoice. "
            "Return ONLY valid JSON with fields: overall_match (boolean), score (0-100), summary (string), "
            "discrepancies (array of {field, expected, found, note}), and recommendation ('approve' or 'reject'). "
            "Be conservative: any material mismatch in quantities, item descriptions, or pre-tax totals should reduce score and may lead to reject. "
            "IMPORTANT TOTALS POLICY: The purchase request 'amount' represents a pre-tax total and MUST be compared to the invoice pre-tax amount (subtotal). "
            "Do NOT penalize differences in taxes; focus on comparing purchase total to invoice subtotal. If the invoice does not explicitly provide a subtotal, "
            "approximate it by subtracting available tax fields (e.g., total_tax_amount, cgst, sgst, igst, cess) and other charges from the invoice total_amount. "
            "Treat deltas ≤ 1% as minor and acceptable. "
            "FUZZY ITEM MATCHING POLICY: Item names may differ between request and invoice (e.g., 'blue pen' vs 'cello smooth blue gel pen'). "
            "Use fuzzy/approximate matching – case-insensitive, token-based similarity. Ignore brand adjectives and extra descriptors. "
            "Consider two items a name match if the purchase item key tokens are contained in or strongly overlap with the invoice item (e.g., Jaccard/token overlap ≥ 0.5), "
            "or if they are clear synonyms. Use quantity and unit/line price proximity (≤ 2% delta) to reinforce mapping. "
            "Create best one-to-one mappings between purchase items and invoice items by maximizing combined name similarity and price proximity. "
            "Do NOT flag cosmetic naming differences as discrepancies when quantities and prices align within tolerance."
        )

        user_prompt = (
            "Compare these two JSON documents. Determine if they match on vendor (if present), items (fuzzy names, quantities, unit prices, line totals), and PRE-TAX total. "
            "Use invoice 'financial_summary.subtotal' as the invoice pre-tax amount. If missing, compute an approximate subtotal as described in the policy.\n\n"
            f"PURCHASE_REQUEST_JSON:\n{json.dumps(purchase, ensure_ascii=False)}\n\n"
            f"INVOICE_OCR_JSON:\n{json.dumps(invoice, ensure_ascii=False)}\n\n"
            "Respond with JSON only: {\n"
            "  \"overall_match\": true/false,\n"
            "  \"score\": number,\n"
            "  \"summary\": \"...\",\n"
            "  \"discrepancies\": [ { \"field\": \"...\", \"expected\": \"...\", \"found\": \"...\", \"note\": \"...\" } ],\n"
            "  \"recommendation\": \"approve\"|\"reject\"\n"
            "}"
        )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=900
        )

        content = response.choices[0].message.content.strip()
        # Try to load as JSON directly; if model wrapped in code fences, strip them
        if content.startswith('```'):
            content = content.split('\n', 1)[1]
            if content.endswith('```'):
                content = content[:-3]
        assessment = json.loads(content)
        return assessment
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})
