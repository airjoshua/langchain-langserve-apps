from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from sales_chatbot import sales_chat

load_dotenv()

app = FastAPI()


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


add_routes(
    app,
    sales_chat.chain,
    path="/sales_bot",
    playground_type="default",
    enable_feedback_endpoint=True,
    enable_public_trace_link_endpoint=True,
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

# gcloud run deploy [your-service-name] --source . --port 8001 --allow-unauthenticated --region us-central1 --set-env-vars=OPENAI_API_KEY=your_key
#gcloud run deploy gen-ai-app --source . --port 8001 --allow-unauthenticated --region us-west1 --set-env-vars=OPENAI_API_KEY=your_key
