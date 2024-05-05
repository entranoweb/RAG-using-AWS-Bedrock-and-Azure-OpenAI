from fastapi import FastAPI, HTTPException, status, Query
from fastapi.responses import JSONResponse, RedirectResponse
import boto3
import json
import os

from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

lambda_client = boto3.client(
    'lambda',
    region_name=os.getenv('REGION_NAME'),
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
)

def get_context(question: str):
    try:
        response = lambda_client.invoke(
            FunctionName='demolamda',
            InvocationType='RequestResponse',
            Payload=json.dumps({"question": question})
        )
        response_payload = response['Payload'].read().decode('utf-8')
        response_payload_dict = json.loads(response_payload)
        
        # Check if the status code in the Lambda response is 200
        if response_payload_dict.get('statusCode') == 200:
            body = response_payload_dict.get('body')
            if not body:
                print("No body in the response")
                return None
            
            # Make sure 'retrievalResults' is in the body
            results = body.get('retrievalResults')
            if not results:
                print("No retrievalResults found")
                return None
            
            # Extract text from each result and concatenate
            extracted_paragraph = " ".join(result['content']['text'] for result in results if result.get('content'))
            return extracted_paragraph.strip()
        else:
            print("Lambda function did not return statusCode 200")
            return None

    except Exception as e:
        print(f"Error retrieving context: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get context from lambda: {str(e)}")

def get_answer_from_kb(query: str):
    try:
        llm = AzureChatOpenAI(
            openai_api_version="2024-02-15-preview",
            openai_api_key=os.getenv('AZURE_OPENAI_API_KEY'),
            azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
            azure_deployment="gpt-35-turbo-16k",
            max_tokens=10000,
            temperature=0.4
        )
        kb_prompt_template = """
        You are a helpful AI assistant who is expert in answering questions. Your task is to answer user's questions as factually as possible. You will be given enough context with information to answer the user's questions. Find the context:
        Context: {context}
        Question: {query}

        Now generate a detailed answer that will be helpful for the user. Return the helpful answer.

        Answer: 
        """
        prompt_template_kb = PromptTemplate(
            input_variables=["context", "query"], template=kb_prompt_template
        )
        context = get_context(query)
        if context is None:
            return {"answer": "Failed to retrieve context for the provided query."}

        llm_chain = LLMChain(llm=llm, prompt=prompt_template_kb)
        result = llm_chain.run({"context": context, "query": query})
        return result

    except Exception as e:
        print(f"Error during answer generation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during answer generation: {str(e)}")

@app.post("/chat_with_knowledge_base")
def chat_with_knowledge_base(query: str = Query(...)):
    try:
        response = get_answer_from_kb(query)
        return JSONResponse(content=response, status_code=200)
    except HTTPException as e:
        return JSONResponse(content={"error": str(e.detail)}, status_code=e.status_code)
