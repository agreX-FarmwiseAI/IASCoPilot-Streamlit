import streamlit as st
from PIL import Image as PILImage
import base64
from io import BytesIO

# Use the same import and initialization for your Bedrock and PGVector setup as provided above
import os
import json
import boto3
from langchain_community.embeddings import BedrockEmbeddings
from langchain.vectorstores.pgvector import PGVector

# Bedrock client setup
bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name="ap-south-1",
    aws_access_key_id=st.secrets["aws_access_key_id"],
    aws_secret_access_key=st.secrets["aws_secret_access_key"],
)

model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
model_kwargs = {
    "max_tokens": 200000,
    "temperature": 0
}

# Database connection string
CONNECTION_STRING = PGVector.connection_string_from_db_params(
    driver="psycopg2",
    user=st.secrets["db_username"],
    password=st.secrets["db_password"],
    host=st.secrets["host_name"],
    port="5432",
    database="postgres"
)

st.set_page_config(page_title="AI Chat with Image Retrieval", layout="centered")

# List of available collection names
collection_names = ["Minutes of Meeting", "Disaster Management"]

# Sidebar dropdown to select collection name
selected_collection = st.sidebar.selectbox("Select Topic", collection_names)

# Set up the vector store
vectorstore = PGVector(
    embedding_function=BedrockEmbeddings(model_id="amazon.titan-embed-image-v1", client=bedrock_client),
    connection_string=CONNECTION_STRING,
    collection_name=selected_collection,  # Use selected collection name
    use_jsonb=True,
)

# Function to invoke Bedrock Model
def invoke_bedrock_model(context, question):
    prompt = f"""
You are a RAG based Chatbot. Your responses should be short and precise. You responses should imitate a normal human.
Answer the question based only on the following context, which can include text and tables retrieved from vector DB.
Do not reveal to the user that you are answering from the 'Context' that was provided to you. 
The 'Context' provided to you also contains description of images. So do not mention about not having access to images. Just answer. Do not complain.
Context: {context}
Question: {question}
"""

    request_payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 200000,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    }

    request_body = json.dumps(request_payload).encode('utf-8')

    try:
        response = bedrock_client.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=request_body
        )

        response_data = response['body'].read().decode('utf-8')
        response_json = json.loads(response_data)

        summary_text = ""
        for content in response_json.get('content', []):
            if content['type'] == 'text':
                summary_text += content['text'] + "\n"

        return summary_text.strip()

    except Exception as e:
        print("Error invoking model:", e)
        return None

# Updated answer function to take context as a parameter (text only)
def answer(question, context=""):
    relevant_docs = vectorstore.similarity_search(question)
    relevant_images = []
    for d in relevant_docs:
        if d.metadata['type'] == 'text':
            context += '[text]' + d.metadata['original_content'] + '\n'
        elif d.metadata['type'] == 'table':
            context += '[table]' + d.metadata['original_content'] + '\n'
        elif d.metadata['type'] == 'image':
            # Collect images for display later
            relevant_images.append(d.metadata['original_content'])

    result = invoke_bedrock_model(context, question)
    return result, relevant_images

# Function to decode and display images from Base64 strings
def display_images(image_list):
    images = []
    for image_str in image_list:
        try:
            image_bytes = base64.b64decode(image_str)
            image = PILImage.open(BytesIO(image_bytes))
            images.append(image)
        except Exception as e:
            print(f"Error decoding image: {e}")
    return images

# Streamlit UI
st.title("IAS CoPilot")

if 'messages' not in st.session_state:
    st.session_state.messages = []

# User input text box using st.chat_input
user_message = st.chat_input("Type your question here:")

# Function to handle user input with context truncation (text only)
if user_message:
    # Add user message to the session state
    st.session_state.messages.append({"role": "user", "content": user_message})

    # Create a truncated context from the last 5 message pairs (excluding images)
    context_messages = st.session_state.messages[-10:]  # Get last 5 user/AI pairs
    context = ""
    for message in context_messages:
        if message["role"] in ["user", "assistant"]:
            role = "User" if message['role'] == "user" else "AI"
            context += f"{role}: {message['content']}\n"

    # Get AI response and relevant images with truncated context
    ai_response, images = answer(user_message, context=context)
    st.session_state.messages.append({"role": "assistant", "content": ai_response, "images": images})
                
# Display chat messages using st.chat_message and image display
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(f"**You:** {message['content']}")
    else:
        with st.chat_message("assistant"):
            st.markdown(f"**AI:** {message['content']}")
            
            # Display retrieved images at the end of each AI response
            if "images" in message and message["images"]:
                images = display_images(message["images"])
                num_images = len(images)
                
                if num_images > 0:
                    # Create columns dynamically based on the number of images
                    cols = st.columns(num_images)
                    
                    # Display each image in its own column
                    for col, img in zip(cols, images):
                        with col:
                            st.image(img)
