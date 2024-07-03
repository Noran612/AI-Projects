import openai
import os
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS

# Load environment variables from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

# Function to handle chatbot interaction
def ask_openai(prompt, model="ft:gpt-3.5-turbo-0125:personal::9gN5sEJr"):
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ]
       
    )
    return response.choices[0].message.content

user_info = {}


@app.route("/")
def home():
    return render_template("chatbot.html")

# Function to save contact information to CSV
def request_human_representative(name, email, phone):
    df = pd.DataFrame([[name, email, phone]], columns=["Name", "Email", "Phone"])
    df.to_csv("contact_information.csv", mode='a', header=not os.path.exists('contact_information.csv'), index=False)
    return "Your request has been received. A human representative will contact you soon."

conversation_context = {
    "awaiting_order_id": False,
    "awaiting_contact_info": False,
    "name": None,
    "email": None,
    "phone": None
}


# Main Chatbot Functionality
@app.route("/chat", methods=["POST"])
def chatbot():
    
    while True:
        #user_input = input("You: ")
        data = request.json
        user_input= data.get("message")
        
        if conversation_context["awaiting_contact_info"]:
            if conversation_context["name"] is None:
                conversation_context["name"] = user_input.strip()
                return "Please provide your email."
            elif conversation_context["email"] is None:
                conversation_context["email"] = user_input.strip()
                return "Please provide your phone number."
            elif conversation_context["phone"] is None:
                conversation_context["phone"] = user_input.strip()
                request_human_representative(conversation_context["name"], conversation_context["email"], conversation_context["phone"])
                conversation_context["awaiting_contact_info"] = False
                conversation_context["name"] = None
                conversation_context["email"] = None
                conversation_context["phone"] = None
                return "Thank you! Your information has been saved. A representative will contact you shortly."
        
        if "speak to human" in user_input.lower():
            conversation_context["awaiting_contact_info"] = True
            return "Please provide your full name"

        
        else:
            response = ask_openai(user_input)
            
            return response

        if user_input.lower() in ["exit", "quit", "bye"]:
            return "GoodBye!"
            break

if __name__ == "__main__":
    app.run(debug=True)
