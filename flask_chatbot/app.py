from flask import Flask, request, render_template, jsonify
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

app = Flask(__name__)

# Load the trained model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("./chatbot_model")
model = GPT2LMHeadModel.from_pretrained("./chatbot_model")

# Chatbot route
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get("input")
    inputs = tokenizer.encode(user_input, return_tensors="pt")

    # Generate response
    with torch.no_grad():
        outputs = model.generate(inputs, max_length=50, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({"response": response})

# Homepage route
@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
