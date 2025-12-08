import sentencepiece
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def chat_with_bot():
    while True:
        # Get user input
        input_text = input("You: ")

        # Exit conditions
        if input_text.lower() in ["quit", "exit", "bye"]:
            print("Chatbot: Goodbye!")
            break

        # Tokenize input and generate response
        inputs = tokenizer.encode(input_text, return_tensors="pt")
        outputs = model.generate(inputs, max_new_tokens=150) 
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        
        # Display bot's response
        print("Chatbot:", response)

# Start chatting
chat_with_bot()