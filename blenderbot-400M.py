from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "facebook/blenderbot-400M-distill"

def load_chat_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    return tokenizer, model


def clean_text(text: str) -> str:
    return text.strip()


class ChatPipeline:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def generate_response(self, user_input: str):
        user_input = clean_text(user_input)

        inputs = self.tokenizer.encode(user_input, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=150,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.strip()

def chat_with_bot():
    tokenizer, model = load_chat_model()
    chat_pipeline = ChatPipeline(tokenizer, model)

    print("Chatbot is ready! Type 'exit' to quit.\n")

    while True:
        user_msg = input("You: ")

        if user_msg.lower() in ["exit", "quit", "bye"]:
            print("Chatbot: Goodbye!")
            break

        bot_response = chat_pipeline.generate_response(user_msg)
        print("Chatbot:", bot_response)

if __name__ == "__main__":
    chat_with_bot()