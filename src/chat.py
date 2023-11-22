def chat():
    print("Transformer Chatbot (type 'quit' to exit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break

        input_tokens = encode_input(user_input)
        output_tokens = transformer(input_tokens)
        response = decode_output(output_tokens)

        print("Bot:", response)

if __name__ == "__main__":
    chat()
