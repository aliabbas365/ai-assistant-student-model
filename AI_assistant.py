from transformers import AutoTokenizer, AutoModelForCausalLM
import pickle

# Load your trained ML model
with open("student_model.pkl", "rb") as f:
    student_model = pickle.load(f)



model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

print("AI Assistant Ready")

system_prompt = """
You are an AI assistant running on a local computer.
Answer the user's question clearly and stay on topic.
If the user asks who you are, say you are a local AI assistant.
"""

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    # ML model check
    if "predict" in user_input.lower():
        gpa = float(input("Enter GPA: "))
        python_skill = int(input("Python skill (0-10): "))
        prediction = student_model.predict([[gpa, python_skill]])
        print(f"AI (ML Model): {prediction[0]}")
        continue

    # Hugging Face AI
    prompt = system_prompt + "\nUser: " + user_input + "\nAssistant:"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.7,
        do_sample=True
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("Assistant:")[-1].strip()
    print("AI:", response)