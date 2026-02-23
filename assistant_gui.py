import tkinter as tk
from tkinter import scrolledtext
import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM
import threading

# -----------------------------
# Load ML model
# -----------------------------
with open("student_model.pkl", "rb") as f:
    student_model = pickle.load(f)

# -----------------------------
# Load Hugging Face AI
# -----------------------------
model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

system_prompt = """
You are an AI assistant running on a local computer.
Answer the user's question clearly and stay on topic.
If the user asks who you are, say you are a local AI assistant.
"""

# -----------------------------
# GUI Setup
# -----------------------------
root = tk.Tk()
root.title("AI Assistant")
root.geometry("600x500")

chat_window = scrolledtext.ScrolledText(root, state='disabled', wrap=tk.WORD)
chat_window.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

input_box = tk.Entry(root, font=("Arial", 12))
input_box.pack(padx=10, pady=5, fill=tk.X)
input_box.focus()

# Variable to track if we're waiting for prediction input
await_prediction = False
prediction_state = {"gpa": None, "python_skill": None}

# -----------------------------
# Function to handle message in a thread
# -----------------------------
def handle_message(user_input):
    global root, await_prediction, prediction_state
    response = None
    
    try:
        # ML prediction - check if we have both values
        if await_prediction and prediction_state["gpa"] is not None and prediction_state["python_skill"] is not None:
            try:
                prediction = student_model.predict([[prediction_state["gpa"], prediction_state["python_skill"]]])
                response = f"ML Model predicts: {prediction[0]}"
                await_prediction = False
                prediction_state = {"gpa": None, "python_skill": None}
            except Exception as e:
                response = f"Error in prediction: {e}"
                await_prediction = False
                prediction_state = {"gpa": None, "python_skill": None}
        # Request prediction data
        elif "predict" in user_input.lower():
            response = "Please enter your GPA (as a decimal):"
            await_prediction = True
            prediction_state["gpa"] = None
            prediction_state["python_skill"] = None
        # Store GPA and ask for python skill
        elif await_prediction and prediction_state["gpa"] is None:
            try:
                prediction_state["gpa"] = float(user_input.strip())
                response = "Now enter your Python skill level (0-10):"
            except ValueError:
                response = "Invalid GPA. Please enter a valid number."
        # Store python skill and make prediction
        elif await_prediction and prediction_state["python_skill"] is None:
            try:
                prediction_state["python_skill"] = int(user_input.strip())
                # Make prediction
                prediction = student_model.predict([[prediction_state["gpa"], prediction_state["python_skill"]]])
                response = f"ML Model predicts: {prediction[0]}"
                await_prediction = False
                prediction_state = {"gpa": None, "python_skill": None}
            except ValueError:
                response = "Invalid skill level. Please enter a number between 0-10."
            except Exception as e:
                response = f"Error: {e}"
                await_prediction = False
        else:
            # LLM response
            prompt = system_prompt + "\nUser: " + user_input + "\nAssistant:"
            inputs = tokenizer(prompt, return_tensors="pt")
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True).split("Assistant:")[-1].strip()

        # Display AI response (schedule on main thread)
        if response and root.winfo_exists():
            root.after(0, lambda resp=response: update_chat(f"AI: {resp}\n\n"))
    except tk.TclError:
        pass  # Window was deleted, ignore

def update_chat(message):
    """Update chat window safely from main thread"""
    try:
        if root.winfo_exists():
            chat_window.config(state='normal')
            chat_window.insert(tk.END, message)
            chat_window.config(state='disabled')
            chat_window.see(tk.END)
    except tk.TclError:
        pass  # Window was deleted, ignore

# -----------------------------
# Function to send message (starts a thread)
# -----------------------------
def send_message(event=None):
    user_input = input_box.get().strip()
    if not user_input:
        return
    input_box.delete(0, tk.END)

    # Display user message
    chat_window.config(state='normal')
    chat_window.insert(tk.END, f"You: {user_input}\n")
    chat_window.config(state='disabled')
    chat_window.see(tk.END)

    # Start thread for AI response
    thread = threading.Thread(target=handle_message, args=(user_input,))
    thread.start()

# Bind Enter key
input_box.bind("<Return>", send_message)

# Add window close handler to prevent errors
def on_closing():
    try:
        root.destroy()
    except:
        pass

root.protocol("WM_DELETE_WINDOW", on_closing)

# -----------------------------
# Initial message
# -----------------------------
chat_window.config(state='normal')
chat_window.insert(tk.END, "AI Assistant Ready. Type your message below.\n\n")
chat_window.config(state='disabled')

root.mainloop()