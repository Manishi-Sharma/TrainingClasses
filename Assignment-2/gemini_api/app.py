from google import genai

# Connect to Gemini
client = genai.Client()

# Get input from user
question = input("You: ")

# Send question to Gemini
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=question
)

# Print Gemini's response
print("\nGemini:")
print(response.text)
