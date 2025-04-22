from google import genai

client = genai.Client(api_key="AIzaSyBzXXA96y8HjMz9V8GVMLsDd-a7wzwa688")

response = client.models.generate_content(
    model="gemini-2.0-flash", contents="你的名字"
)
print(response.text)