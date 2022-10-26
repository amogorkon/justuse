import os
import openai

openai.api_key = "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"

start_sequence = "\nAI:"
restart_sequence = "\nHuman: "

response = openai.Completion.create(
  model="text-curie-001",
  prompt="The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, funny and fairly sarcastic.\n\nHuman: Hello, who are you?\nAI: I am an AI created by OpenAI. How can I help you today?\nHuman: what is the difference between a cow and a pig?\n",
  temperature=0.9,
  max_tokens=80,
  top_p=1,
  frequency_penalty=0.3,
  presence_penalty=0.6,
  stop=[" Human:", " AI:"]
)

print(response["choices"][0]["text"])