# 🤖 Integration Examples

## Real-World Integration Scenarios

Learn how to integrate Local LLMs into various applications through practical examples. From Discord bots to REST APIs, this guide provides complete, production-ready implementations.

## Table of Contents

- [Discord Bot](#discord-bot)
- [REST API](#rest-api)
- [Telegram Bot](#telegram-bot)
- [Web Application](#web-application)
- [CLI Tool](#cli-tool)

## Discord Bot

```python
import discord
from discord.ext import commands
from transformers import AutoModelForCausalLM, AutoTokenizer

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

# Load model
tokenizer = AutoTokenizer.from_pretrained('mistral-7b')
model = AutoModelForCausalLM.from_pretrained('mistral-7b')

@bot.event
async def on_ready():
    print(f'{bot.user} is now running!')

@bot.command(name='ask')
async def ask(ctx, *, question):
    async with ctx.typing():
        inputs = tokenizer(question, return_tensors='pt')
        outputs = model.generate(**inputs, max_length=200)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Discord has message limit
        if len(response) > 2000:
            response = response[:1997] + '...'
        
        await ctx.send(response)

bot.run('YOUR_TOKEN')
```

## REST API

```python
from fastapi import FastAPI
from pydantic import BaseModel
import asyncio

app = FastAPI()

class QueryRequest(BaseModel):
    prompt: str
    max_tokens: int = 100

@app.post('/api/generate')
async def generate(request: QueryRequest):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        generate_text,
        request.prompt,
        request.max_tokens
    )
    return {'response': result}

def generate_text(prompt: str, max_tokens: int):
    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = model.generate(**inputs, max_length=max_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Run: uvicorn app:app --reload
```

## Telegram Bot

```python
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import logging

logging.basicConfig(level=logging.INFO)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        'Hi! Send me any message and I\'ll respond with AI.'
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text
    
    # Show typing indicator
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action='typing'
    )
    
    # Generate response
    inputs = tokenizer(user_message, return_tensors='pt')
    outputs = model.generate(**inputs, max_length=150)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    await update.message.reply_text(response)

def main():
    app = Application.builder().token('YOUR_TOKEN').build()
    
    app.add_handler(CommandHandler('start', start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    app.run_polling()

if __name__ == '__main__':
    main()
```

## Web Application

```html
<!DOCTYPE html>
<html>
<head>
    <title>LLM Chat</title>
    <style>
        .chat-container { max-width: 800px; margin: 0 auto; }
        .message { padding: 10px; margin: 5px 0; border-radius: 5px; }
        .user { background-color: #007bff; color: white; }
        .assistant { background-color: #e9ecef; }
    </style>
</head>
<body>
    <div class="chat-container">
        <div id="messages"></div>
        <input type="text" id="input" placeholder="Ask something..." />
        <button onclick="sendMessage()">Send</button>
    </div>

    <script>
        async function sendMessage() {
            const input = document.getElementById('input');
            const prompt = input.value;
            
            if (!prompt.trim()) return;
            
            // Add user message
            addMessage(prompt, 'user');
            input.value = '';
            
            // Get response
            const response = await fetch('/api/generate', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({prompt: prompt, max_tokens: 200})
            });
            
            const data = await response.json();
            addMessage(data.response, 'assistant');
        }
        
        function addMessage(text, sender) {
            const div = document.createElement('div');
            div.className = `message ${sender}`;
            div.textContent = text;
            document.getElementById('messages').appendChild(div);
        }
    </script>
</body>
</html>
```

## CLI Tool

```python
import click
from transformers import AutoModelForCausalLM, AutoTokenizer

@click.group()
def cli():
    pass

@cli.command()
@click.option('--model', default='mistral-7b', help='Model to use')
def chat(model):
    tokenizer = AutoTokenizer.from_pretrained(model)
    llm_model = AutoModelForCausalLM.from_pretrained(model)
    
    click.echo('Interactive mode. Type "exit" to quit.')
    while True:
        prompt = click.prompt('You')
        if prompt.lower() == 'exit':
            break
        
        inputs = tokenizer(prompt, return_tensors='pt')
        outputs = llm_model.generate(**inputs, max_length=200)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        click.echo(f'AI: {response}\n')

@cli.command()
@click.argument('prompt')
@click.option('--model', default='mistral-7b')
def ask(prompt, model):
    tokenizer = AutoTokenizer.from_pretrained(model)
    llm_model = AutoModelForCausalLM.from_pretrained(model)
    
    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = llm_model.generate(**inputs, max_length=200)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    click.echo(response)

if __name__ == '__main__':
    cli()

# Usage: python app.py chat
#        python app.py ask "What is AI?"
```

## Common Integration Patterns

### Error Handling

```python
try:
    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = model.generate(**inputs, max_length=200)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
except Exception as e:
    logger.error(f"Generation failed: {e}")
    response = "Sorry, I encountered an error."
```

### Timeout Handling

```python
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("Generation timeout")

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(30)  # 30 seconds timeout

try:
    outputs = model.generate(**inputs, max_length=200)
finally:
    signal.alarm(0)  # Cancel alarm
```

### Rate Limiting

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post('/api/generate')
@limiter.limit("10/minute")
async def generate(request: QueryRequest):
    # ...
```

---

**Last Updated**: December 2025
**Status**: Active Development
