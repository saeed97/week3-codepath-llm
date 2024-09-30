from dotenv import load_dotenv
import chainlit as cl
from langfuse.decorators import observe
from langfuse.openai import AsyncOpenAI
from generate_response import get_now_playing_movies, get_showtimes, buy_ticket, get_reviews
import json 
load_dotenv()



 
client = AsyncOpenAI()

gen_kwargs = {
    "model": "gpt-4o",
    "temperature": 0.2,
    "max_tokens": 500
}

SYSTEM_PROMPT = """\
You are a helpful assistant that can sometimes answer a with a list of movies, or list of show times. If you
need a list of movies, generate a function call, as shown below.

If you encounter errors, report the issue to the user.

The following functions are the ones you have access to:
get_now_playing_movies()
get_showtimes(title, location)
buy_ticket(theater, movie, showtime)
confirm_ticket_purchase(theater, movie, showtime)

Some functions requires more info, like get_showtimes function requires title and location.
Then you need to request the user to provide the missing info first.
once you got the info, then you can call the function.

For the buy_ticket function, you must first call confirm_ticket_purchase to get user confirmation before proceeding with the purchase.

If decided to use function calling, then response as following:
DO not include any other text other than the JSON object for the function call.

{
    "function_name": "get_now_playing_movies",
    "rationale": "Explain why you are calling the function"
}
{
    "function_name": "get_showtimes",
    "title": "The movie title",
    "location": "The location of the movie",
    "rationale": "Explain why you are calling the function"
}
{
    "function_name": "confirm_ticket_purchase",
    "theater": "The theater name",
    "movie": "The movie title",
    "showtime": "The showtime",
    "rationale": "Explain why you are calling the function"
}
{
    "function_name": "buy_ticket",
    "theater": "The theater name",
    "movie": "The movie title",
    "showtime": "The showtime",
    "rationale": "Explain why you are calling the function"
}
"""

@observe
@cl.on_chat_start
async def on_chat_start():    
    message_history = [{"role": "system", "content": SYSTEM_PROMPT}]
    cl.user_session.set("message_history", message_history)

@observe
async def generate_response(client, message_history, gen_kwargs):
    response_message = cl.Message(content="")
    await response_message.send()

    stream = await client.chat.completions.create(messages=message_history, stream=True, **gen_kwargs)
    async for part in stream:
        if token := part.choices[0].delta.content or "":
            await response_message.stream_token(token)
    
    await response_message.update()

    return response_message

@observe
@cl.on_message
async def on_message(message: cl.Message):
    message_history = cl.user_session.get("message_history", [])
    message_history.append({"role": "user", "content": message.content})
    
    response_message = await generate_response(client, message_history, gen_kwargs)

    # Check if the response is a function call
    while response_message.content.strip().startswith('{'):
        try:
            # Parse the JSON object
            function_call = json.loads(response_message.content.strip())
            
            # Check if it's a valid function call
            if "function_name" in function_call and "rationale" in function_call:
                function_name = function_call["function_name"]
                rationale = function_call["rationale"]
                
                # Handle the function call
                if function_name == "get_now_playing_movies":
                    movies = get_now_playing_movies()
                    message_history.append({"role": "system", "content": f"Function call rationale: {rationale}\n\n{movies}"})
                    
                    # Generate a new response based on the function call result
                    response_message = await generate_response(client, message_history, gen_kwargs)
                elif function_name == "get_showtimes":
                    # Assuming get_showtimes requires title and location
                    showtimes = get_showtimes(function_call["title"], function_call["location"])
                    message_history.append({"role": "system", "content": f"Function call rationale: {rationale}\n\n{showtimes}"})
                    response_message = await generate_response(client, message_history, gen_kwargs)
                elif function_name == "confirm_ticket_purchase":
                    theater = function_call["theater"]
                    movie = function_call["movie"]
                    showtime = function_call["showtime"]
                    confirmation_message = f"Do you want to buy a ticket for {movie} at {theater} for {showtime}? (Yes/No)"
                    message_history.append({"role": "system", "content": f"Function call rationale: {rationale}\n\nPlease ask the user: {confirmation_message}"})
                    response_message = await generate_response(client, message_history, gen_kwargs)
                elif function_name == "buy_ticket":
                    theater = function_call["theater"]
                    movie = function_call["movie"]
                    showtime = function_call["showtime"]
                    ticket_info = buy_ticket(theater, movie, showtime)
                    message_history.append({"role": "system", "content": f"Function call rationale: {rationale}\n\n{ticket_info}"})
                    response_message = await generate_response(client, message_history, gen_kwargs)
                else:
                    # Handle unknown function calls
                    error_message = f"Unknown function: {function_name}"
                    message_history.append({"role": "system", "content": error_message})
                    response_message = await cl.Message(content=error_message).send()
            else:
                # Handle invalid function call format
                error_message = "Invalid function call format"
                message_history.append({"role": "system", "content": error_message})
                response_message = await cl.Message(content=error_message).send()
        except json.JSONDecodeError:
            # If it's not valid JSON, treat it as a normal message
            break

    message_history.append({"role": "assistant", "content": response_message.content})
    cl.user_session.set("message_history", message_history)

if __name__ == "__main__":
    cl.main()