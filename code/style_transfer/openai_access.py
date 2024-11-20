import openai
import time
import requests
from openai import OpenAI

API_KEY = 'your_api_key'
base_url = 'youe_base_url'
client = OpenAI(
    api_key=API_KEY,
    base_url=base_url
)

def get_oai_completion(prompt,model):
    model_name = "gpt-4"
    if model == "gpt-3.5":
        model_name = "gpt-3.5-turbo"
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},

                ],
            temperature=1,
            max_tokens=2048,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None
        )
        # res = response["choices"][0]["message"]["content"]
        res = response.choices[0].message.content

        gpt_output = res
        return gpt_output
    except requests.exceptions.Timeout:
        # Handle the timeout error here
        print("The OpenAI API request timed out. Please try again later.")
        return None
    except openai.InternalServerErrorr as e:
        # Handle the invalid request error here
        print(f"The OpenAI API request was invalid: {e}")
        return None
    except openai.APIError as e:
        if "The operation was timeout" in str(e):
            # Handle the timeout error here
            print("The OpenAI API request timed out. Please try again later.")
#             time.sleep(3)
            return get_oai_completion(prompt)
        else:
            # Handle other API errors here
            print(f"The OpenAI API returned an error: {e}")
            return None
    except openai.RateLimitError as e:
        return get_oai_completion(prompt)

def call_chatgpt(ins,model):
    success = False
    re_try_count = 15
    ans = ''
    while not success and re_try_count >= 0:
        re_try_count -= 1
        try:
            ans = get_oai_completion(ins,model)
            success = True
            # print('success...')
        except:
            time.sleep(5)
            print('retry...')
    return ans

if __name__ == '__main__':
    prompt = 'hello'
    test = get_oai_completion(prompt)
    print(test)