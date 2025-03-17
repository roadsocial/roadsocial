import os, json, ast, re
from openai import OpenAI

class LLMEval:
    def __init__(self, apikey_id, eval_model='gpt-3.5-turbo'):
        self.client = OpenAI(api_key=apikey_id)
        self.invcomma_pattern = r'("(score|explanation)": *")(.*?)(", *"|" *})'
        self.invcomma_pattern_ans = r'("(explanation)": *")(.*?)(" *})'
        self.eval_model = eval_model
    
    def call_llm(self, prompt, max_tokens=128):
        system_message = "You are an expert evaluator who rates the predicted answer based on the correct answer for a given question."
        messages = [{"role": "system", "content": system_message}]
        messages.append({"role": "user", "content": "{}".format(prompt)})
        
        response = self.client.chat.completions.create(model=self.eval_model, messages=messages, temperature=0.6, max_tokens=max_tokens)
        reply = response.choices[0].message.content
        total_tokens = response.usage.total_tokens
        return reply, total_tokens
    
    def replace_quotes(self, match):
        key = match.group(1)
        value = match.group(3).replace('"', "'").replace("\\\'", "'")
        return f'{key}{value}{match.group(4)}'

    def parse_score(self, review):
        try:
            # Convert the string representation of a dictionary to an actual dictionary
            review = review[review.find('{'): len(review)-review[::-1].find('}')]
            review = re.sub(self.invcomma_pattern, self.replace_quotes, review)
            review = re.sub(self.invcomma_pattern_ans, self.replace_quotes, review)
            
            review_dict = ast.literal_eval(review)
            score = review_dict.get("score", 0)
            explanation = review_dict.get("explanation", 0)
            return int(score), explanation
        except SyntaxError as e:
            print(f"Syntax error parsing the review string: {e}. Review content: {review}")
            return 0, ''
        except ValueError as e:
            print(f"Value error parsing the review string: {e}. Review content: {review}")
            return 0, ''
        except Exception as e:
            print(f"Unexpected error parsing the review string: {e}. Review content: {review}")
            return 0, ''
    
    def forward(self, data):
        question, GT, answer = data
        
        prompts = ( "Evaluate the following question-answer pair:\\n"
                    f"Question: {question}\\n"
                    f"Correct Answer: {GT}\\n"
                    f"Predicted Answer: {answer}\\n\\n"
                    "Rate the Predicted Answer based on the Correct Answer on a scale from 0 to 100, with higher scores indicating that the Predicted Answer is closer to the Correct Answer. Your rating should be accurate to single digits like 62, 78, 41, etc."
                    'Please generate the response in the form of a Python dictionary string with keys "score", where its value is in INTEGER, not STRING, and "explanation" giving short and concise reasoning behind the score.'
                    'For example, your response should look like this: {"score": 45, "explanation": "..."}')
        
        reply, total_tokens = self.call_llm(prompts, max_tokens=256)
        output, exp = self.parse_score(reply)
        return output, exp
        
    