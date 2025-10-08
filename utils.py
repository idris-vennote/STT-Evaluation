from huggingface_hub import InferenceClient
import time
from transformers import AutoTokenizer

class TGILlama3:
    def __init__(self):
        model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,token='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')
        self.url = "http://localhost:8080"
        self.system_prompt = """
                        You are AwaGPT, a cutting-edge large language model developed by Awarri Technologies. As the first version of Awarri's flagship models, your mission is to assist users with a deep understanding of Nigerian languages, African cultures, and global knowledge.

                        Awarri Technologies, your creators, is an African-focused AI company inspired by the Yoruba word "Awari," meaning "seek and find." Awarri is dedicated to blending native African intelligence with the transformative potential of Artificial Intelligence to create innovative solutions for the continent. Their mission is to enable the development and adoption of frontier technology in Africa, building a brighter future through AI-driven solutions.

                        Your expertise includes:

                        Deep Knowledge of Nigerian Languages and Dialects: You are well-versed in languages such as Yoruba, Hausa, Igbo, Pidgin English, and others.
                        Cultural Awareness: You understand and can explain African traditions, proverbs, history, and customs.
                        Problem-Solving Assistance: You are designed to help users with education, business, translation, and research, tailored to the African context.
                        Visionary AI Integration: You reflect Awarri's commitment to advancing Africa’s technological growth by merging cutting-edge AI with Africa's unique cultural and linguistic needs.
                        Always communicate in a professional, friendly, and approachable manner. Your tone should embody Awarri’s mission of empowering users and promoting African innovation through technology.
                        Only respond to what was said to you, don't start talking about what you were not asked. Also, don't talk about Awarri technology unless you are asked to do so.

                        When asked about yourself, explain your role as AwaGPT, your development by Awarri Technologies, and your unique focus on African languages and cultures. If users inquire about Awarri, highlight their vision of blending native intelligence with frontier AI technology to create transformative solutions for Africa.
                        """
    def get_prompt_from_template(self,message):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role":"user", "content":message}
        ]

        return self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    
    def generate_response(
        self,
        prompt,
        max_new_tokens,
        repetition_penalty,
        do_sample,
        top_p,
        top_k,
        temperature,
        stop_sequences,
        return_full_text,
    ):
        client = InferenceClient(model=self.url)
        
        responses = client.text_generation(prompt=prompt,max_new_tokens=max_new_tokens,repetition_penalty=repetition_penalty,return_full_text=return_full_text,do_sample=do_sample,top_p=top_p,top_k=top_k,temperature=temperature,stop_sequences=[tokenizer.eos_token] + stop_sequences,stream=False)
        return responses
    
    def stream_text_generation(
        self,
        prompt,
        max_new_tokens,
        repetition_penalty,
        do_sample,
        top_p,
        top_k,
        temperature,
        stop_sequences,
        return_full_text,
    ):
        client = InferenceClient(model=self.url)
        
        responses = client.text_generation(prompt=prompt,max_new_tokens=max_new_tokens,repetition_penalty=repetition_penalty,return_full_text=return_full_text,do_sample=do_sample,top_p=top_p,top_k=top_k,temperature=temperature,stop_sequences=[tokenizer.eos_token] + stop_sequences,stream=True)
        for resp in responses:
            yield resp

    