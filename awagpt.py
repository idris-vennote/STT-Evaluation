import base64
import time
import requests
import mimetypes
from typing import Union, List
import json


def _convert_to_base64(file_path):
    """
    Converts an audio file to web-compatible base64 format.
    """
    # Get the MIME type of the file
    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type:
        raise ValueError("Unable to determine MIME type. Please provide a valid audio file.")

    with open(file_path, "rb") as audio_file:
        base64_audio = base64.b64encode(audio_file.read()).decode("utf-8")
    
    # Return the data in web-compatible base64 format
    return f"data:{mime_type};base64,{base64_audio}"



class AwaGPT:
    def __init__(self, api_key, chat_interface=False, system_instruction=None):
        self.api_key = api_key
        self.chat_interface = chat_interface
        self.system_instruction = system_instruction

    def generate_text(self, prompt: str = None):
        """
        Generate text using the API. Accepts a text prompt.

        Args:
            prompt (str): Text prompt for the API.

        Returns:
            str: Response text from the API.
        """
        if not prompt:
            raise ValueError("prompt must be provided.")

        url = "https://vennote-1001380162242.us-central1.run.app/generate_text"
        payload = {
            "api_key": self.api_key,
            "query": prompt,
            "chat_interface": str(self.chat_interface).lower(),
            "system_instruction": str(self.system_instruction)
        }

        attempt = 0
        backoff = 1
        while attempt <= 2:
            try:
                response = requests.post(url, json=payload)
                response.raise_for_status()  # Raise an HTTPError for bad responses
                return response.json()['response']
            except requests.exceptions.HTTPError as e:
                error_message = f"HTTP error occurred: {e.response.status_code} - {e.response.text}"
            except requests.exceptions.RequestException as e:
                error_message = f"Request failed: {e}"
            except KeyError:
                error_message = "Invalid response from API, 'response' key not found."
            except Exception as e:
                error_message = f"An unexpected error occurred: {e}"

            attempt += 1
            if attempt > 2:
                raise ValueError(f"All retry attempts failed. Last error: {error_message}")
            time.sleep(backoff)
            backoff *= 2 

    def stream_text(self, prompt: str = None):
        """
        Stream text using the API. Accepts a text prompt.

        Args:
            prompt (str): Text prompt for the API.

        Yields:
            str: Streamed chunks of response text from the API.
        """
        if not prompt:
            raise ValueError("prompt must be provided.")
        if str(self.chat_interface).lower() == 'true':
            prompt = json.dumps(prompt)

        url = "https://vennote-1001380162242.us-central1.run.app/generate_text"
        params = {
            "api_key": self.api_key,
            "query": prompt,
            "chat_interface": str(self.chat_interface).lower(),
            "system_instruction": str(self.system_instruction)
        }

        attempt = 0
        backoff = 1
        while attempt <= 2:
            try:
                with requests.get(url, params=params, stream=True) as response:
                    response.raise_for_status()  # Raise an HTTPError for bad responses
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            yield chunk.decode('utf-8')
                    return  # Exit after successful fetch
            except requests.exceptions.HTTPError as e:
                error_message = f"HTTP error occurred: {e.response.status_code} - {e.response.text}"
            except requests.exceptions.RequestException as e:
                error_message = f"Request failed: {e}"
            except Exception as e:
                error_message = f"An unexpected error occurred: {e}"

            attempt += 1
            if attempt > 2:
                raise ValueError(f"All retry attempts failed. Last error: {error_message}")
            time.sleep(backoff)
            backoff *= 2 
        
    def transcribe_audio(self, data, language: str = None):
        """
        Accepts base64 data or an audio file path (wav/mp3).
        Converts to base64 format if necessary, then sends to the endpoint.
        """
        
        # List of supported languages
        supported_languages = ['english', 'hausa', 'yoruba', 'igbo']

        # Check if the language is supported
        if language and language.lower() not in supported_languages:
            raise ValueError("The chosen language is not currently supported.")
    
        try:
            if isinstance(data, str) and data.startswith("data:audio/"):
                base64_data = data
            else:
                # Assume data is a file path and convert it
                base64_data = _convert_to_base64(data)
        except Exception as e:
            raise ValueError(f"Invalid input: {e}")

        url = "https://vennote-1001380162242.us-central1.run.app/transcribe_audio"
        payload = {
            "api_key": self.api_key,
            "base64_data": base64_data,
            "lang": language
        }

        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()  # Raise an exception for HTTP errors
            return response.json().get("audio_txt", "No transcription returned.")
        except requests.exceptions.HTTPError as e:
            raise ValueError(f"HTTP error occurred: {e.response.status_code} - {e.response.text}")
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Request failed: {e}")


    def synthesis_text(self, text, language: str = 'English', save_path: str = None):
        """
        Accepts base64 data or an audio file path (wav/mp3).
        Converts to base64 format if necessary, then sends to the endpoint.
        """
        
        # List of supported languages
        supported_languages = ['english', 'hausa', 'yoruba', 'igbo', 'yoruba_second_speaker', 'yoruba_third_speaker','hausa_second_speaker']

        # Check if the language is supported
        if language.lower() not in supported_languages:
            raise ValueError("The chosen language is not currently supported.")
    

        url = "https://vennote-1001380162242.us-central1.run.app/synthesis_text"
        payload = {
            "api_key": self.api_key,
            "audio_txt": text,
            "lang": language
        }

        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()  # Raise an exception for HTTP errors
            audio_base64 = response.json().get("base64_data", "No transcription returned.")
        except requests.exceptions.HTTPError as e:
            raise ValueError(f"HTTP error occurred: {e.response.status_code} - {e.response.text}")
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Request failed: {e}")


        if save_path:
            import base64
            import os
            from os.path import join, dirname

            try:
                # Decode the base64 audio data
                binary_audio = base64.b64decode(audio_base64)
            except Exception as decode_error:
                raise ValueError(f"Failed to decode base64 audio data: {decode_error}")

            # Check if the provided save_path is absolute. If not, construct it relative to the current working directory.
            if not os.path.isabs(save_path):
                cwd = os.getcwd()
                full_save_path = join(cwd, save_path)
            else:
                full_save_path = save_path

            # Ensure the directory exists; create if it doesn't
            os.makedirs(dirname(full_save_path), exist_ok=True)

            # Write the binary audio data to the file
            with open(full_save_path, "wb") as wav_file:
                wav_file.write(binary_audio)

            return full_save_path
        else:
            return audio_base64

    def translate_text(self, base_lang: str = None, target_lang: str = None, text_to_translate: str = None):
        """
        Translate text using the API.
        
        Args:
            base_lang (str): Base language code.
            target_lang (str): Target language code.
            text_to_translate (str): Text to be translated.

        Returns:
            str: Translated text from the API.
        """
        
        allowed_languages = {'english', 'yoruba', 'igbo', 'hausa'}

        # Check for missing arguments and validate language inputs
        missing_args = [arg for arg, name in [(base_lang, "base_lang"), 
                                            (target_lang, "target_lang"), 
                                            (text_to_translate, "text_to_translate")] if not arg]
        if missing_args:
            raise ValueError(f"Missing required arguments: {', '.join(name for _, name in missing_args)}")

        for lang, name in [(base_lang, "base_lang"), (target_lang, "target_lang")]:
            if lang.lower() not in allowed_languages:
                raise ValueError(f"Unsupported {name}: {lang}. Allowed languages are: {', '.join(allowed_languages)}")


        url = "https://vennote-1001380162242.us-central1.run.app/translate"
        payload = {
            "api_key": self.api_key,
            "base_lang": base_lang,
            "target_lang": target_lang,
            "text_to_translate": text_to_translate
        }

        attempt = 0
        backoff = 1
        while attempt <= 2:
            try:
                response = requests.post(url, json=payload)
                response.raise_for_status()  # Raise an exception for HTTP errors
                return response.json()['response']
            except requests.exceptions.HTTPError as e:
                error_message = f"HTTP error occurred: {e.response.status_code} - {e.response.text}"
            except requests.exceptions.RequestException as e:
                error_message = f"Request failed: {e}"
            except KeyError:
                error_message = "Invalid response from API, 'response' key not found."
            except Exception as e:
                error_message = f"An unexpected error occurred: {e}"

            attempt += 1
            if attempt > 2:
                raise ValueError(f"All retry attempts failed. Last error: {error_message}")
            time.sleep(backoff)
            backoff *= 2  # Exponential backoff
        

