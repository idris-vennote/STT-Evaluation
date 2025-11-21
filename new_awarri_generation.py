import os
import json
import time
import base64
import logging
import requests
from pathlib import Path
from typing import Dict, Any, Tuple
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NewAwarriTranscriptionProcessor:
    def __init__(self):
        # Initialize API credentials
        self.awarri_new_api_url = os.getenv("AWARRI_NEW_API_URL")
        self.awarri_new_api_key = os.getenv("AWARRI_NEW_API_KEY")
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Ground truth data
        self.ground_truth = {
            "short": [
                "How much airtime is left in my account?",
                "Can you check my current data balance now?",
                "What's the status of my MTN account today?",
                "I need to contact MTN customer support now.",
                "How do I buy 1GB of data today?"
            ],
            "medium": [
                "How can I recharge my MTN phone with airtime quickly?",
                "What's the cheapest data bundle available for a week?",
                "Can I transfer money to another MTN number easily?",
                "Why is my network signal so weak in this area?",
                "Please check my MTN bill for the last month."
            ],
            "long": [
                "I've been overcharged for international calls I didn't make; can you investigate and process a refund for me please?",
                "I want to upgrade my plan to include more data and unlimited calls; what's the best option for heavy users?",
                "My phone isn't receiving texts from other networks despite restarting twice; what steps can I take to fix this?",
                "Can you explain the roaming charges from my Kenya trip last week and how to avoid them next time?",
                "I'm setting up a new MTN SIM for my business; what bundles are best for high-volume SMS and calls?"
            ]
        }
        
        # Language mapping for new Awarri API
        self.awarri_language_map = {
            "yoruba": "yoruba",
            "igbo": "igbo",
            "hausa": "hausa",
            "english": "english"
        }
        
        # Weights for better model calculation
        self.accuracy_weight = 0.7
        self.latency_weight = 0.3

    def encode_audio_to_base64(self, file_path: str) -> str:
        """Read and encode audio file to base64 with required prefix."""
        try:
            with open(file_path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
            return f"data:audio/webm;base64,{encoded}"
        except Exception as e:
            logger.error(f"Failed to encode audio file {file_path}: {str(e)}")
            raise

    def transcribe_with_new_awarri(self, audio_file_path: str, language: str) -> Tuple[str, float]:
        """Transcribe audio using new Awarri API and measure latency."""
        try:
            start_time = time.time()
            
            base64_data = self.encode_audio_to_base64(audio_file_path)
            
            payload = {
                "base64_data": base64_data,
                "lang": self.awarri_language_map.get(language, "english")
            }
            
            headers = {
                "accept": "application/json",
                "Content-Type": "application/json",
                "X-API-Key": self.awarri_new_api_key
            }
            
            response = requests.post(self.awarri_new_api_url, json=payload, headers=headers)
            response.raise_for_status()
            
            end_time = time.time()
            latency = end_time - start_time
            
            # Extract transcription from response
            response_data = response.json()
            transcription = response_data.get("text", "") or response_data.get("transcription", "") or str(response_data)
            
            logger.info(f"New Awarri transcription completed in {latency:.2f} seconds")
            return transcription, latency
            
        except requests.exceptions.HTTPError as e:
            logger.error(f"New Awarri API HTTP error: {str(e)}")
            logger.error(f"Status code: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return f"Error: {str(e)}", 0.0
        except Exception as e:
            logger.error(f"New Awarri transcription failed: {str(e)}")
            return f"Error: {str(e)}", 0.0

    def translate_with_gpt(self, text: str, source_language: str) -> str:
        """Translate text to English using GPT."""
        try:
            if text.startswith("Error:"):
                return text
            
            # If already English, return as is
            if source_language.lower() == "english":
                return text
                
            prompt = f"""Translate the following {source_language} text to English. 
            If the text is already in English or mostly English, return it as is.
            Only return the translation, no additional text or explanations.
            
            Text to translate: {text}"""
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a professional translator. Translate accurately and naturally."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.1
            )
            
            translation = response.choices[0].message.content.strip()
            return translation
            
        except Exception as e:
            logger.error(f"GPT translation failed: {str(e)}")
            return f"Translation error: {str(e)}"

    def score_accuracy_with_gpt(self, transcription: str, ground_truth: str) -> float:
        """Score transcription accuracy against ground truth using GPT."""
        try:
            if transcription.startswith("Error:") or transcription.startswith("Translation error:"):
                return 0.0
                
            prompt = f"""Compare these two texts and provide an accuracy score from 0-100 based on semantic similarity and meaning preservation.
            
            Consider:
            - Semantic meaning (most important)
            - Key information preservation
            - Overall message intent
            - Minor word variations should not heavily penalize
            
            Ground Truth: "{ground_truth}"
            Transcription: "{transcription}"
            
            Respond with only a number between 0 and 100 (e.g., 85.5). No additional text."""
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert evaluator of transcription accuracy. Provide precise numerical scores."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10,
                temperature=0.1
            )
            
            score_text = response.choices[0].message.content.strip()
            try:
                score = float(score_text)
                return max(0.0, min(100.0, score))
            except ValueError:
                logger.error(f"Invalid score format from GPT: {score_text}")
                return 0.0
                
        except Exception as e:
            logger.error(f"GPT scoring failed: {str(e)}")
            return 0.0

    def calculate_better_model_three_way(self, spitch_data: Dict, awarri_data: Dict, awarri_new_data: Dict) -> str:
        """Determine better model among three models based on weighted accuracy and latency."""
        # Get max latency for normalization
        latencies = [spitch_data['latency'], awarri_data['latency'], awarri_new_data['latency']]
        max_latency = max(latencies)
        
        if max_latency > 0:
            spitch_latency_score = (max_latency - spitch_data['latency']) / max_latency * 100
            awarri_latency_score = (max_latency - awarri_data['latency']) / max_latency * 100
            awarri_new_latency_score = (max_latency - awarri_new_data['latency']) / max_latency * 100
        else:
            spitch_latency_score = awarri_latency_score = awarri_new_latency_score = 50
        
        # Calculate weighted scores
        spitch_weighted = (spitch_data['accuracy'] * self.accuracy_weight + 
                          spitch_latency_score * self.latency_weight)
        awarri_weighted = (awarri_data['accuracy'] * self.accuracy_weight + 
                          awarri_latency_score * self.latency_weight)
        awarri_new_weighted = (awarri_new_data['accuracy'] * self.accuracy_weight + 
                              awarri_new_latency_score * self.latency_weight)
        
        # Find the best model
        scores = {
            'spitch': spitch_weighted,
            'awarri': awarri_weighted,
            'awarri_new': awarri_new_weighted
        }
        
        best_model = max(scores, key=scores.get)
        
        # Check for ties (within 1% threshold)
        best_score = scores[best_model]
        tied_models = [model for model, score in scores.items() if abs(score - best_score) < 1.0]
        
        if len(tied_models) > 1:
            return "tie"
        
        return best_model

    def process_audio_file(self, audio_file_path: str, language: str, duration: str, 
                          file_number: int, existing_data: Dict) -> Dict[str, Any]:
        """Process a single audio file and add new Awarri data to existing structure."""
        logger.info(f"Processing: {audio_file_path}")
        
        ground_truth = self.ground_truth[duration][file_number - 1]
        
        # Transcribe with new Awarri API
        awarri_new_transcription, awarri_new_latency = self.transcribe_with_new_awarri(audio_file_path, language)
        
        # Translate to English
        awarri_new_translation = self.translate_with_gpt(awarri_new_transcription, language)
        
        # Score accuracy against ground truth
        awarri_new_accuracy = self.score_accuracy_with_gpt(awarri_new_translation, ground_truth)
        
        # Structure the new Awarri data
        awarri_new_data = {
            "transcription": awarri_new_transcription,
            "translation": awarri_new_translation,
            "latency": round(awarri_new_latency, 2),
            "accuracy": round(awarri_new_accuracy, 1)
        }
        
        # Add to existing data structure
        result = existing_data.copy()
        result["awarri_new"] = awarri_new_data
        
        # Recalculate better_model with all three models
        if 'spitch' in result and 'awarri' in result:
            result['better_model'] = self.calculate_better_model_three_way(
                result['spitch'], 
                result['awarri'], 
                awarri_new_data
            )
        
        return result

    def process_all_files(self, base_path: str = "audio_files", 
                         existing_file: str = "transcription_results.json") -> Dict[str, Any]:
        """Process all audio files and add new Awarri data to existing results."""
        
        # Load existing data
        try:
            with open(existing_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            logger.info(f"Loaded existing data from {existing_file}")
        except FileNotFoundError:
            logger.error(f"Existing file {existing_file} not found!")
            return {}
        except json.JSONDecodeError:
            logger.error(f"Error reading {existing_file}")
            return {}
        
        languages = ["yoruba", "igbo", "hausa", "english"]
        durations = ["short", "medium", "long"]
        
        for language in languages:
            if language not in results:
                logger.warning(f"Language {language} not found in existing data, skipping...")
                continue
            
            for duration in durations:
                for i in range(1, 6):  # Files numbered 1-5
                    file_name = f"{language}_{duration}_{i}.wav"
                    file_path = os.path.join(base_path, language, duration, file_name)
                    
                    if file_name not in results[language]:
                        logger.warning(f"File {file_name} not found in existing data, skipping...")
                        continue
                    
                    if os.path.exists(file_path):
                        try:
                            # Get existing data for this file
                            existing_file_data = results[language][file_name]
                            
                            # Process and add new Awarri data
                            updated_file_data = self.process_audio_file(
                                file_path, language, duration, i, existing_file_data
                            )
                            
                            # Update results
                            results[language][file_name] = updated_file_data
                            
                        except Exception as e:
                            logger.error(f"Failed to process {file_path}: {str(e)}")
                            # Add error entry for awarri_new only
                            results[language][file_name]["awarri_new"] = {
                                "transcription": f"Error: {str(e)}", 
                                "translation": "", 
                                "latency": 0.0, 
                                "accuracy": 0.0
                            }
                    else:
                        logger.warning(f"File not found: {file_path}")
        
        return results

    def save_results(self, results: Dict[str, Any], output_file: str = "transcription_results.json"):
        """Save results to JSON file."""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save results: {str(e)}")

def main():
    """Main function to run the new Awarri transcription processing."""
    processor = NewAwarriTranscriptionProcessor()
    
    logger.info("Starting new Awarri API transcription processing...")
    logger.info("Loading existing data and adding new Awarri results...")
    
    results = processor.process_all_files()
    
    if results:
        logger.info("Saving updated results...")
        processor.save_results(results)
        
        logger.info("Processing completed!")
        
        # Print summary statistics
        total_files = sum(len(lang_data) for lang_data in results.values())
        files_with_new_awarri = sum(
            1 for lang_data in results.values() 
            for file_data in lang_data.values() 
            if 'awarri_new' in file_data
        )
        logger.info(f"Added new Awarri data to {files_with_new_awarri} files out of {total_files} total files")
    else:
        logger.error("Failed to process files")

if __name__ == "__main__":
    main()