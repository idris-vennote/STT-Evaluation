import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
from dotenv import load_dotenv
from openai import OpenAI
from spitch import Spitch
from awagpt import AwaGPT  # Assuming this is the correct import

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TranscriptionProcessor:
    def __init__(self):
        # Initialize API clients
        self.awarri_client = AwaGPT(os.getenv("AWARRI_API_KEY"))
        
        spitch_api_key = os.getenv("SPITCH_API_KEY")
        os.environ["SPITCH_API_KEY"] = spitch_api_key
        self.spitch_client = Spitch()
        
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
        
        # Language mappings
        self.spitch_language_map = {"yoruba": "yo", "igbo": "ig", "hausa": "ha"}
        
        # Weights for better model calculation (accuracy_weight + latency_weight should = 1.0)
        self.accuracy_weight = 0.7
        self.latency_weight = 0.3

    def transcribe_with_awarri(self, audio_file_path: str, language: str) -> Tuple[str, float]:
        """Transcribe audio using Awarri API and measure latency."""
        try:
            start_time = time.time()
            response = self.awarri_client.transcribe_audio(audio_file_path, language=language)
            end_time = time.time()
            latency = end_time - start_time
            
            # Extract text from response (adjust based on actual response structure)
            transcription = str(response) if response else ""
            
            logger.info(f"Awarri transcription completed in {latency:.2f} seconds")
            return transcription, latency
            
        except Exception as e:
            logger.error(f"Awarri transcription failed: {str(e)}")
            return f"Error: {str(e)}", 0.0

    def transcribe_with_spitch(self, audio_file_path: str, language: str) -> Tuple[str, float]:
        """Transcribe audio using Spitch API and measure latency."""
        try:
            start_time = time.time()
            
            with open(audio_file_path, "rb") as f:
                response = self.spitch_client.speech.transcribe(
                    language=self.spitch_language_map[language],
                    content=f.read()
                )
            
            end_time = time.time()
            latency = end_time - start_time
            
            transcription = response.text if hasattr(response, 'text') else str(response)
            
            logger.info(f"Spitch transcription completed in {latency:.2f} seconds")
            return transcription, latency
            
        except Exception as e:
            logger.error(f"Spitch transcription failed: {str(e)}")
            return f"Error: {str(e)}", 0.0

    def translate_with_gpt(self, text: str, source_language: str) -> str:
        """Translate text to English using GPT."""
        try:
            if text.startswith("Error:"):
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
                return max(0.0, min(100.0, score))  # Ensure score is between 0-100
            except ValueError:
                logger.error(f"Invalid score format from GPT: {score_text}")
                return 0.0
                
        except Exception as e:
            logger.error(f"GPT scoring failed: {str(e)}")
            return 0.0

    def calculate_better_model(self, awarri_data: Dict, spitch_data: Dict) -> str:
        """Determine better model based on weighted accuracy and latency."""
        # Normalize latency scores (lower latency is better, so invert)
        max_latency = max(awarri_data['latency'], spitch_data['latency'])
        if max_latency > 0:
            awarri_latency_score = (max_latency - awarri_data['latency']) / max_latency * 100
            spitch_latency_score = (max_latency - spitch_data['latency']) / max_latency * 100
        else:
            awarri_latency_score = spitch_latency_score = 50  # Default if both are 0
        
        # Calculate weighted scores
        awarri_weighted = (awarri_data['accuracy'] * self.accuracy_weight + 
                          awarri_latency_score * self.latency_weight)
        spitch_weighted = (spitch_data['accuracy'] * self.accuracy_weight + 
                          spitch_latency_score * self.latency_weight)
        
        if awarri_weighted > spitch_weighted:
            return "awarri"
        elif spitch_weighted > awarri_weighted:
            return "spitch"
        else:
            return "tie"

    def process_audio_file(self, audio_file_path: str, language: str, duration: str, file_number: int) -> Dict[str, Any]:
        """Process a single audio file and return complete data structure."""
        logger.info(f"Processing: {audio_file_path}")
        
        # Get ground truth for this file
        ground_truth = self.ground_truth[duration][file_number - 1]
        
        # Transcribe with both services
        awarri_transcription, awarri_latency = self.transcribe_with_awarri(audio_file_path, language)
        spitch_transcription, spitch_latency = self.transcribe_with_spitch(audio_file_path, language)
        
        # Translate transcriptions to English
        awarri_translation = self.translate_with_gpt(awarri_transcription, language)
        spitch_translation = self.translate_with_gpt(spitch_transcription, language)
        
        # Score accuracy against ground truth
        awarri_accuracy = self.score_accuracy_with_gpt(awarri_translation, ground_truth)
        spitch_accuracy = self.score_accuracy_with_gpt(spitch_translation, ground_truth)
        
        # Structure the data
        awarri_data = {
            "transcription": awarri_transcription,
            "translation": awarri_translation,
            "latency": round(awarri_latency, 2),
            "accuracy": round(awarri_accuracy, 1)
        }
        
        spitch_data = {
            "transcription": spitch_transcription,
            "translation": spitch_translation,
            "latency": round(spitch_latency, 2),
            "accuracy": round(spitch_accuracy, 1)
        }
        
        # Determine better model
        better_model = self.calculate_better_model(awarri_data, spitch_data)
        
        return {
            "ground_truth": ground_truth,
            "awarri": awarri_data,
            "spitch": spitch_data,
            "better_model": better_model
        }

    def process_all_files(self, base_path: str = "audio_files") -> Dict[str, Any]:
        """Process all audio files and return complete JSON structure."""
        results = {}
        
        languages = ["yoruba", "igbo", "hausa"]
        durations = ["short", "medium", "long"]
        
        for language in languages:
            results[language] = {}
            
            for duration in durations:
                for i in range(1, 6):  # Files numbered 1-5
                    file_name = f"{language}_{duration}_{i}.wav"
                    file_path = os.path.join(base_path, language, duration, file_name)
                    
                    if os.path.exists(file_path):
                        try:
                            file_data = self.process_audio_file(file_path, language, duration, i)
                            results[language][file_name] = file_data
                        except Exception as e:
                            logger.error(f"Failed to process {file_path}: {str(e)}")
                            # Create error entry
                            results[language][file_name] = {
                                "ground_truth": self.ground_truth[duration][i-1],
                                "awarri": {"transcription": f"Error: {str(e)}", "translation": "", "latency": 0.0, "accuracy": 0.0},
                                "spitch": {"transcription": f"Error: {str(e)}", "translation": "", "latency": 0.0, "accuracy": 0.0},
                                "better_model": "tie"
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
    """Main function to run the transcription processing."""
    processor = TranscriptionProcessor()
    
    logger.info("Starting transcription processing...")
    results = processor.process_all_files()
    
    logger.info("Saving results...")
    processor.save_results(results)
    
    logger.info("Processing completed!")
    
    # Print summary statistics
    total_files = sum(len(lang_data) for lang_data in results.values())
    logger.info(f"Processed {total_files} audio files across 3 languages")

if __name__ == "__main__":
    main()