import streamlit as st
import pandas as pd
import json
import os
import time
import tempfile
import base64
import requests
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import load_dotenv
from openai import OpenAI
#from audio_recorder_streamlit import audio_recorder
from awagpt import AwaGPT
from spitch import Spitch

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(
    page_title="Transcription API Comparison: Spitch vs Awarri",
    page_icon="🎤",
    layout="wide"
)

# Language mapping for Spitch API
LANGUAGE_MAPPING = {
    "yoruba": "yo",
    "igbo": "ig",
    "hausa": "ha",
    "english": "en"
}

@st.cache_data
def load_transcription_data():
    """Load transcription data from JSON file"""
    try:
        with open('transcription_results.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("❌ transcription_results.json file not found! Please ensure the file exists in the same directory.")
        return {}
    except json.JSONDecodeError:
        st.error("❌ Error reading transcription_results.json. Please check the file format.")
        return {}

@st.cache_resource
def initialize_api_clients():
    """Initialize API clients (cached to avoid reinitializing)"""
    try:
        # Initialize Awarri (old)
        awarri_client = AwaGPT(os.getenv("AWARRI_API_KEY"))
        
        # Initialize Spitch
        spitch_api_key = os.getenv("SPITCH_API_KEY")
        os.environ["SPITCH_API_KEY"] = spitch_api_key
        spitch_client = Spitch()
        
        # Initialize OpenAI
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # New Awarri API credentials
        awarri_new_url = os.getenv("AWARRI_NEW_API_URL")
        awarri_new_key = os.getenv("AWARRI_NEW_API_KEY")
        
        return awarri_client, spitch_client, openai_client, awarri_new_url, awarri_new_key
    except Exception as e:
        st.error(f"Failed to initialize API clients: {str(e)}")
        return None, None, None, None, None

def load_audio_file(file_path):
    """Check if audio file exists and return path"""
    if os.path.exists(file_path):
        return file_path
    return None

def calculate_weighted_score(accuracy, latency, max_latency, accuracy_weight):
    """
    Calculate weighted score based on accuracy and latency.
    
    Args:
        accuracy: Accuracy percentage (0-100)
        latency: Latency in seconds
        max_latency: Maximum latency in the dataset (for normalization)
        accuracy_weight: Weight for accuracy (0-100), latency weight is (100 - accuracy_weight)
    
    Returns:
        Weighted score (0-100)
    """
    if max_latency > 0:
        speed_score = ((max_latency - latency) / max_latency) * 100
    else:
        speed_score = 100
    
    speed_score = max(0, min(100, speed_score))
    
    latency_weight = 100 - accuracy_weight
    weighted_score = (accuracy * accuracy_weight / 100) + (speed_score * latency_weight / 100)
    
    return weighted_score

def determine_better_model_three_way(spitch_data, awarri_data, awarri_new_data, max_latency, accuracy_weight):
    """
    Determine which model performed better among three models based on weighted score.
    
    Args:
        spitch_data: Dictionary with spitch accuracy and latency
        awarri_data: Dictionary with awarri accuracy and latency
        awarri_new_data: Dictionary with awarri_new accuracy and latency
        max_latency: Maximum latency in the dataset
        accuracy_weight: Weight for accuracy (0-100)
    
    Returns:
        'spitch', 'awarri', 'awarri_new', or 'tie'
    """
    spitch_score = calculate_weighted_score(
        spitch_data.get('accuracy', 0),
        spitch_data.get('latency', 0),
        max_latency,
        accuracy_weight
    )
    
    awarri_score = calculate_weighted_score(
        awarri_data.get('accuracy', 0),
        awarri_data.get('latency', 0),
        max_latency,
        accuracy_weight
    )
    
    awarri_new_score = calculate_weighted_score(
        awarri_new_data.get('accuracy', 0),
        awarri_new_data.get('latency', 0),
        max_latency,
        accuracy_weight
    )
    
    scores = {
        'spitch': spitch_score,
        'awarri': awarri_score,
        'awarri_new': awarri_new_score
    }
    
    best_model = max(scores, key=scores.get)
    best_score = scores[best_model]
    
    # Check for ties (within 1.0 threshold)
    tied_models = [model for model, score in scores.items() if abs(score - best_score) < 1.0]
    
    if len(tied_models) > 1:
        return 'tie'
    
    return best_model

def recalculate_winners(language_data, accuracy_weight):
    """
    Recalculate winners for all files in a language based on new weights.
    
    Args:
        language_data: Dictionary of file data for a language
        accuracy_weight: Weight for accuracy (0-100)
    
    Returns:
        Updated language_data with new better_model values
    """
    # Find max latency for normalization
    all_latencies = []
    for data in language_data.values():
        all_latencies.append(data.get('spitch', {}).get('latency', 0))
        all_latencies.append(data.get('awarri', {}).get('latency', 0))
        if 'awarri_new' in data:
            all_latencies.append(data.get('awarri_new', {}).get('latency', 0))
    max_latency = max(all_latencies) if all_latencies else 1
    
    # Update better_model for each file
    updated_data = {}
    for filename, data in language_data.items():
        updated_entry = data.copy()
        
        # Only recalculate if all three models exist
        if 'spitch' in data and 'awarri' in data and 'awarri_new' in data:
            updated_entry['better_model'] = determine_better_model_three_way(
                data.get('spitch', {}),
                data.get('awarri', {}),
                data.get('awarri_new', {}),
                max_latency,
                accuracy_weight
            )
        
        updated_data[filename] = updated_entry
    
    return updated_data

def encode_audio_to_base64(file_path):
    """Read and encode audio file to base64 with required prefix."""
    with open(file_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:audio/webm;base64,{encoded}"

def transcribe_with_apis(audio_file_path, language):
    """Transcribe audio with all three APIs and measure latency"""
    awarri_client, spitch_client, openai_client, awarri_new_url, awarri_new_key = initialize_api_clients()
    
    if not all([awarri_client, spitch_client, openai_client]):
        return None
    
    # Language mappings
    spitch_language_map = {"yoruba": "yo", "igbo": "ig", "hausa": "ha", "english": "en"}
    awarri_language_map = {"yoruba": "yoruba", "igbo": "igbo", "hausa": "hausa", "english": "English"}
    awarri_new_language_map = {"yoruba": "yoruba", "igbo": "igbo", "hausa": "hausa", "english": "english"}
    
    results = {}
    
    # Transcribe with old Awarri
    try:
        start_time = time.time()
        awarri_response = awarri_client.transcribe_audio(
            audio_file_path, 
            language=awarri_language_map.get(language.lower(), "English")
        )
        awarri_latency = time.time() - start_time
        
        results['awarri'] = {
            'transcription': str(awarri_response) if awarri_response else "No transcription available",
            'latency': round(awarri_latency, 2),
            'status': 'success'
        }
    except Exception as e:
        results['awarri'] = {
            'transcription': f"Error: {str(e)}",
            'latency': 0.0,
            'status': 'error'
        }
    
    # Transcribe with Spitch
    try:
        start_time = time.time()
        with open(audio_file_path, "rb") as f:
            spitch_response = spitch_client.speech.transcribe(
                language=spitch_language_map.get(language.lower(), "en"),
                content=f.read()
            )
        spitch_latency = time.time() - start_time
        
        results['spitch'] = {
            'transcription': spitch_response.text if hasattr(spitch_response, 'text') else str(spitch_response),
            'latency': round(spitch_latency, 2),
            'status': 'success'
        }
    except Exception as e:
        results['spitch'] = {
            'transcription': f"Error: {str(e)}",
            'latency': 0.0,
            'status': 'error'
        }
    
    # Transcribe with new Awarri
    if awarri_new_url and awarri_new_key:
        try:
            start_time = time.time()
            
            base64_data = encode_audio_to_base64(audio_file_path)
            
            payload = {
                "base64_data": base64_data,
                "lang": awarri_new_language_map.get(language.lower(), "english")
            }
            
            headers = {
                "accept": "application/json",
                "Content-Type": "application/json",
                "X-API-Key": awarri_new_key
            }
            
            response = requests.post(awarri_new_url, json=payload, headers=headers)
            response.raise_for_status()
            
            awarri_new_latency = time.time() - start_time
            
            response_data = response.json()
            transcription = response_data.get("text", "") or response_data.get("transcription", "") or str(response_data)
            
            results['awarri_new'] = {
                'transcription': transcription,
                'latency': round(awarri_new_latency, 2),
                'status': 'success'
            }
        except Exception as e:
            results['awarri_new'] = {
                'transcription': f"Error: {str(e)}",
                'latency': 0.0,
                'status': 'error'
            }
    else:
        results['awarri_new'] = {
            'transcription': "API credentials not configured",
            'latency': 0.0,
            'status': 'error'
        }
    
    return results

def translate_text(text, source_language, openai_client):
    """Translate text to English using GPT"""
    try:
        if text.startswith("Error:") or not text.strip():
            return text
            
        prompt = f"""Translate the following {source_language} text to English. 
        If the text is already in English or mostly English, return it as is.
        Only return the translation, no additional text or explanations.
        
        Text to translate: {text}"""
        
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a professional translator. Translate accurately and naturally."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.1
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        return f"Translation error: {str(e)}"

def live_testing_tab():
    """Create the live testing tab interface"""
    st.header("Live API Testing")
    st.markdown("""
    Test all three transcription APIs in real-time. Record your audio, select a language, 
    and compare the results from Spitch, Awarri (old), and Awarri New side by side.
    """)
    
    # Check if API keys are configured
    required_keys = ["AWARRI_API_KEY", "SPITCH_API_KEY", "OPENAI_API_KEY", "AWARRI_NEW_API_URL", "AWARRI_NEW_API_KEY"]
    if not all([os.getenv(key) for key in required_keys]):
        st.error("API keys not configured. Please check your .env file.")
        st.info(f"Required keys: {', '.join(required_keys)}")
        return
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Audio input
        st.subheader("Audio Input")
        audio_input = st.audio_input("Record your audio")
        
        # Language selection
        language = st.selectbox(
            "Select Language",
            ["Yoruba", "Igbo", "Hausa", "English"],
            key="live_language_selector"
        )
        
        # Transcribe button
        transcribe_button = st.button("Transcribe with All APIs", type="primary")
    
    with col2:
        st.subheader("Quick Guide")
        st.markdown("""
        **How to use:**
        1. Click the record button and speak clearly
        2. Select the language you spoke in
        3. Click 'Transcribe with All APIs'
        4. Compare results and performance
        
        **Languages supported:**
        - Yoruba (Nigerian)
        - Igbo (Nigerian) 
        - Hausa (Nigerian)
        - English
        
        **Models tested:**
        - Spitch AI
        - Awarri (Legacy)
        - Awarri New
        """)
    
    # Process transcription when button is clicked
    if transcribe_button and audio_input:
        with st.spinner("Transcribing with all APIs..."):
            # Save audio to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(audio_input.read())
                tmp_audio_path = tmp_file.name
            
            try:
                # Get transcriptions
                results = transcribe_with_apis(tmp_audio_path, language)
                
                if results:
                    st.success("Transcription completed!")
                    
                    # Display results
                    st.subheader("Results Comparison")
                    
                    # Performance metrics
                    col3, col4, col5 = st.columns(3)
                    
                    with col3:
                        st.metric(
                            label="Spitch Latency",
                            value=f"{results['spitch']['latency']:.2f}s"
                        )
                    
                    with col4:
                        st.metric(
                            label="Awarri Latency",
                            value=f"{results['awarri']['latency']:.2f}s"
                        )
                    
                    with col5:
                        st.metric(
                            label="Awarri New Latency",
                            value=f"{results['awarri_new']['latency']:.2f}s"
                        )
                    
                    # Original transcriptions
                    st.subheader("Original Transcriptions")
                    col6, col7, col8 = st.columns(3)
                    
                    with col6:
                        st.write("**Spitch Result:**")
                        if results['spitch']['status'] == 'success':
                            st.success(results['spitch']['transcription'])
                        else:
                            st.error(results['spitch']['transcription'])
                    
                    with col7:
                        st.write("**Awarri Result:**")
                        if results['awarri']['status'] == 'success':
                            st.success(results['awarri']['transcription'])
                        else:
                            st.error(results['awarri']['transcription'])
                    
                    with col8:
                        st.write("**Awarri New Result:**")
                        if results['awarri_new']['status'] == 'success':
                            st.success(results['awarri_new']['transcription'])
                        else:
                            st.error(results['awarri_new']['transcription'])
                    
                    # English translations (only for non-English languages)
                    if language.lower() != "english":
                        st.subheader("English Translations")
                        
                        _, _, openai_client, _, _ = initialize_api_clients()
                        if openai_client:
                            col9, col10, col11 = st.columns(3)
                            
                            with col9:
                                st.write("**Spitch Translation:**")
                                if results['spitch']['status'] == 'success':
                                    spitch_translation = translate_text(
                                        results['spitch']['transcription'], 
                                        language, 
                                        openai_client
                                    )
                                    st.info(spitch_translation)
                                else:
                                    st.error("Translation unavailable")
                            
                            with col10:
                                st.write("**Awarri Translation:**")
                                if results['awarri']['status'] == 'success':
                                    awarri_translation = translate_text(
                                        results['awarri']['transcription'], 
                                        language, 
                                        openai_client
                                    )
                                    st.info(awarri_translation)
                                else:
                                    st.error("Translation unavailable")
                            
                            with col11:
                                st.write("**Awarri New Translation:**")
                                if results['awarri_new']['status'] == 'success':
                                    awarri_new_translation = translate_text(
                                        results['awarri_new']['transcription'], 
                                        language, 
                                        openai_client
                                    )
                                    st.info(awarri_new_translation)
                                else:
                                    st.error("Translation unavailable")
                        else:
                            st.error("OpenAI client not available for translation")
                    
                    # Performance comparison
                    successful_models = [model for model, data in results.items() if data['status'] == 'success']
                    if len(successful_models) >= 2:
                        st.subheader("Performance Analysis")
                        
                        latencies = {model: results[model]['latency'] for model in successful_models}
                        fastest_model = min(latencies, key=latencies.get)
                        
                        model_names = {
                            'spitch': 'Spitch',
                            'awarri': 'Awarri (Legacy)',
                            'awarri_new': 'Awarri New'
                        }
                        
                        st.info(f"🏆 Fastest: {model_names.get(fastest_model, fastest_model)} ({latencies[fastest_model]:.2f}s)")
                
                else:
                    st.error("Failed to get transcriptions from APIs")
            
            finally:
                # Clean up temporary file
                try:
                    os.unlink(tmp_audio_path)
                except:
                    pass
    
    elif transcribe_button and not audio_input:
        st.warning("Please record audio before transcribing")

def create_performance_summary(language_data, language_name):
    """Create performance summary statistics for 3 models"""
    if not language_data:
        return None, None
    
    spitch_accuracies = [data['spitch']['accuracy'] for data in language_data.values() if 'accuracy' in data.get('spitch', {})]
    awarri_accuracies = [data['awarri']['accuracy'] for data in language_data.values() if 'accuracy' in data.get('awarri', {})]
    awarri_new_accuracies = [data['awarri_new']['accuracy'] for data in language_data.values() if 'accuracy' in data.get('awarri_new', {})]
    
    spitch_latencies = [data['spitch']['latency'] for data in language_data.values() if 'latency' in data.get('spitch', {})]
    awarri_latencies = [data['awarri']['latency'] for data in language_data.values() if 'latency' in data.get('awarri', {})]
    awarri_new_latencies = [data['awarri_new']['latency'] for data in language_data.values() if 'latency' in data.get('awarri_new', {})]
    
    if not spitch_accuracies or not awarri_accuracies:
        return None, None
    
    spitch_wins = sum(1 for data in language_data.values() if data.get('better_model') == 'spitch')
    awarri_wins = sum(1 for data in language_data.values() if data.get('better_model') == 'awarri')
    awarri_new_wins = sum(1 for data in language_data.values() if data.get('better_model') == 'awarri_new')
    ties = sum(1 for data in language_data.values() if data.get('better_model') == 'tie')
    
    summary = {
        'spitch': {
            'avg_accuracy': sum(spitch_accuracies) / len(spitch_accuracies) if spitch_accuracies else 0,
            'avg_latency': sum(spitch_latencies) / len(spitch_latencies) if spitch_latencies else 0,
            'wins': spitch_wins
        },
        'awarri': {
            'avg_accuracy': sum(awarri_accuracies) / len(awarri_accuracies) if awarri_accuracies else 0,
            'avg_latency': sum(awarri_latencies) / len(awarri_latencies) if awarri_latencies else 0,
            'wins': awarri_wins
        },
        'awarri_new': {
            'avg_accuracy': sum(awarri_new_accuracies) / len(awarri_new_accuracies) if awarri_new_accuracies else 0,
            'avg_latency': sum(awarri_new_latencies) / len(awarri_new_latencies) if awarri_new_latencies else 0,
            'wins': awarri_new_wins
        },
        'ties': ties,
        'total_samples': len(language_data)
    }
    
    return summary, (spitch_accuracies, awarri_accuracies, awarri_new_accuracies, spitch_latencies, awarri_latencies, awarri_new_latencies)

def create_performance_charts(data_arrays, language_name):
    """Create performance comparison charts for 3 models"""
    spitch_accuracies, awarri_accuracies, awarri_new_accuracies, spitch_latencies, awarri_latencies, awarri_new_latencies = data_arrays
    
    sample_names = [f"Sample {i+1}" for i in range(len(spitch_accuracies))]
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Accuracy Comparison (%)', 'Latency Comparison (seconds)'),
        vertical_spacing=0.15
    )
    
    # Accuracy comparison
    fig.add_trace(
        go.Bar(name='Spitch', x=sample_names, y=spitch_accuracies, 
               marker_color='#ff6b6b', opacity=0.8),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(name='Awarri', x=sample_names, y=awarri_accuracies, 
               marker_color='#4ecdc4', opacity=0.8),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(name='Awarri New', x=sample_names, y=awarri_new_accuracies, 
               marker_color='#95e1d3', opacity=0.8),
        row=1, col=1
    )
    
    # Latency comparison
    fig.add_trace(
        go.Bar(name='Spitch', x=sample_names, y=spitch_latencies, 
               marker_color='#ff6b6b', opacity=0.8, showlegend=False),
        row=2, col=1
    )
    fig.add_trace(
        go.Bar(name='Awarri', x=sample_names, y=awarri_latencies, 
               marker_color='#4ecdc4', opacity=0.8, showlegend=False),
        row=2, col=1
    )
    fig.add_trace(
        go.Bar(name='Awarri New', x=sample_names, y=awarri_new_latencies, 
               marker_color='#95e1d3', opacity=0.8, showlegend=False),
        row=2, col=1
    )
    
    fig.update_layout(
        title_text=f"{language_name} Performance Analysis",
        height=600,
        barmode='group'
    )
    
    fig.update_yaxes(title_text="Accuracy (%)", row=1, col=1)
    fig.update_yaxes(title_text="Latency (seconds)", row=2, col=1)
    fig.update_xaxes(title_text="Audio Samples", row=2, col=1)
    
    return fig

def display_audio_comparison(filename, data, audio_base_path):
    """Display individual audio file comparison for 3 models"""
    st.subheader(f"🎵 {filename}")
    
    # Audio file path
    audio_path = os.path.join(audio_base_path, filename)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Audio player
        if load_audio_file(audio_path):
            st.audio(audio_path)
        else:
            st.warning(f"⚠️ Audio file not found: {audio_path}")
    
    with col2:
        # Performance metrics
        col2a, col2b, col2c = st.columns(3)
        
        with col2a:
            st.write("**Spitch**")
            if 'accuracy' in data.get('spitch', {}):
                st.metric("Accuracy", f"{data['spitch']['accuracy']:.1f}%")
            if 'latency' in data.get('spitch', {}):
                st.metric("Latency", f"{data['spitch']['latency']:.2f}s")
        
        with col2b:
            st.write("**Awarri**")
            if 'accuracy' in data.get('awarri', {}):
                st.metric("Accuracy", f"{data['awarri']['accuracy']:.1f}%")
            if 'latency' in data.get('awarri', {}):
                st.metric("Latency", f"{data['awarri']['latency']:.2f}s")
        
        with col2c:
            st.write("**Awarri New**")
            if 'accuracy' in data.get('awarri_new', {}):
                st.metric("Accuracy", f"{data['awarri_new']['accuracy']:.1f}%")
            if 'latency' in data.get('awarri_new', {}):
                st.metric("Latency", f"{data['awarri_new']['latency']:.2f}s")
    
    # Ground Truth
    if data.get('ground_truth'):
        st.write("**Ground Truth (English):**")
        st.info(data['ground_truth'])
    
    # Original Transcriptions
    st.write("**Original Transcriptions:**")
    col3, col4, col5 = st.columns(3)
    
    with col3:
        st.write("**Spitch:**")
        if 'transcription' in data.get('spitch', {}):
            st.text_area("", data['spitch']['transcription'], height=100, key=f"spitch_orig_{filename}", disabled=True, label_visibility="collapsed")
        else:
            st.warning("No transcription")
    
    with col4:
        st.write("**Awarri:**")
        if 'transcription' in data.get('awarri', {}):
            st.text_area("", data['awarri']['transcription'], height=100, key=f"awarri_orig_{filename}", disabled=True, label_visibility="collapsed")
        else:
            st.warning("No transcription")
    
    with col5:
        st.write("**Awarri New:**")
        if 'transcription' in data.get('awarri_new', {}):
            st.text_area("", data['awarri_new']['transcription'], height=100, key=f"awarri_new_orig_{filename}", disabled=True, label_visibility="collapsed")
        else:
            st.warning("No transcription")
    
    # English Translations
    st.write("**🌐 English Translations:**")
    col6, col7, col8 = st.columns(3)
    
    with col6:
        st.write("**Spitch:**")
        if 'translation' in data.get('spitch', {}):
            st.text_area("", data['spitch']['translation'], height=100, key=f"spitch_trans_{filename}", disabled=True, label_visibility="collapsed")
        else:
            st.warning("No translation")
    
    with col7:
        st.write("**Awarri:**")
        if 'translation' in data.get('awarri', {}):
            st.text_area("", data['awarri']['translation'], height=100, key=f"awarri_trans_{filename}", disabled=True, label_visibility="collapsed")
        else:
            st.warning("No translation")
    
    with col8:
        st.write("**Awarri New:**")
        if 'translation' in data.get('awarri_new', {}):
            st.text_area("", data['awarri_new']['translation'], height=100, key=f"awarri_new_trans_{filename}", disabled=True, label_visibility="collapsed")
        else:
            st.warning("No translation")
    
    # Better model indicator
    model_names = {
        'spitch': 'Spitch',
        'awarri': 'Awarri (Legacy)',
        'awarri_new': 'Awarri New'
    }
    
    if data.get('better_model') == 'spitch':
        st.success(f"🏆 **Winner: {model_names['spitch']}** - Better overall performance (weighted score)")
    elif data.get('better_model') == 'awarri':
        st.success(f"🏆 **Winner: {model_names['awarri']}** - Better overall performance (weighted score)")
    elif data.get('better_model') == 'awarri_new':
        st.success(f"🏆 **Winner: {model_names['awarri_new']}** - Better overall performance (weighted score)")
    elif data.get('better_model') == 'tie':
        st.info("🤝 **Result: Tie** - Similar performance")
    else:
        st.info("Result: Unable to determine winner")
    
    st.divider()

def create_detailed_metrics_table(language_data):
    """Create a detailed metrics table for all files with 3 models"""
    rows = []
    for filename, data in language_data.items():
        row = {
            'File': filename,
            'Spitch Accuracy (%)': data.get('spitch', {}).get('accuracy', 0),
            'Awarri Accuracy (%)': data.get('awarri', {}).get('accuracy', 0),
            'Awarri New Accuracy (%)': data.get('awarri_new', {}).get('accuracy', 0),
            'Spitch Latency (s)': data.get('spitch', {}).get('latency', 0),
            'Awarri Latency (s)': data.get('awarri', {}).get('latency', 0),
            'Awarri New Latency (s)': data.get('awarri_new', {}).get('latency', 0),
            'Better Model': data.get('better_model', 'N/A').replace('_', ' ').title()
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df

def main():
    # Title and description
    st.title("🎤 Transcription API Comparison")
    st.subheader("Spitch vs Awarri (Legacy) vs Awarri New")
    
    st.markdown("""
    This application compares the performance of **Spitch**, **Awarri (Legacy)**, and **Awarri New** transcription APIs 
    across three Nigerian native languages (**Yoruba**, **Igbo**, **Hausa**) and **English**.
    
    Each language is tested with short, medium, and long audio samples to evaluate:
    - **Accuracy**: GPT-4 scored semantic similarity to ground truth (0-100%)
    - ⚡ **Speed**: Response latency in seconds
    - **Overall Performance**: Weighted combination (adjustable with slider in each tab)
    """)
    
    # Load data
    data = load_transcription_data()
    
    if not data:
        st.stop()
    
    # Language tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🇳🇬 Yoruba", "🇳🇬 Igbo", "🇳🇬 Hausa", "🇬🇧 English", "🎙️ Live Test"])
    
    languages = [
        ("yoruba", "Yoruba", tab1),
        ("igbo", "Igbo", tab2),
        ("hausa", "Hausa", tab3),
        ("english", "English", tab4)
    ]
    
    for lang_key, lang_name, tab in languages:
        with tab:
            if lang_key not in data:
                st.error(f"No data found for {lang_name} language")
                continue
            
            # Weight adjustment slider at the top of each tab
            st.markdown("### ⚖️ Adjust Performance Weights")
            
            col_slider, col_info = st.columns([3, 1])
            
            with col_slider:
                accuracy_weight = st.slider(
                    "Accuracy Weight (%)",
                    min_value=0,
                    max_value=100,
                    value=70,
                    step=5,
                    key=f"weight_slider_{lang_key}",
                    help="Adjust how much accuracy vs speed matters in determining the winner"
                )
            
            with col_info:
                latency_weight = 100 - accuracy_weight
                st.metric("Speed Weight", f"{latency_weight}%")
            
            st.markdown(f"**Current weights:** {accuracy_weight}% Accuracy, {latency_weight}% Speed")
            st.divider()
            
            # Recalculate winners based on current weights
            language_data = recalculate_winners(data[lang_key], accuracy_weight)
            
            # Performance summary
            summary, data_arrays = create_performance_summary(language_data, lang_name)
            
            if summary:
                st.header(f"📊 {lang_name} Performance Summary")
                
                # Average metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write("**Spitch**")
                    st.metric("Avg Accuracy", f"{summary['spitch']['avg_accuracy']:.1f}%")
                    st.metric("Avg Latency", f"{summary['spitch']['avg_latency']:.2f}s")
                
                with col2:
                    st.write("**Awarri (Legacy)**")
                    st.metric("Avg Accuracy", f"{summary['awarri']['avg_accuracy']:.1f}%")
                    st.metric("Avg Latency", f"{summary['awarri']['avg_latency']:.2f}s")
                
                with col3:
                    st.write("**Awarri New**")
                    st.metric("Avg Accuracy", f"{summary['awarri_new']['avg_accuracy']:.1f}%")
                    st.metric("Avg Latency", f"{summary['awarri_new']['avg_latency']:.2f}s")
                
                # Win/Loss summary
                st.subheader("Head-to-Head Results")
                col4, col5, col6, col7 = st.columns(4)
                
                with col4:
                    st.metric("🔴 Spitch Wins", summary['spitch']['wins'])
                
                with col5:
                    st.metric("🔵 Awarri Wins", summary['awarri']['wins'])
                
                with col6:
                    st.metric("🟢 Awarri New Wins", summary['awarri_new']['wins'])
                
                with col7:
                    st.metric("🤝 Ties", summary['ties'])
                
                # Performance charts
                if data_arrays:
                    fig = create_performance_charts(data_arrays, lang_name)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Detailed metrics table
                st.subheader("Detailed Metrics")
                metrics_df = create_detailed_metrics_table(language_data)
                st.dataframe(metrics_df, use_container_width=True)
            
            # Audio samples by category
            categories = ['short', 'medium', 'long']
            
            for category in categories:
                category_files = {k: v for k, v in language_data.items() if category in k}
                
                if category_files:
                    st.header(f"🎵 {category.title()} Audio Samples")
                    
                    # Create audio base path
                    audio_base_path = f"audio_files/{lang_key}/{category}"
                    
                    for filename, file_data in category_files.items():
                        display_audio_comparison(filename, file_data, audio_base_path)
    
    # Live Test Tab
    with tab5:
        live_testing_tab()

if __name__ == "__main__":
    main()