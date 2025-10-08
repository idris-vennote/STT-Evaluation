import streamlit as st
import pandas as pd
import json
import os
import time
import tempfile
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import load_dotenv
from openai import OpenAI
from audio_recorder_streamlit import audio_recorder
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
        # Initialize Awarri
        awarri_client = AwaGPT(os.getenv("AWARRI_API_KEY"))
        
        # Initialize Spitch
        spitch_api_key = os.getenv("SPITCH_API_KEY")
        os.environ["SPITCH_API_KEY"] = spitch_api_key
        spitch_client = Spitch()
        
        # Initialize OpenAI
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        return awarri_client, spitch_client, openai_client
    except Exception as e:
        st.error(f"Failed to initialize API clients: {str(e)}")
        return None, None, None

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
    # Normalize latency to a speed score (lower latency = higher score)
    # Avoid division by zero
    if max_latency > 0:
        speed_score = ((max_latency - latency) / max_latency) * 100
    else:
        speed_score = 100
    
    # Ensure speed_score is between 0 and 100
    speed_score = max(0, min(100, speed_score))
    
    # Calculate weighted score
    latency_weight = 100 - accuracy_weight
    weighted_score = (accuracy * accuracy_weight / 100) + (speed_score * latency_weight / 100)
    
    return weighted_score

def determine_better_model(spitch_data, awarri_data, max_latency, accuracy_weight):
    """
    Determine which model performed better based on weighted score.
    
    Args:
        spitch_data: Dictionary with spitch accuracy and latency
        awarri_data: Dictionary with awarri accuracy and latency
        max_latency: Maximum latency in the dataset
        accuracy_weight: Weight for accuracy (0-100)
    
    Returns:
        'spitch', 'awarri', or 'tie'
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
    
    # Use a small threshold to determine ties
    if abs(spitch_score - awarri_score) < 1.0:
        return 'tie'
    elif spitch_score > awarri_score:
        return 'spitch'
    else:
        return 'awarri'

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
    max_latency = max(all_latencies) if all_latencies else 1
    
    # Update better_model for each file
    updated_data = {}
    for filename, data in language_data.items():
        updated_entry = data.copy()
        updated_entry['better_model'] = determine_better_model(
            data.get('spitch', {}),
            data.get('awarri', {}),
            max_latency,
            accuracy_weight
        )
        updated_data[filename] = updated_entry
    
    return updated_data

def transcribe_with_apis(audio_file_path, language):
    """Transcribe audio with both APIs and measure latency"""
    awarri_client, spitch_client, openai_client = initialize_api_clients()
    
    if not all([awarri_client, spitch_client, openai_client]):
        return None, None
    
    # Language mapping for Spitch
    spitch_language_map = {"yoruba": "yo", "igbo": "ig", "hausa": "ha", "english": "en"}
    # Language mapping for Awarri
    awarri_language_map = {"yoruba": "yoruba", "igbo": "igbo", "hausa": "hausa", "english": "English"}
    
    results = {}
    
    # Transcribe with Awarri
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
    Test both transcription APIs in real-time. Record your audio, select a language, 
    and compare the results from Spitch and Awarri side by side.
    """)
    
    # Check if API keys are configured
    if not all([os.getenv("AWARRI_API_KEY"), os.getenv("SPITCH_API_KEY"), os.getenv("OPENAI_API_KEY")]):
        st.error("API keys not configured. Please check your .env file.")
        st.info("Required keys: AWARRI_API_KEY, SPITCH_API_KEY, OPENAI_API_KEY")
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
        transcribe_button = st.button("Transcribe with Both APIs", type="primary")
    
    with col2:
        st.subheader("Quick Guide")
        st.markdown("""
        **How to use:**
        1. Click the record button and speak clearly
        2. Select the language you spoke in
        3. Click 'Transcribe with Both APIs'
        4. Compare results and performance
        
        **Languages supported:**
        - Yoruba (Nigerian)
        - Igbo (Nigerian) 
        - Hausa (Nigerian)
        - English
        """)
    
    # Process transcription when button is clicked
    if transcribe_button and audio_input:
        with st.spinner("Transcribing with both APIs..."):
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
                    col3, col4 = st.columns(2)
                    
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
                    
                    # Original transcriptions
                    st.subheader("Original Transcriptions")
                    col5, col6 = st.columns(2)
                    
                    with col5:
                        st.write("**Spitch Result:**")
                        if results['spitch']['status'] == 'success':
                            st.success(results['spitch']['transcription'])
                        else:
                            st.error(results['spitch']['transcription'])
                    
                    with col6:
                        st.write("**Awarri Result:**")
                        if results['awarri']['status'] == 'success':
                            st.success(results['awarri']['transcription'])
                        else:
                            st.error(results['awarri']['transcription'])
                    
                    # English translations
                    if results['spitch']['status'] == 'success' or results['awarri']['status'] == 'success':
                        st.subheader("English Translations")
                        
                        _, _, openai_client = initialize_api_clients()
                        if openai_client:
                            col7, col8 = st.columns(2)
                            
                            with col7:
                                st.write("**Spitch Translation:**")
                                if results['spitch']['status'] == 'success':
                                    spitch_translation = translate_text(
                                        results['spitch']['transcription'], 
                                        language, 
                                        openai_client
                                    )
                                    st.info(spitch_translation)
                                else:
                                    st.error("Translation unavailable (transcription failed)")
                            
                            with col8:
                                st.write("**Awarri Translation:**")
                                if results['awarri']['status'] == 'success':
                                    awarri_translation = translate_text(
                                        results['awarri']['transcription'], 
                                        language, 
                                        openai_client
                                    )
                                    st.info(awarri_translation)
                                else:
                                    st.error("Translation unavailable (transcription failed)")
                        else:
                            st.error("OpenAI client not available for translation")
                    
                    # Performance comparison
                    if results['spitch']['status'] == 'success' and results['awarri']['status'] == 'success':
                        st.subheader("Performance Analysis")
                        
                        spitch_faster = results['spitch']['latency'] < results['awarri']['latency']
                        speed_diff = abs(results['spitch']['latency'] - results['awarri']['latency'])
                        
                        if spitch_faster:
                            st.info(f"Spitch was {speed_diff:.2f}s faster than Awarri")
                        else:
                            st.info(f"Awarri was {speed_diff:.2f}s faster than Spitch")
                
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
    """Create performance summary statistics"""
    if not language_data:
        return None, None
    
    spitch_accuracies = [data['spitch']['accuracy'] for data in language_data.values() if 'accuracy' in data.get('spitch', {})]
    awarri_accuracies = [data['awarri']['accuracy'] for data in language_data.values() if 'accuracy' in data.get('awarri', {})]
    spitch_latencies = [data['spitch']['latency'] for data in language_data.values() if 'latency' in data.get('spitch', {})]
    awarri_latencies = [data['awarri']['latency'] for data in language_data.values() if 'latency' in data.get('awarri', {})]
    
    if not spitch_accuracies or not awarri_accuracies:
        return None, None
    
    spitch_wins = sum(1 for data in language_data.values() if data.get('better_model') == 'spitch')
    awarri_wins = sum(1 for data in language_data.values() if data.get('better_model') == 'awarri')
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
        'ties': ties,
        'total_samples': len(language_data)
    }
    
    return summary, (spitch_accuracies, awarri_accuracies, spitch_latencies, awarri_latencies)

def create_performance_charts(data_arrays, language_name):
    """Create performance comparison charts"""
    spitch_accuracies, awarri_accuracies, spitch_latencies, awarri_latencies = data_arrays
    
    sample_names = [f"Sample {i+1}" for i in range(len(spitch_accuracies))]
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Accuracy Comparison (%)', 'Latency Comparison (seconds)'),
        vertical_spacing=0.15
    )
    
    # Accuracy comparison
    fig.add_trace(
        go.Bar(name='Spitch AI', x=sample_names, y=spitch_accuracies, 
               marker_color='#ff6b6b', opacity=0.8),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(name='Awarri', x=sample_names, y=awarri_accuracies, 
               marker_color='#4ecdc4', opacity=0.8),
        row=1, col=1
    )
    
    # Latency comparison
    fig.add_trace(
        go.Bar(name='Spitch AI', x=sample_names, y=spitch_latencies, 
               marker_color='#ff6b6b', opacity=0.8, showlegend=False),
        row=2, col=1
    )
    fig.add_trace(
        go.Bar(name='Awarri', x=sample_names, y=awarri_latencies, 
               marker_color='#4ecdc4', opacity=0.8, showlegend=False),
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
    """Display individual audio file comparison"""
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
        col2a, col2b = st.columns(2)
        
        with col2a:
            if 'accuracy' in data.get('spitch', {}):
                st.metric(
                    label="Spitch Accuracy", 
                    value=f"{data['spitch']['accuracy']:.1f}%"
                )
            if 'latency' in data.get('spitch', {}):
                st.metric(
                    label="⚡ Spitch Latency", 
                    value=f"{data['spitch']['latency']:.2f}s"
                )
        
        with col2b:
            if 'accuracy' in data.get('awarri', {}):
                st.metric(
                    label="Awarri Accuracy", 
                    value=f"{data['awarri']['accuracy']:.1f}%"
                )
            if 'latency' in data.get('awarri', {}):
                st.metric(
                    label="⚡ Awarri Latency", 
                    value=f"{data['awarri']['latency']:.2f}s"
                )
    
    # Ground Truth
    if data.get('ground_truth'):
        st.write("**Ground Truth (English):**")
        st.info(data['ground_truth'])
    
    # Original Transcriptions
    st.write("**Original Transcriptions:**")
    col3, col4 = st.columns(2)
    
    with col3:
        st.write("**🤖 Spitch AI Transcription:**")
        if 'transcription' in data.get('spitch', {}):
            st.text_area("Spitch Transcription", data['spitch']['transcription'], height=100, key=f"spitch_orig_{filename}", disabled=True, label_visibility="hidden")
        else:
            st.warning("No transcription available")
    
    with col4:
        st.write("**🤖 Awarri Transcription:**")
        if 'transcription' in data.get('awarri', {}):
            st.text_area("", data['awarri']['transcription'], height=100, key=f"awarri_orig_{filename}", disabled=True)
        else:
            st.warning("No transcription available")
    
    # English Translations
    st.write("**🌐 English Translations:**")
    col5, col6 = st.columns(2)
    
    with col5:
        st.write("**Spitch Translation:**")
        if 'translation' in data.get('spitch', {}):
            st.text_area("", data['spitch']['translation'], height=100, key=f"spitch_trans_{filename}", disabled=True)
        else:
            st.warning("No translation available")
    
    with col6:
        st.write("**Awarri Translation:**")
        if 'translation' in data.get('awarri', {}):
            st.text_area("", data['awarri']['translation'], height=100, key=f"awarri_trans_{filename}", disabled=True)
        else:
            st.warning("No translation available")
    
    # Better model indicator
    if data.get('better_model') == 'spitch':
        st.success("🏆 **Winner: Spitch AI** - Better overall performance (weighted score)")
    elif data.get('better_model') == 'awarri':
        st.success("🏆 **Winner: Awarri** - Better overall performance (weighted score)")
    elif data.get('better_model') == 'tie':
        st.info("🤝 **Result: Tie** - Similar performance")
    else:
        st.info("Result: Unable to determine winner")
    
    st.divider()

def create_detailed_metrics_table(language_data):
    """Create a detailed metrics table for all files"""
    rows = []
    for filename, data in language_data.items():
        row = {
            'File': filename,
            'Spitch Accuracy (%)': data['spitch']['accuracy'],
            'Awarri Accuracy (%)': data['awarri']['accuracy'],
            'Spitch Latency (s)': data['spitch']['latency'],
            'Awarri Latency (s)': data['awarri']['latency'],
            'Better Model': data['better_model'].title()
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df

def main():
    # Title and description
    st.title("🎤 Transcription API Comparison")
    st.subheader("Spitch AI vs Awarri - Nigerian Native Languages")
    
    st.markdown("""
    This application compares the performance of **Spitch AI** and **Awarri** transcription APIs 
    across three Nigerian native languages: **Yoruba**, **Igbo**, and **Hausa**.
    
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
    tab1, tab2, tab3, tab4 = st.tabs(["🇳🇬 Yoruba", "🇳🇬 Igbo", "🇳🇬 Hausa", "🎙️ Live Test"])
    
    languages = [
        ("yoruba", "Yoruba", tab1),
        ("igbo", "Igbo", tab2),
        ("hausa", "Hausa", tab3)
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
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        label="Spitch Avg Accuracy", 
                        value=f"{summary['spitch']['avg_accuracy']:.1f}%"
                    )
                
                with col2:
                    st.metric(
                        label="⚡ Spitch Avg Latency", 
                        value=f"{summary['spitch']['avg_latency']:.2f}s"
                    )
                
                with col3:
                    st.metric(
                        label="Awarri Avg Accuracy", 
                        value=f"{summary['awarri']['avg_accuracy']:.1f}%"
                    )
                
                with col4:
                    st.metric(
                        label="⚡ Awarri Avg Latency", 
                        value=f"{summary['awarri']['avg_latency']:.2f}s"
                    )
                
                # Win/Loss summary
                st.subheader("Head-to-Head Results")
                col5, col6, col7 = st.columns(3)
                
                with col5:
                    st.metric(
                        label="🔴 Spitch Wins", 
                        value=summary['spitch']['wins']
                    )
                
                with col6:
                    st.metric(
                        label="🔵 Awarri Wins", 
                        value=summary['awarri']['wins']
                    )
                
                with col7:
                    st.metric(
                        label="🤝 Ties", 
                        value=summary['ties']
                    )
                
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
    with tab4:
        live_testing_tab()

if __name__ == "__main__":
    main()