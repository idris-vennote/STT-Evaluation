# Transcription API Comparison: Spitch vs Awarri

![Transcription API Comparison](https://via.placeholder.com/800x200/4ecdc4/ffffff?text=Spitch+vs+Awarri) <!-- Replace with actual banner image if available -->

This Streamlit application provides a comprehensive comparison between two transcription APIs—**Spitch AI** and **Awarri**—focusing on their performance in transcribing Nigerian native languages: **Yoruba**, **Igbo**, and **Hausa**. The app evaluates accuracy (semantic similarity to ground truth via GPT-4) and latency (response time), with dynamic weighting for overall performance scoring.

## Features

- **Language-Specific Tabs**: Dedicated sections for Yoruba, Igbo, Hausa, and a live testing tab.
- **Performance Metrics**:
  - Average accuracy and latency summaries.
  - Head-to-head win/loss/tie counts.
  - Detailed metrics table with weighted scores.
- **Visualizations**: Interactive Plotly charts comparing accuracy and latency across audio samples.
- **Audio Samples**: Playable short, medium, and long audio clips with side-by-side transcriptions, translations, and winner indicators.
- **Dynamic Weighting**: Single slider to balance accuracy (default 70%) vs. latency (default 30%) weights—adjust in real-time to recalculate winners and scores.
- **Live Testing**: Real-time audio recording and transcription using both APIs, with translations and basic comparisons.
- **English Translations**: Automatic GPT-4 translations of transcriptions for cross-language evaluation.

## Demo Screenshots

<!-- Add screenshots here, e.g., -->
- **Summary Dashboard**: Metrics and charts for each language.
- **Audio Comparison**: Side-by-side views with audio player and scores.
- **Live Test**: Record and transcribe on-the-fly.

## Prerequisites

- Python 3.8+
- Streamlit (`pip install streamlit`)
- Plotly (`pip install plotly`)
- Pandas (`pip install pandas`)
- OpenAI (`pip install openai`)
- Custom libraries: `awagpt`, `spitch`, `audio_recorder_streamlit` (install via `pip install awagpt spitch audio-recorder-streamlit`)
- Audio files in `audio_files/{language}/{duration}/` (e.g., `audio_files/yoruba/short/yoruba_short_1.wav`)
- `transcription_results.json` generated from the data processing script.

### Environment Setup

1. Create a `.env` file in the project root:
   ```
   AWARRI_API_KEY=your_awarri_key
   SPITCH_API_KEY=your_spitch_key
   OPENAI_API_KEY=your_openai_key
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Installation

1. Clone the repository:
   ```
   git clone <your-repo-url>
   cd transcription-api-comparison
   ```

2. Set up the virtual environment:
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install requirements:
   ```
   pip install -r requirements.txt
   ```

4. Generate or place your `transcription_results.json` (use the provided data generation script).

5. Run the app:
   ```
   streamlit run app.py  # Replace 'app.py' with your main script name
   ```

The app will open at `http://localhost:8501`.

## Project Structure

```
transcription-api-comparison/
├── app.py                  # Main Streamlit app
├── transcription_results.json  # Pre-generated results (accuracy, latency, etc.)
├── generate_data.py        # Script to generate JSON from audio files (optional)
├── audio_files/            # Directory for audio samples
│   ├── yoruba/
│   │   ├── short/
│   │   ├── medium/
│   │   └── long/
│   ├── igbo/              # Similar structure
│   └── hausa/             # Similar structure
├── requirements.txt        # Dependencies
├── .env                    # API keys (gitignored)
└── README.md              # This file
```

## Data Generation

To generate `transcription_results.json` from raw audio files:

1. Place audio files (WAV format) in `audio_files/{language}/{duration}/` (e.g., `yoruba_short_1.wav`).
2. Run the data generation script:
   ```
   python generate_data.py
   ```
   - It transcribes using both APIs, translates with GPT-4, scores accuracy, and computes initial better_model (70/30 weights).
   - Ground truth texts are hardcoded for telecom-related queries.

## Usage

1. **Browse Language Tabs**: Select a language to view summaries, charts, tables, and audio samples.
2. **Adjust Weights**: Use the slider under "Weight Configuration" (0.0 = full latency focus, 1.0 = full accuracy focus). Watch metrics, tables, and winners update live.
3. **Live Testing**: In the "Live Test" tab, record audio, select a language, and click "Transcribe with Both APIs" for real-time comparison.

## Customization

- **Add Languages**: Extend `languages` list in `main()` and update JSON structure.
- **Change Defaults**: Modify slider default (e.g., `0.7` for 70% accuracy).
- **Charts**: Edit `create_performance_charts()` for custom visualizations.
- **Ground Truth**: Update in `generate_data.py` for new queries.

## Limitations

- Requires valid API keys for live testing.
- Audio files must be WAV format.
- Accuracy scoring relies on GPT-4 (costs apply for generation).
- No error handling for missing audio files (shows warning).

## Contributing

1. Fork the repo.
2. Create a feature branch (`git checkout -b feature/amazing-feature`).
3. Commit changes (`git commit -m 'Add amazing feature'`).
4. Push to branch (`git push origin feature/amazing-feature`).
5. Open a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Streamlit**: For the interactive UI framework.
- **Plotly**: For visualizations.
- **OpenAI GPT-4**: For translations and accuracy scoring.
- **Spitch AI & Awarri**: For the transcription APIs.

---

*Built with ❤️ for Nigerian language tech evaluation. Questions? Open an issue!*
