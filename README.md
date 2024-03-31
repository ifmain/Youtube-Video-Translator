# Multilingual Video Translator

This project offers a comprehensive solution for downloading, splitting, translating, and reassembling videos with a focus on transforming the audio track. It utilizes advanced machine learning models and audio processing techniques to separate vocal tracks, translate the spoken language, and synthesize the translated audio. This process makes it possible to create a version of the video in a different language while maintaining the original video's visuals and timing.

## Features

- **Video Downloading**: Automatically downloads videos from provided URLs.
- **Audio Separation**: Separates vocal and instrumental tracks from the downloaded video.
- **Speech Recognition**: Converts speech from the vocal track into text.
- **Translation**: Translates the extracted text into the desired language.
- **Text-to-Speech**: Synthesizes the translated text back into audio.
- **Audio Processing**: Adjusts the timing of the translated audio to match the original speech.
- **Video Assembly**: Merges the translated audio and the original video, preserving the original visual content.

## Installation

To set up this project, clone the repository and install the required dependencies:

```bash
git clone https://github.com/ifmain/Youtube-Video-Translator
cd Youtube-Video-Translator
sudo apt-get install ffmpeg
pip install -r requirements.txt
```

Please ensure you have [FFmpeg](https://ffmpeg.org/download.html) installed on your system as it's crucial for audio and video processing.

## Usage

To use this project, run the `main.py` script and follow the prompts to input the URL of the video you wish to translate:

```bash
python main.py
```

### Important Warning

The translation model used in this project is set to translate from English to Russian by default. If you wish to translate into a different language, you must change the `trans_model_path` variable accordingly:

```python
trans_model_path='Helsinki-NLP/opus-mt-en-ru' # Change model in this string to translate to your language
```

You can find suitable model paths for your desired language by searching the [Hugging Face Model Hub](https://huggingface.co/models).

## Contributing

Contributions are welcome! If you have suggestions for improving this project, feel free to open an issue or a pull request.

## License

This project is licensed under the Apache License 2.0. For more details, see the official [license text](https://www.apache.org/licenses/LICENSE-2.0).
