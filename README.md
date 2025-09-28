# YouTube-video-Summarizer-
A complete video processing pipeline that downloads videos, extracts audio, transcribes speech, and summarizes the content using AI models.   Supports **YouTube URLs** and **local video files**.




 Video Summarizer Pipeline

This project provides a **complete pipeline** to process videos into **transcripts and summaries** using:

* [yt-dlp](https://github.com/yt-dlp/yt-dlp) → download YouTube videos
* [FFmpeg](https://ffmpeg.org/) → extract audio
* [OpenAI Whisper](https://github.com/openai/whisper) → transcribe speech to text
* [HuggingFace Transformers](https://huggingface.co/transformers/) → summarize text



 Features

*  Download videos from YouTube (or use local files)
*  Extract audio automatically
*  Transcribe audio to text using **Whisper**
*  Summarize transcripts with **BART**
*  Save transcript & summary in `output/` folder



  Installation

1. Clone this repo:

```bash
git clone https://github.com/YOUR_USERNAME/video-summarizer.git
cd video-summarizer
```

2. Create a virtual environment:

```bash
python -m venv .venv
```

3. Activate it:

Windows PowerShell:

  ```powershell
  .venv\Scripts\Activate.ps1
  ```
  Command Prompt:

  ```cmd
  .venv\Scripts\activate.bat
  ```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

---

  Usage

 **Option 1 – Interactive Mode (recommended)**

Run without arguments and follow the prompts:

```bash
python video_pipeline.py
```

You’ll be asked to enter:

* YouTube URL or local video path
* Whisper model size (tiny, base, small, medium, large)
* Whether to keep audio/video files

**Option 2 – Direct Mode**

Run with a YouTube URL:

bash
python video_pipeline.py "https://www.youtube.com/watch?v=EXAMPLE"


 Output

After running, you’ll find:

* `output/transcript.txt` → full transcript
* `output/summary.txt` → summarized version


 Requirements

* Python 3.8+
* FFmpeg (installed automatically via `imageio-ffmpeg`)
* Internet connection (for downloading & HuggingFace models)



 Example

text
 Video Processing Pipeline
 Enter URL or file path: https://www.youtube.com/watch?v=f_N3PGvnVKg

 Transcript saved to: output/transcript.txt
 Summary saved to: output/summary.txt
 Processing complete!


 Contributing

Pull requests are welcome! If you’d like to improve transcription, summarization, or add new features, feel free to fork and submit PRs.


