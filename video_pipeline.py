#!/usr/bin/env python3
"""
Complete Video Processing Pipeline
Combines video download, audio extraction, transcription, and summarization
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import time
from typing import Optional, List

# Video/Audio processing
import yt_dlp
from imageio_ffmpeg import get_ffmpeg_exe

# Transcription
import whisper

# Summarization
from transformers import pipeline, BartTokenizer
import torch
from tqdm import tqdm


class VideoProcessor:
    """Main class for processing videos into transcripts and summaries"""
    
    def __init__(self, whisper_model="base", summarizer_model="facebook/bart-large-cnn"):
        """
        Initialize the video processor
        
        Args:
            whisper_model: Size of Whisper model ("tiny", "base", "small", "medium", "large")
            summarizer_model: HuggingFace model for summarization
        """
        print(f"Initializing VideoProcessor...")
        print(f"Loading Whisper model: {whisper_model}")
        self.whisper_model = whisper.load_model(whisper_model)
        
        # Check for GPU
        self.device = 0 if torch.cuda.is_available() else -1
        device_name = "GPU" if self.device == 0 else "CPU"
        print(f"Using {device_name} for summarization")
        
        print(f"Loading summarizer: {summarizer_model}")
        self.summarizer = pipeline("summarization", model=summarizer_model, device=self.device)
        self.tokenizer = BartTokenizer.from_pretrained(summarizer_model)
        
        print("VideoProcessor initialized successfully!\n")
    
    def download_video(self, url: str, output_path: str = "downloaded_video.mp4") -> str:
        """Download video from YouTube URL using yt-dlp"""
        print(f"Downloading video from: {url}")
        try:
            # Configure yt-dlp options
            ydl_opts = {
                "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
                "outtmpl": output_path,
                "merge_output_format": "mp4",
                "writeinfojson": False,
                "writesubtitles": False,
                "writeautomaticsub": False,
                "quiet": False,
                "no_warnings": False,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Get video info first
                info = ydl.extract_info(url, download=False)
                print(f"Video title: {info.get('title', 'Unknown')}")
                duration = info.get('duration')
                if duration:
                    print(f"Duration: {duration} seconds ({duration//60}:{duration%60:02d})")
                
                # Download the video
                ydl.download([url])
                
            print(f"✅ Downloaded to: {output_path}\n")
            return output_path
            
        except Exception as e:
            print(f"❌ Error downloading video: {e}")
            raise
    
    def extract_audio(self, video_path: str, audio_path: str = "extracted_audio.wav") -> str:
        """Extract audio from video file"""
        print(f"Extracting audio from: {video_path}")
        try:
            ffmpeg_exe = get_ffmpeg_exe()
            cmd = [
                ffmpeg_exe, "-y",
                "-i", video_path,
                "-vn",  # No video
                "-acodec", "pcm_s16le",
                "-ar", "16000",  # Whisper prefers 16kHz
                "-ac", "1",      # Mono audio
                audio_path
            ]
            
            # Run with progress information
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(f"FFmpeg error: {result.stderr}")
                
            print(f"✅ Audio extracted to: {audio_path}\n")
            return audio_path
            
        except Exception as e:
            print(f"❌ Error extracting audio: {e}")
            raise
    
    def transcribe(self, audio_path: str, language: Optional[str] = None) -> dict:
        """
        Transcribe audio using Whisper
        
        Args:
            audio_path: Path to audio file
            language: Language code (e.g., 'en', 'es', 'fr') or None for auto-detection
            
        Returns:
            Dictionary containing transcript and metadata
        """
        print(f"Transcribing audio: {audio_path}")
        if language:
            print(f"Language: {language}")
        else:
            print("Language: Auto-detect")
        print("This may take a while depending on audio length...")
        
        start_time = time.time()
        result = self.whisper_model.transcribe(
            audio_path, 
            fp16=False,
            language=language,
            verbose=False
        )
        elapsed = time.time() - start_time
        
        transcript = result["text"].strip()
        detected_language = result.get("language", "unknown")
        word_count = len(transcript.split())
        
        print(f"✅ Transcription complete in {elapsed:.1f} seconds")
        print(f"   Detected language: {detected_language}")
        print(f"   Words transcribed: {word_count}\n")
        
        return {
            "text": transcript,
            "language": detected_language,
            "duration": elapsed,
            "word_count": word_count
        }
    
    def chunk_text(self, text: str, max_tokens: int = 800) -> List[str]:
        """Split text into chunks safe for the model"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for word in words:
            # Estimate tokens (rough approximation)
            token_count = max(1, len(word) // 4)
            if current_tokens + token_count > max_tokens and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_tokens = token_count
            else:
                current_chunk.append(word)
                current_tokens += token_count
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def summarize(self, text: str, max_length: int = 150, min_length: int = 50) -> str:
        """Summarize text, handling long inputs automatically"""
        print("Summarizing transcript...")
        
        # Check if text is already short enough
        word_count = len(text.split())
        if word_count <= 100:
            print("Text is already brief, returning as-is")
            return text
        
        iteration = 0
        current_text = text
        
        while True:
            iteration += 1
            print(f"Summarization iteration {iteration}")
            
            # Chunk the text
            chunks = self.chunk_text(current_text, max_tokens=800)
            print(f"  Split into {len(chunks)} chunks")
            
            if len(chunks) == 1 and len(current_text.split()) <= 300:
                # Single chunk, small enough - do final summarization
                try:
                    summary = self.summarizer(
                        current_text,
                        max_length=max_length,
                        min_length=min(min_length, len(current_text.split()) // 3),
                        do_sample=False,
                        clean_up_tokenization_spaces=True
                    )[0]['summary_text']
                    print("✅ Summary complete!\n")
                    return summary
                except Exception as e:
                    print(f"❌ Error in final summarization: {e}")
                    return current_text
            
            # Summarize each chunk
            summaries = []
            for i, chunk in enumerate(tqdm(chunks, desc="  Processing chunks")):
                try:
                    chunk_summary = self.summarizer(
                        chunk, 
                        max_length=min(max_length, len(chunk.split()) // 2), 
                        min_length=min(min_length, len(chunk.split()) // 4), 
                        do_sample=False,
                        clean_up_tokenization_spaces=True
                    )[0]['summary_text']
                    summaries.append(chunk_summary)
                except Exception as e:
                    print(f"Warning: Error summarizing chunk {i+1}: {e}")
                    # Include original chunk if summarization fails
                    summaries.append(chunk)
            
            # Combine summaries
            current_text = " ".join(summaries)
            word_count = len(current_text.split())
            print(f"  Combined summary: {word_count} words")
            
            # Check if we need another iteration
            if word_count <= 200 or len(chunks) == 1:
                print("✅ Summary complete!\n")
                break
            elif iteration >= 3:  # Prevent infinite loops
                print("⚠️  Maximum iterations reached, using current summary\n")
                break
        
        return current_text
    
    def process_video(self, 
                     video_source: str,
                     output_dir: str = "output",
                     keep_audio: bool = False,
                     keep_video: bool = False,
                     language: Optional[str] = None,
                     max_summary_length: int = 150) -> dict:
        """
        Complete pipeline to process a video
        
        Args:
            video_source: Either a file path or YouTube URL
            output_dir: Directory to save outputs
            keep_audio: Whether to keep extracted audio file
            keep_video: Whether to keep downloaded video file
            language: Language code for transcription
            max_summary_length: Maximum length for summary
            
        Returns:
            Dictionary with paths to generated files and metadata
        """
        print("🎬 Starting video processing pipeline\n")
        
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True, parents=True)
        
        # Determine if source is URL or file
        if video_source.startswith(('http://', 'https://', 'www.')):
            video_path = os.path.join(output_dir, "video.mp4")
            video_path = self.download_video(video_source, video_path)
            is_downloaded = True
        else:
            if not os.path.exists(video_source):
                raise FileNotFoundError(f"Video file not found: {video_source}")
            video_path = video_source
            is_downloaded = False
            print(f"Using local video: {video_path}\n")
        
        # Extract audio
        audio_path = os.path.join(output_dir, "audio.wav")
        audio_path = self.extract_audio(video_path, audio_path)
        
        # Transcribe
        transcription_result = self.transcribe(audio_path, language=language)
        transcript = transcription_result["text"]
        
        # Summarize
        summary = self.summarize(transcript, max_length=max_summary_length)
        
        # Save outputs
        transcript_path = os.path.join(output_dir, "transcript.txt")
        summary_path = os.path.join(output_dir, "summary.txt")
        metadata_path = os.path.join(output_dir, "metadata.txt")
        
        # Write transcript
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(transcript)
        print(f"📝 Transcript saved to: {transcript_path}")
        
        # Write summary
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary)
        print(f"📄 Summary saved to: {summary_path}")
        
        # Write metadata
        metadata = f"""Video Processing Results
========================
Source: {video_source}
Detected Language: {transcription_result['language']}
Transcription Duration: {transcription_result['duration']:.1f} seconds
Word Count: {transcription_result['word_count']}
Summary Length: {len(summary.split())} words
Processing Date: {time.strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        with open(metadata_path, "w", encoding="utf-8") as f:
            f.write(metadata)
        print(f"📊 Metadata saved to: {metadata_path}")
        
        # Display results in terminal
        print("\n" + "="*60)
        print("📝 TRANSCRIPT")
        print("="*60)
        print(transcript)
        print("\n" + "="*60)
        print("📄 SUMMARY")
        print("="*60)
        print(summary)
        print("="*60)
        
        # Clean up temporary files
        cleanup_count = 0
        if not keep_audio and os.path.exists(audio_path):
            os.remove(audio_path)
            print(f"🗑️  Removed temporary audio file")
            cleanup_count += 1
        
        if not keep_video and is_downloaded and os.path.exists(video_path):
            os.remove(video_path)
            print(f"🗑️  Removed downloaded video file")
            cleanup_count += 1
        
        if cleanup_count > 0:
            print()
        
        print("✨ Processing complete!")
        
        return {
            "transcript": transcript_path,
            "summary": summary_path,
            "metadata": metadata_path,
            "audio": audio_path if keep_audio else None,
            "video": video_path if keep_video or not is_downloaded else None,
            "stats": {
                "language": transcription_result["language"],
                "word_count": transcription_result["word_count"],
                "summary_length": len(summary.split())
            }
        }


def interactive_mode():
    """Interactive mode - prompts user for input"""
    print("🎥 Video Processing Pipeline")
    print("=" * 40)
    print("📹 Paste a YouTube URL or enter a local video file path")
    print("💡 Examples:")
    print("   • https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    print("   • C:\\Videos\\my_video.mp4")
    print("   • video.mp4")
    print("-" * 40)
    
    # Get video source
    video_source = input("🔗 Enter URL or file path: ").strip()
    if not video_source:
        print("❌ No input provided. Exiting.")
        return
    
    # Get whisper model choice
    print("\n🎤 Choose Whisper model quality:")
    print("   1. tiny   - Fastest, least accurate")
    print("   2. base   - Good balance (recommended)")
    print("   3. small  - Better accuracy")
    print("   4. medium - High accuracy")
    print("   5. large  - Best accuracy, slowest")
    
    model_choice = input("Choose (1-5, default=2): ").strip() or "2"
    model_map = {"1": "tiny", "2": "base", "3": "small", "4": "medium", "5": "large"}
    whisper_model = model_map.get(model_choice, "base")
    
    # Ask about keeping files
    keep_files = input("\n💾 Keep downloaded files? (y/n, default=n): ").strip().lower()
    keep_audio = keep_files in ['y', 'yes']
    keep_video = keep_files in ['y', 'yes']
    
    print(f"\n⚙️  Configuration:")
    print(f"   Source: {video_source}")
    print(f"   Model: {whisper_model}")
    print(f"   Keep files: {'Yes' if keep_files in ['y', 'yes'] else 'No'}")
    print("\n🚀 Starting processing...")
    print("=" * 40)
    
    # Initialize processor
    try:
        processor = VideoProcessor(whisper_model=whisper_model)
    except Exception as e:
        print(f"❌ Failed to initialize processor: {e}")
        return
    
    # Process video
    try:
        results = processor.process_video(
            video_source,
            output_dir="output",
            keep_audio=keep_audio,
            keep_video=keep_video,
            language=None,  # Auto-detect
            max_summary_length=150
        )
        
        print(f"\n📁 Files saved to: output/")
        for key, path in results.items():
            if key != "stats" and path:
                print(f"   {key.title()}: {path}")
        
        # Display stats
        stats = results["stats"]
        print(f"\n📈 Final Statistics:")
        print(f"   Language: {stats['language']}")
        print(f"   Original: {stats['word_count']} words")
        print(f"   Summary: {stats['summary_length']} words")
        print(f"   Compression: {stats['summary_length']/stats['word_count']*100:.1f}%")
        
        # Ask if user wants to process another video
        print("\n" + "="*50)
        another = input("🔄 Process another video? (y/n): ").strip().lower()
        if another in ['y', 'yes']:
            print("\n")
            interactive_mode()  # Recursive call for another video
        else:
            print("👋 Thanks for using Video Pipeline!")
                
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")


def main():
    """Main function - supports both command line and interactive modes"""
    if len(sys.argv) == 1:
        # No arguments provided - run interactive mode
        interactive_mode()
        return
    
    # Command line mode (original functionality)
    parser = argparse.ArgumentParser(
        description="Process video files or YouTube videos into transcripts and summaries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
  %(prog)s video.mp4 --whisper-model large
  %(prog)s video.mp4 --keep-audio --keep-video -o results
  %(prog)s video.mp4 --language en --max-summary-length 200
        """
    )
    
    parser.add_argument(
        "source",
        help="Video file path or YouTube URL"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="output",
        help="Output directory for generated files (default: output)"
    )
    parser.add_argument(
        "--whisper-model", "-w",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base)"
    )
    parser.add_argument(
        "--language", "-l",
        help="Language code for transcription (e.g., 'en', 'es', 'fr'). Auto-detect if not specified."
    )
    parser.add_argument(
        "--keep-audio",
        action="store_true",
        help="Keep extracted audio file"
    )
    parser.add_argument(
        "--keep-video",
        action="store_true",
        help="Keep downloaded video file (for YouTube URLs)"
    )
    parser.add_argument(
        "--max-summary-length",
        type=int,
        default=150,
        help="Maximum length for summary chunks (default: 150)"
    )
    parser.add_argument(
        "--version",
        action="version",
        version="Video Pipeline 2.0"
    )
    
    args = parser.parse_args()
    
    print("🎥 Video Processing Pipeline")
    print("=" * 30)
    
    # Initialize processor
    try:
        processor = VideoProcessor(whisper_model=args.whisper_model)
    except Exception as e:
        print(f"❌ Failed to initialize processor: {e}")
        sys.exit(1)
    
    # Process video
    try:
        results = processor.process_video(
            args.source,
            output_dir=args.output_dir,
            keep_audio=args.keep_audio,
            keep_video=args.keep_video,
            language=args.language,
            max_summary_length=args.max_summary_length
        )
        
        print(f"\n📊 Final Results:")
        print(f"   Output Directory: {args.output_dir}")
        for key, path in results.items():
            if key != "stats" and path:
                print(f"   {key.title()}: {path}")
        
        # Display stats
        stats = results["stats"]
        print(f"\n📈 Statistics:")
        print(f"   Language: {stats['language']}")
        print(f"   Transcript: {stats['word_count']} words")
        print(f"   Summary: {stats['summary_length']} words")
        print(f"   Compression: {stats['summary_length']/stats['word_count']*100:.1f}%")
                
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()