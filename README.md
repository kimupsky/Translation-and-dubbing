이 프로젝트는 로컬 영상 또는 음성 파일로부터

STT → 화자 분리 → 감성 분석 → 번역 → SRT 생성
의 전체 파이프라인을 자동 처리하는 Python 스크립트입니다.

Whisper, pyannote.audio, Hugging Face, Gemini API 등을 활용하여
정확하고 풍부한 자막(SRT) 생성 경험을 제공합니다.

✨ 주요 기능

🔉 음성 → 텍스트(STT)
OpenAI Whisper 모델을 사용하여 고품질 텍스트 변환 수행

🗣️ 화자 분리(Speaker Diarization)
pyannote.audio 모델로 발화별 화자를 자동 식별 및 라벨링

😊 감성 분석(Sentiment Analysis)
Transformers 기반 감성 분석 모델로 각 발화의 감정(긍정/부정/중립) 분석

🌐 Gemini 번역
Google Gemini API를 사용하여 지정 언어로 번역

🎬 SRT 자막 파일 생성
시간 정보 + 화자 정보 + 감성 정보가 포함된 고품질 자막 생성

🎵 오디오 추출
영상 파일에서 자동으로 오디오 트랙만 추출

📦 사전 준비 사항

이 스크립트를 실행하려면 다음이 필요합니다:

Python 3.8+

FFmpeg (PATH에 반드시 등록)
👉 https://ffmpeg.org/download.html

Hugging Face Access Token (Read 권한)
👉 https://huggingface.co/settings/tokens

Google Gemini API Key

🔧 설치
1. 저장소 클론
git clone [YOUR_REPOSITORY_URL]
cd [YOUR_REPOSITORY_NAME]

2. 패키지 설치

가상환경 활성화 후:

pip install -r requirements.txt


또는 직접 설치:

pip install openai-whisper pyannote.audio transformers pydub soundfile ffmpeg-python torch google-generativeai


💡 참고: GPU 사용 시 torch는 CUDA 버전으로 설치해야 합니다.
👉 https://pytorch.org/get-started/locally/

🔑 환경 변수 설정
Windows (CMD)
set GEMINI_API_KEY="your_gemini_api_key"
set HF_TOKEN="hf_your_huggingface_token"

Windows (PowerShell)
$env:GEMINI_API_KEY="your_gemini_api_key"
$env:HF_TOKEN="hf_your_huggingface_token"

Linux / macOS
export GEMINI_API_KEY="your_gemini_api_key"
export HF_TOKEN="hf_your_huggingface_token"

▶️ 사용법
python gemini_translate_whisper_diarization.py <filepath> [-l <language>] [-o <output_dir>]

파라미터 설명
옵션	설명
<filepath>	처리할 영상 또는 음성 파일 경로 (필수)
-l, --language	번역 대상 언어 (기본값: Korean)
-o, --output_dir	출력 디렉토리 설정 (기본값: output)
예시
python gemini_translate_whisper_diarization.py "C:\path\to\video.mp4" -l Korean

📁 출력 파일 구성

스크립트 실행 후 다음과 같은 파일들이 생성됩니다:

[filename]_audio_original.mp3
→ 영상에서 추출된 원본 오디오

[filename]_stt_result.txt
→ Whisper STT 전체 텍스트

[filename]_utterances_with_timestamps_speaker_sentiment.txt
→ 발화별 시작·종료 시간 + 화자 + 감성 정보

[filename]_translated.txt
→ Gemini 번역 결과 텍스트

[filename]_final_subtitle.srt
→ 최종 생성된 SRT 자막 파일 (화자/감성 포함)
   23
   24 `pyannote.audio`와 `whisper`는 `torch`를 사용하여 GPU 가속을 활용할 수 있습니다. GPU를 사용하려면 NVIDIA 드라이버, CUDA,
      cuDNN이 올바르게 설치되어 있어야 하며, `torch`를 CUDA 지원 버전으로 설치해야 합니다. CPU만 사용하는 경우에도 스크립트는
      작동하지만, 처리 속도가 느릴 수 있습니다.
