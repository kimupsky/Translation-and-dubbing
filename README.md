# Translation-and-dubbing
3 이 프로젝트는 로컬 영상 또는 음성 파일에서 음성을 텍스트로 변환(STT)하고, 화자를 분리하며, 각 발화의 감성을 분석한 후, Gemini
      API를 사용하여 번역하고 SRT 자막 파일을 생성하는 Python 스크립트입니다.
    4
    5 ## 주요 기능
    6
    7 *   **음성-텍스트 변환 (STT)**: OpenAI의 Whisper 모델을 사용하여 고품질의 음성-텍스트 변환을 수행합니다.
    8 *   **화자 분리 (Speaker Diarization)**: `pyannote.audio`를 사용하여 오디오 내의 여러 화자를 식별하고 각 발화의 화자를
      라벨링합니다.
    9 *   **감성 분석 (Sentiment Analysis)**: Hugging Face Transformers 라이브러리를 사용하여 각 발화 텍스트의 감성(긍정, 부정,
      중립)을 분석합니다.
   10 *   **Gemini 번역**: Google Gemini API를 사용하여 STT 결과를 지정된 언어로 번역합니다.
   11 *   **SRT 자막 생성**: 번역된 텍스트와 시간 정보를 기반으로 화자 및 감성 정보가 포함된 SRT 자막 파일을 생성합니다.
   12 *   **오디오 추출**: 영상 파일에서 음성 트랙을 추출합니다.
   13
   14 ## 사전 준비 사항
   15
   16 이 스크립트를 실행하려면 다음이 필요합니다:
   17
   18 1.  **Python 3.8+**: Python이 설치되어 있어야 합니다.
   19 2.  **FFmpeg**: 오디오/비디오 처리를 위해 FFmpeg가 시스템에 설치되어 있고 PATH에 등록되어 있어야 합니다. [FFmpeg 공식
      웹사이트](https://ffmpeg.org/download.html)에서 다운로드할 수 있습니다.
   20 3.  **Hugging Face 계정 및 Access Token**: `pyannote.audio` 및 `transformers` 모델 다운로드 및 사용을 위해 필요합니다. [
      Hugging Face 웹사이트](https://huggingface.co/settings/tokens)에서 "Read" 권한의 토큰을 발급받으세요.
   21 4.  **Gemini API Key**: 텍스트 번역을 위해 Google Gemini API 키가 필요합니다.
   22
   23 ## 설치
   24
   25 1.  **저장소 클론**:

      git clone [YOUR_REPOSITORY_URL]
      cd [YOUR_REPOSITORY_NAME]


   1
   2 2.  **Python 패키지 설치**:
   3     가상 환경을 활성화한 후 다음 명령어를 실행하여 필요한 라이브러리들을 설치합니다.

      pip install -r requirements.txt
  또는 개별 설치:
  pip install openai-whisper pyannote.audio transformers pydub soundfile ffmpeg-python torch google-generativeai


    1     *   **참고**: `torch`는 GPU 사용을 위해 CUDA 버전으로 설치할 수 있습니다. 자세한 내용은 [PyTorch 공식 웹사이트](
      https://pytorch.org/get-started/locally/)를 참조하세요.
    2
    3 ## 환경 변수 설정
    4
    5 스크립트 실행 전에 다음 환경 변수를 설정해야 합니다.
    6
    7 *   **`GEMINI_API_KEY`**: Google Gemini API 키
    8 *   **`HF_TOKEN`**: Hugging Face Access Token (Read 권한)
    9
   10 **Windows (명령 프롬프트):**

  set GEMINI_API_KEY="your_gemini_api_key"
  set HF_TOKEN="hf_your_huggingface_token"

   1
   2 **Windows (PowerShell):**

  $env:GEMINI_API_KEY="your_gemini_api_key"
  $env:HF_TOKEN="hf_your_huggingface_token"


   1
   2 **Linux/macOS:**

  export GEMINI_API_KEY="your_gemini_api_key"
  export HF_TOKEN="hf_your_huggingface_token"


   1 
   2 ## 사용법
   3 
   4 스크립트는 로컬 영상 또는 음성 파일을 입력받아 처리합니다.

  python gemini_translate_whisper_diarization.py <filepath> [-l <language>] [-o <output_dir>]


   1 
   2 *   `<filepath>`: 처리할 로컬 영상 또는 음성 파일의 경로 (필수).
   3 *   `-l`, `--language`: 번역할 목표 언어 (기본값: `Korean`). 예: `English`, `Japanese`.
   4 *   `-o`, `--output_dir`: 모든 결과물이 저장될 기본 디렉토리 (기본값: `output`).
   5 
   6 **예시:**

  python gemini_translate_whisper_diarization.py "C:\path\to\your_video.mp4" -l Korean


    1 
    2 ## 출력 파일
    3 
    4 스크립트 실행이 완료되면 지정된 출력 디렉토리(`output/your_video_whisper_diarization_1/` 등) 내에 다음 파일들이 생성됩니다:
    5 
    6 *   `[filename]_audio_original.mp3`: 원본 영상/음성에서 추출된 오디오 파일.
    7 *   `[filename]_stt_result.txt`: Whisper로 변환된 전체 텍스트.
    8 *   `[filename]_utterances_with_timestamps_speaker_sentiment.txt`: 각 발화의 시작/종료 시간, 화자, 감성 정보 및 텍스트가
      포함된 파일.
    9 *   `[filename]_translated.txt`: Gemini API로 번역된 전체 텍스트.
   10 *   `[filename]_final_subtitle.srt`: 화자 및 감성 정보가 포함된 최종 SRT 자막 파일.
   11 
   12 ## 중요 사항 및 문제 해결
   13 
   14 ### `pyannote/speaker-diarization` 모델 접근 오류 (403 Client Error)
   15 
   16 `pyannote/speaker-diarization` 모델은 Hugging Face Hub의 **게이트(gated) 모델**입니다. 이 모델을 사용하려면 다음 단계를
      따라야 합니다:
   17
   18 1.  [Hugging Face 모델 페이지](https://huggingface.co/pyannote/speaker-diarization)를 방문합니다.
   19 2.  페이지에서 모델 사용을 위한 약관을 읽고 동의 버튼을 클릭하여 접근 권한을 요청합니다. 이 과정에서 연락처 정보 공유에
      동의해야 합니다.
   20 3.  접근 권한이 승인되면, 설정된 `HF_TOKEN` 환경 변수를 통해 모델을 다운로드하고 사용할 수 있습니다.
   21
   22 ### `torch` 및 GPU 사용
   23
   24 `pyannote.audio`와 `whisper`는 `torch`를 사용하여 GPU 가속을 활용할 수 있습니다. GPU를 사용하려면 NVIDIA 드라이버, CUDA,
      cuDNN이 올바르게 설치되어 있어야 하며, `torch`를 CUDA 지원 버전으로 설치해야 합니다. CPU만 사용하는 경우에도 스크립트는
      작동하지만, 처리 속도가 느릴 수 있습니다.
