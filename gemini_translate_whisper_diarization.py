# 1. 표준 라이브러리
import argparse
import os
import re
import sys
import time
import shutil
import math # Added for segmenting audio

# 2. 외부 라이브러리
import google.generativeai as genai
import ffmpeg
import torch
from pydub import AudioSegment
import numpy as np
import soundfile as sf # For saving audio segments

# New imports for Whisper, Pyannote, Transformers
import whisper
from pyannote.audio import Pipeline
from transformers import pipeline as hf_pipeline
import huggingface_hub # Added for explicit login

# --- 설정 부분 ---
# API 키는 환경 변수에서 불러옵니다.
# 예: export GEMINI_API_KEY="your_key"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Hugging Face Access Token (for pyannote.audio and some transformers models)
# 예: export HF_TOKEN="hf_YOUR_TOKEN_HERE"
HF_TOKEN = os.getenv("HF_TOKEN")

# --- 메인 로직 ---

def setup_environment(output_dir):
    """출력 디렉토리를 생성하고 API 키 유효성을 검사합니다."""
    print("Step 1: 환경 설정 시작...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"'{output_dir}' 디렉토리를 생성했습니다.")
    
    if not GEMINI_API_KEY:
        print("오류: Gemini API 키가 설정되지 않았습니다. 환경 변수 'GEMINI_API_KEY'를 설정해주세요.")
        return False
    genai.configure(api_key=GEMINI_API_KEY)
    print("Gemini API가 성공적으로 설정되었습니다.")

    if not HF_TOKEN:
        print("오류: Hugging Face Access Token이 설정되지 않았습니다. 환경 변수 'HF_TOKEN'을 설정해주세요.")
        print("      pyannote.audio 및 일부 Hugging Face 모델 사용에 필요합니다. (https://huggingface.co/settings/tokens 에서 발급 가능)")
        return False
    
    try:
        huggingface_hub.login(token=HF_TOKEN)
        print("Hugging Face에 성공적으로 로그인했습니다.")
    except Exception as e:
        print(f"오류: Hugging Face 로그인 중 문제 발생. ({e})")
        return False
    
    return True

def sanitize_filename(title):
    """파일 이름으로 사용하기에 안전한 문자열로 변환합니다."""
    return re.sub(r'[\\/*?:"<>|]', "", title).replace(" ", "_")

def extract_audio(video_path, sanitized_title, output_dir):
    """영상 또는 음성 파일에서 음성(mp3)을 추출/변환합니다."""
    print(f"\nStep 2: 원본 음성 추출 시작... (파일: {video_path})")
    if not video_path:
        return None
    
    audio_path = os.path.join(output_dir, f"{sanitized_title}_audio_original.mp3")
    
    try:
        (
            ffmpeg
            .input(video_path)
            .output(audio_path, acodec='mp3', audio_bitrate='192k')
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        print(f"원본 음성 추출 완료! -> {audio_path}")
        return audio_path
    except ffmpeg.Error as e:
        print(f"오류: 음성 처리에 실패했습니다. FFMPEG가 설치 및 PATH에 등록되었는지 확인하세요.")
        print(f"FFMPEG stderr: {e.stderr.decode('utf8')}")
        return None
    except Exception as e:
        print(f"오류: 음성 처리 중 알 수 없는 예외 발생. ({e})")
        return None

# remove_background_noise 함수는 현재 STT 변경과 직접 관련이 없으므로 제거합니다.
# def remove_background_noise(input_audio_path, sanitized_title, output_dir):
#     """오디오 파일에서 배경 소음을 제거합니다."""
#     ...

def transcribe_audio_with_whisper_diarization(audio_path, sanitized_title, output_dir):
    """
    Whisper, pyannote.audio, Hugging Face Transformers를 사용하여
    음성 파일을 텍스트로 변환하고, 화자 분리 및 감성 분석을 수행합니다.
    """
    print(f"\nStep 3: Whisper, Pyannote, Transformers를 이용한 STT, 화자 분리, 감성 분석 시작...")
    if not audio_path:
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"사용할 디바이스: {device}")

    # 1. Pyannote.audio를 이용한 화자 분리 (Speaker Diarization)
    print("  - Pyannote.audio 파이프라인 로드 및 화자 분리 시작...")
    try:
        diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", revision="2.1") # revision 인자 사용
        diarization_pipeline.to(torch.device(device))
        diarization = diarization_pipeline(audio_path)
    except Exception as e:
        print(f"오류: pyannote.audio 화자 분리 중 문제 발생. ({e})")
        print("      Hugging Face Access Token이 올바르게 설정되었는지 확인하세요.")
        return None

    # 2. Whisper 모델 로드
    print("  - Whisper 모델 로드 (large-v3)...")
    try:
        whisper_model = whisper.load_model("large-v3", device=device)
    except Exception as e:
        print(f"오류: Whisper 모델 로드 중 문제 발생. ({e})")
        return None

    # 3. Hugging Face 감성 분석 파이프라인 로드 (한국어 모델)
    print("  - Hugging Face 감성 분석 파이프라인 로드 (klue/roberta-base-sentiment)...")
    try:
        sentiment_analyzer = hf_pipeline("sentiment-analysis", model="klue/roberta-base-sentiment", device=0 if device == "cuda" else -1)
    except Exception as e:
        print(f"오류: Hugging Face 감성 분석 모델 로드 중 문제 발생. ({e})")
        print("      'klue/roberta-base-sentiment' 모델이 한국어 감성 분석에 적합합니다.")
        return None

    all_utterances = []
    audio = AudioSegment.from_file(audio_path)

    # 임시 오디오 세그먼트 저장 디렉토리
    temp_audio_segments_dir = os.path.join(output_dir, "temp_whisper_segments")
    os.makedirs(temp_audio_segments_dir, exist_ok=True)

    print("  - 각 화자 세그먼트별로 Whisper STT 및 감성 분석 수행...")
    for i, speech_segment in enumerate(diarization.itertracks(yield_label=True)):
        speaker = speech_segment.set_label
        start_time_ms = int(speech_segment.segment.start * 1000)
        end_time_ms = int(speech_segment.segment.end * 1000)

        # 오디오 세그먼트 추출 및 저장
        segment_audio = audio[start_time_ms:end_time_ms]
        segment_audio_path = os.path.join(temp_audio_segments_dir, f"segment_{i}_{speaker}.wav")
        segment_audio.export(segment_audio_path, format="wav")

        # Whisper STT
        try:
            print(f"    - [{speaker}] {format_time(start_time_ms)} - {format_time(end_time_ms)} STT 진행 중...")
            result = whisper_model.transcribe(segment_audio_path, language="ko", fp16=False if device == "cpu" else True) # 한국어 지정
            text = result["text"].strip()
        except Exception as e:
            print(f"경고: Whisper STT 중 문제 발생. 해당 세그먼트 건너뜀. ({e})")
            text = ""

        # 감성 분석
        sentiment = "Neutral"
        if text:
            try:
                sentiment_result = sentiment_analyzer(text)
                # 'LABEL_0': 중립, 'LABEL_1': 긍정, 'LABEL_2': 부정 (klue/roberta-base-sentiment 기준)
                label_map = {'LABEL_0': 'Neutral', 'LABEL_1': 'Positive', 'LABEL_2': 'Negative'}
                sentiment = label_map.get(sentiment_result[0]['label'], 'Neutral')
            except Exception as e:
                print(f"경고: 감성 분석 중 문제 발생. 기본값 'Neutral' 사용. ({e})")
        
        if text: # 텍스트가 있는 경우에만 추가
            all_utterances.append({
                'start': start_time_ms,
                'end': end_time_ms,
                'text': text,
                'speaker': speaker,
                'sentiment': sentiment
            })
            print(f"      -> 텍스트: \"{text[:50]}...\", 감정: {sentiment}")

    # 임시 파일 정리
    shutil.rmtree(temp_audio_segments_dir)
    print("  - 임시 오디오 세그먼트 파일 정리 완료.")

    # STT 전체 텍스트 파일 저장
    stt_output_path = os.path.join(output_dir, f"{sanitized_title}_stt_result.txt")
    with open(stt_output_path, "w", encoding="utf-8") as f:
        f.write("\n".join([u['text'] for u in all_utterances]))
    print(f"STT 전체 텍스트가 파일에 저장되었습니다. -> {stt_output_path}")

    # 대사별 시간, 화자, 감정 정보 파일 저장
    utterances_path = os.path.join(output_dir, f"{sanitized_title}_utterances_with_timestamps_speaker_sentiment.txt")
    with open(utterances_path, "w", encoding="utf-8") as f:
        for utterance in all_utterances:
            start_time_str = format_time(utterance['start'])
            end_time_str = format_time(utterance['end'])
            f.write(f"[{start_time_str} --> {end_time_str}] [Speaker: {utterance['speaker']}] [Sentiment: {utterance['sentiment']}] {utterance['text']}\n")
    print(f"대사별 시간, 화자, 감정 정보가 파일에 저장되었습니다. -> {utterances_path}")

    return all_utterances


def translate_text_with_gemini(utterances, target_language, sanitized_title, output_dir):
    """Gemini API를 사용하여 발화 목록을 번역합니다."""
    print(f"\nStep 4: Gemini를 이용한 자막 번역 시작...")
    if not utterances:
        return None
    
    original_texts = [u['text'] for u in utterances]
    
    try:
        model = genai.GenerativeModel('gemini-2.5-flash') # Assuming it's already 2.5-flash
        
        if len(original_texts) == 1:
            # If only one original segment, send it directly without separator logic
            combined_text = original_texts[0]
            prompt = f"""
            You are a professional subtitle translator.
            Translate the following text into {target_language}.
            Do not add any extra explanations, comments, or introductory phrases.
            Your output should only contain the translated text.

            --- Original Text ---
            {combined_text}
            """
            print("Gemini API에 번역을 요청합니다 (단일 세그먼트)...")
            response = model.generate_content(prompt)
            translated_texts = [response.text.strip()] # Ensure it's a list with one element
        else:
            # For multiple segments, use the separator logic
            separator = "|||---|||"
            combined_text = separator.join(original_texts)
            prompt = f"""
            You are a professional subtitle translator.
            Translate the following text segments into {target_language}.
            The segments are separated by \"{separator}\".
            Maintain the original tone and context for each segment.
            Preserve the \"{separator}\" separator between the translated segments in your output.
            Do not add any extra explanations, comments, or introductory phrases.
            Your output should only contain the translated segments separated by \"{separator}\".

            --- Original Text Segments ---
            {combined_text}
            """
            print("Gemini API에 번역을 요청합니다 (다중 세그먼트)...")
            response = model.generate_content(prompt)
            translated_texts = response.text.strip().split(separator)

        # --- Post-processing to ensure lengths match ---
        if len(translated_texts) != len(original_texts):
            print(f"경고: 원본({len(original_texts)})과 번역본({len(translated_texts)})의 문장 수가 다릅니다. 자막 싱크가 정확하지 않을 수 있습니다.")
            # Attempt to align lengths by padding/truncating
            if len(translated_texts) < len(original_texts):
                while len(translated_texts) < len(original_texts):
                    translated_texts.append("") # Pad with empty strings
            elif len(translated_texts) > len(original_texts):
                translated_texts = translated_texts[:len(original_texts)] # Truncate

        translated_path = os.path.join(output_dir, f"{sanitized_title}_translated.txt")
        full_text = " ".join(translated_texts)
        formatted_text = re.sub(r'([.?!])\s*', r'\1\n', full_text)
        with open(translated_path, "w", encoding="utf-8") as f:
            f.write(formatted_text)
        print(f"번역 완료! -> {translated_path}")
        return translated_texts
    except Exception as e:
        print(f"오류: Gemini 번역 중 문제가 발생했습니다. ({e})")
        return None

def format_time(ms):
    """밀리초를 SRT 타임스탬프 형식(HH:MM:SS,ms)으로 변환합니다."""
    seconds, milliseconds = divmod(ms, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02},{int(milliseconds):03}"

def create_srt_file(utterances, translated_texts, sanitized_title, output_dir):
    """타임스탬프가 있는 발화와 번역된 텍스트로 .srt 자막 파일을 생성합니다."""
    print(f"\nStep 5: .srt 자막 파일 생성 시작...")
    if not utterances or not translated_texts:
        print("오류: 자막을 생성할 데이터가 없습니다.")
        return

    srt_path = os.path.join(output_dir, f"{sanitized_title}_final_subtitle.srt")
    
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, (utterance, translated) in enumerate(zip(utterances, translated_texts)):
            start = format_time(utterance['start'])
            end = format_time(utterance['end'])
            speaker = utterance.get('speaker', None) # Get speaker, if available
            sentiment = utterance.get('sentiment', None) # Get sentiment, if available

            f.write(f"{i + 1}\n")
            f.write(f"{start} --> {end}\n")
            
            prefix = ""
            if sentiment:
                prefix += f"[{sentiment.capitalize()}] "
            if speaker:
                prefix += f"{speaker}: "
            
            f.write(f"{prefix}{translated.strip()}\n\n")

    print(f"자막 파일 생성 완료! -> {srt_path}")

# Coqui TTS 관련 함수는 현재 STT 변경과 직접 관련이 없으므로 제거합니다.
# def create_dubbed_audio(...):
#     ...

def main():
    parser = argparse.ArgumentParser(description="로컬 영상/음성 파일을 번역하여 SRT 자막 또는 더빙 파일을 생성합니다.")
    parser.add_argument("filepath", type=str, help="번역할 로컬 영상 또는 음성 파일의 경로")
    parser.add_argument("-l", "--language", type=str, default="Korean", help="번역할 목표 언어 (예: Korean, Japanese, English)")
    parser.add_argument("-o", "--output_dir", type=str, default="output", help="모든 결과물의 기본 저장 디렉토리")
    # parser.add_argument("--dub", action="store_true", help="이 플래그를 설정하면 더빙 파일을 생성합니다.") # 더빙 기능은 현재 제거

    args = parser.parse_args()

    if not os.path.exists(args.filepath):
        print(f"오류: 파일이 존재하지 않습니다. '{args.filepath}'")
        sys.exit(1)

    # --- 고유 실행 폴더 생성 ---
    base_filename = sanitize_filename(os.path.splitext(os.path.basename(args.filepath))[0])
    i = 1
    while True:
        run_output_dir = os.path.join(args.output_dir, f"{base_filename}_whisper_diarization_{i}")
        if not os.path.exists(run_output_dir):
            break
        i += 1

    if not setup_environment(run_output_dir):
        sys.exit(1)

    original_audio_file = extract_audio(args.filepath, base_filename, run_output_dir)
    if not original_audio_file:
        sys.exit(1)

    # STT에는 원본 오디오 사용 (새로운 함수 호출)
    utterances = transcribe_audio_with_whisper_diarization(original_audio_file, base_filename, run_output_dir)
    
    if not utterances:
        print("STT, 화자 분리 및 감성 분석 작업에 실패하여 다음 단계를 진행할 수 없습니다.")
        sys.exit(1)
        
    translated_texts = translate_text_with_gemini(utterances, args.language, base_filename, run_output_dir)
    
    if not translated_texts:
        print("번역 작업에 실패하여 다음 단계를 진행할 수 없습니다.")
        sys.exit(1)

    create_srt_file(utterances, translated_texts, base_filename, run_output_dir)

    # 더빙 기능은 현재 제거되었으므로 주석 처리
    # if args.dub:
    #     ...
    
    print("\n--- 모든 작업이 완료되었습니다. ---")
    print(f"결과물은 '{run_output_dir}' 폴더에서 확인하세요.")

if __name__ == "__main__":
    main()
