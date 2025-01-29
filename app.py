# pip install transformers torch scipy uroman

from transformers import VitsModel, AutoTokenizer
import torch
import scipy

# 모델과 토크나이저 로드
model = VitsModel.from_pretrained("facebook/mms-tts-kor")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-kor")

text = "안녕하세요. 한국어 텍스트 투 스피치 모델을 실행 중입니다. 제 말 잘 들리죠?"
inputs = tokenizer(text, return_tensors="pt")

# 음성 생성
with torch.no_grad():
    output = model(**inputs).waveform

# 오디오 파일로 저장
scipy.io.wavfile.write("techno.wav", rate=model.config.sampling_rate, data=output[0].numpy())


