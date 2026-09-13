from fastapi import FastAPI, UploadFile, File
import whisper
from deep_translator import GoogleTranslator
import datetime, os

app = FastAPI()
model = whisper.load_model("base") # Model Speech-to-Text gratis & akurat

def format_timestamp(seconds: float) -> str:
    td = datetime.timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours, minutes, secs = total_seconds // 3600, (total_seconds % 3600) // 60, total_seconds % 60
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

@app.post("/generate-srt/")
async def generate_srt(file: UploadFile = File(...)):
    temp_filename = f"temp_{file.filename}"
    with open(temp_filename, "wb") as f:
        f.write(await file.read())
    
    result = model.transcribe(temp_filename, language="ja")
    translator = GoogleTranslator(source='ja', target='id')
    srt_content = ""
    
    for idx, segment in enumerate(result["segments"], start=1):
        start = format_timestamp(segment["start"])
        end = format_timestamp(segment["end"])
        text_id = translator.translate(segment["text"].strip())
        srt_content += f"{idx}\n{start} --> {end}\n{text_id}\n\n"
        
    if os.path.exists(temp_filename):
        os.remove(temp_filename)
        
    return {"srt_result": srt_content}