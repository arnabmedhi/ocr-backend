from flask import Flask, request, jsonify
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import re

# Initialize the Flask app
app = Flask(__name__)

# Load the TrOCR model
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
processor = ViTImageProcessor.from_pretrained("microsoft/trocr-base-handwritten")
tokenizer = AutoTokenizer.from_pretrained("microsoft/trocr-base-handwritten")

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    image = Image.open(file).convert("RGB")

    # Perform OCR
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    extracted_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Optional: Parse extracted text into structured data
    structured_data = parse_ocr_text(extracted_text)

    return jsonify({"extracted_text": extracted_text, "structured_data": structured_data})

def parse_ocr_text(text):
    """Parses the extracted text into structured fields."""
    data = {}
    data['Name'] = re.search(r"Name:\s*(.*)", text).group(1).strip() if "Name:" in text else None
    data['DOB'] = re.search(r"DOB:\s*(.*)", text).group(1).strip() if "DOB:" in text else None
    data['Address'] = re.search(r"Address:\s*(.*)", text).group(1).strip() if "Address:" in text else None
    return data

if __name__ == '__main__':
    app.run(debug=True)