
from flask import Flask, request, jsonify
from flask_cors import CORS
import PyPDF2
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import T5Tokenizer, T5ForConditionalGeneration
from joblib import load
import torch
import logging

app = Flask(__name__)
CORS(app)  # Apply CORS to all routes

# Load tokenizer outside of the function for efficiency
tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Preprocess text function
def preprocess_text(text):
    # Lowercasing
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    # Join tokens back into text
    preprocessed_text = ' '.join(lemmatized_tokens)
    return preprocessed_text

# Load the trained model
model = load('summaryModel.joblib')

# Generate summary function
def generate_summary(input_text):
    # Preprocess the input text
    preprocessed_input = preprocess_text(input_text)
    # Tokenization
    input_tokenized = tokenizer(preprocessed_input, return_tensors="pt")
    # Model inference
    model.eval()
    with torch.no_grad():
        output_ids = model.generate(input_tokenized['input_ids'], max_length=50, num_beams=4, early_stopping=True)
    # Decode the summary
    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return summary

# Route for file upload
logging.basicConfig(level=logging.DEBUG)

@app.route('/api/upload', methods=['POST'])
def upload_file():
    logging.info('Received file upload request')

    if 'file' not in request.files:
        logging.error('No file part')
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        logging.error('No selected file')
        return jsonify({'error': 'No selected file'})

    if file.filename.endswith('.pdf'):
        logging.info('Processing PDF file')
        import pdfplumber

        def extract_text_from_pdf(file):
            text = ''
            with pdfplumber.open(file) as pdf:
              for page in pdf.pages:
                 # Check if the page contains images
                 if page.images:
                # Skip text extraction for pages with images
                   continue
                 else:
                    # Extract text for pages without images
                   text += page.extract_text()
            return text
        text = extract_text_from_pdf(file)
        logging.debug(f'Extracted text: {text}')

        summary = generate_summary(text)
        logging.debug(f'Generated summary: {summary}')

        return jsonify({'text': summary})
    else:
        logging.error('Unsupported file format')
        return jsonify({'error': 'Unsupported file format'})


if __name__ == '__main__':
    app.run(debug=True)  # Run Flask app in debug mode

# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import pdfplumber
# import pytesseract
# from PIL import Image
# import cv2
# import numpy as np
# import re
# import nltk
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# from transformers import T5Tokenizer, T5ForConditionalGeneration
# from joblib import load
# import torch
# import logging

# app = Flask(__name__)
# CORS(app)  # Apply CORS to all routes

# # Initialize tokenizer and model
# tokenizer = T5Tokenizer.from_pretrained('t5-small')
# model = load('summaryModel.joblib')

# # Preprocess text function
# def preprocess_text(text):
#     text = text.lower()  # Lowercasing
#     text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters and digits
#     tokens = word_tokenize(text)  # Tokenization
#     stop_words = set(stopwords.words('english'))
#     filtered_tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
#     lemmatizer = WordNetLemmatizer()
#     lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]  # Lemmatization
#     preprocessed_text = ' '.join(lemmatized_tokens)  # Join tokens back into text
#     return preprocessed_text

# # Function to generate summary
# def generate_summary(input_text):
#     preprocessed_input = preprocess_text(input_text)  # Preprocess the input text
#     input_tokenized = tokenizer(preprocessed_input, return_tensors="pt")  # Tokenization
#     model.eval()
#     with torch.no_grad():
#         output_ids = model.generate(input_tokenized['input_ids'], max_length=50, num_beams=4, early_stopping=True)  # Model inference
#     summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)  # Decode the summary
#     return summary

# # Route for file upload
# logging.basicConfig(level=logging.DEBUG)

# @app.route('/api/upload', methods=['POST'])
# def upload_file():
#     logging.info('Received file upload request')

#     if 'file' not in request.files:
#         logging.error('No file part')
#         return jsonify({'error': 'No file part'})

#     file = request.files['file']

#     if file.filename == '':
#         logging.error('No selected file')
#         return jsonify({'error': 'No selected file'})

#     if file.filename.endswith('.pdf'):
#         logging.info('Processing PDF file')

#         extracted_text = extract_text_from_pdf(file)  # Extract text from PDF
#         logging.debug(f'Extracted text: {extracted_text}')

#         summary = generate_summary(extracted_text)  # Generate summary from extracted text
#         logging.debug(f'Generated summary: {summary}')

#         return jsonify({'text': summary})
#     else:
#         logging.error('Unsupported file format')
#         return jsonify({'error': 'Unsupported file format'})

# def extract_text_from_pdf(file):
#     text = ''
#     with pdfplumber.open(file) as pdf:
#         for page in pdf.pages:
#             page_image = page.to_image()  # Convert page to image
#             img_cv2 = cv2.cvtColor(np.array(page_image.original), cv2.COLOR_RGB2BGR)  # Convert to OpenCV format

#             # Perform OCR on the entire page image
#             ocr_text = pytesseract.image_to_string(Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)))

#             # Detect image regions and exclude them from text extraction
#             image_regions = detect_image_regions(img_cv2)
#             processed_text = process_text_excluding_images(ocr_text, image_regions)

#             text += processed_text  # Append processed text to result

#     return text

# def detect_image_regions(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (5, 5), 0)
#     _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     image_regions = []
#     for contour in contours:
#         x, y, w, h = cv2.boundingRect(contour)
#         area = cv2.contourArea(contour)
#         if area > 1000:
#             image_regions.append((x, y, x + w, y + h))  # Store coordinates of detected image regions

#     return image_regions

# def process_text_excluding_images(ocr_text, image_regions):
#     lines = ocr_text.split('\n')
#     processed_lines = []

#     for line in lines:
#         exclude_line = False
#         for (x1, y1, x2, y2) in image_regions:
#             # Check if line coordinates intersect with any image region
#             if y1 <= line.y_max <= y2:
#                 exclude_line = True
#                 break
#         if not exclude_line:
#             processed_lines.append(line)

#     return '\n'.join(processed_lines)

# if __name__ == '__main__':
#     app.run(debug=True)
