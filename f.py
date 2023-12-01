import os
import shutil
import numpy as np
from docx import Document
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def convert_docx_to_pdf(docx_path, pdf_path):
.
   
    doc = Document(docx_path)
    pdf_canvas = canvas.Canvas(pdf_path, pagesize=letter)

    for paragraph in doc.paragraphs:
        pdf_canvas.drawString(72, 800, paragraph.text)
        pdf_canvas.showPage()

    pdf_canvas.save()

def convert_all_docs_to_pdfs(directory):
  
    pdf_directory = os.path.join(directory, 'PDFs')
    os.makedirs(pdf_directory, exist_ok=True)

    for file in os.listdir(directory):
        if file.endswith('.docx'):
            docx_path = os.path.join(directory, file)
            pdf_path = os.path.join(pdf_directory, file.replace('.docx', '.pdf'))
            convert_docx_to_pdf(docx_path, pdf_path)
            print(f"Converted {file} to {pdf_path}")

def neural_network_conversion(docx_path, pdf_path):
    h: Path to save the PDF file.
    """
 
    model = Sequential()
    model.add(Dense(10, input_dim=1, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error')

 
    X = np.random.rand(100, 1)
    y = X * 2 + 1 + np.random.randn(100, 1) * 0.1

   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

   
    model.fit(X_train, y_train, epochs=10, verbose=0)


    pdf_content = model.predict(X_test)

    pdf_canvas = canvas.Canvas(pdf_path, pagesize=letter)
    for prediction in pdf_content:
        pdf_canvas.drawString(72, 800, f"Prediction: {prediction[0]:.2f}")
        pdf_canvas.showPage()

    pdf_canvas.save()

if __name__ == "__main__":
    print('\nPlease note that this will overwrite any existing PDF files')
    print('For best results, close Microsoft Word before proceeding')
    input('Press enter to continue.')

    current_directory = os.getcwd()

    docx_directory = os.path.join(current_directory, 'DOCX')
    os.makedirs(docx_directory, exist_ok=True)

    
    for file in os.listdir(current_directory):
        if file.endswith(('.doc', '.docx', '.tmd')):
            file_path = os.path.join(current_directory, file)
            new_path = os.path.join(docx_directory, file)
            shutil.move(file_path, new_path)

    convert_all_docs_to_pdfs(docx_directory)

  
    neural_network_conversion(os.path.join(docx_directory, 'example.docx'),
                              os.path.join(docx_directory, 'example_neural_network.pdf'))

    print('\nConversion finished')
