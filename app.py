import os
from flask import Flask, request, render_template
from predict import model_predict

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            result = model_predict(file_path)
            return render_template('index.html', prediction=result)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
