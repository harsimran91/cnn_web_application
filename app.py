from flask import Flask,render_template,request,flash,redirect,url_for
from load import load
from eval import eval
from werkzeug.utils import secure_filename
import os
from keras.preprocessing import image
import numpy as np


cache='./static'
app = Flask(__name__)
loaded_model,graph=load()
app.config['UPLOAD_FOLDER'] = cache
@app.route('/',methods=['GET','POST'])
def index():
    if request.method=='POST':
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect('/')
        else:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            test_image = image.load_img(os.path.join(app.config['UPLOAD_FOLDER'], filename),target_size=(64,64))
            test_image = image.img_to_array(test_image,data_format="channels_first")
            test_image = np.expand_dims(test_image,axis=0)
            with graph.as_default():
                result1 = loaded_model.predict(test_image)
                if result1 == 1:
                    result = "This is a Dog Image"
                    
                else:
                    print(result1)
                    result = "This is a Cat Image"
                    
                print(result)
                return render_template('index.html',result =result,img=os.path.join(app.config['UPLOAD_FOLDER'], filename))
    else:
        return render_template('index.html',result=None,img=None)



if __name__=='__main__':
    app.run(debug=True)