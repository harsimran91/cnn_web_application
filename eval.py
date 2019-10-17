#evaluate

def eval(a,loaded_model):
    
    from keras.preprocessing import image
    import numpy as np
    test_image = image.load_img(a,target_size=(64,64))
    test_image = image.img_to_array(test_image,data_format="channels_first")
    test_image = np.expand_dims(test_image,axis=0)
    result = loaded_model.predict_proba(test_image)
    if result == 1:
        return print("This is a Dog Image")
    else:
        return print("This is a Cat Image")