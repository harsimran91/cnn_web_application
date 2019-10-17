def load():
    from keras.models import model_from_json
    import numpy as np
    from keras.preprocessing import image
    import tensorflow as tf
    json_file = open('classifier.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("classifier.h5")
    print("Loaded model from disk")
    loaded_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    graph = tf.get_default_graph()
    return loaded_model,graph