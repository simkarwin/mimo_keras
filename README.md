#  Custom Data-Generator for multiple-input multiple-output models in TF-keras
Develop your own keras Data-Generator in TF-Keras to load and batch every data type with any format from a massive dataset in computers with limited main or GPU memory... mimo-keras is a package that enables feeding models with any format and any number of inputs and outputs.

MIMO-keras — Never use keras imagedatagenerator to load data in batch
have you ever used image_data_generator() or load_form_directory() to load batch data and feed your deep model in keras? mimo-keras makes data loader quite simple and straightforward even for multiple input/output models or data with formats that are not supported by default in keras.

## mimo-keras is like image_data_generator().load_from_directory(), but better:

1. supports pandas, images and other formats in a generator without needing to define a new data generator for each input or output.
2. It can load data in every format.
3. You can write your own data loader function.
4. You can use your custom preprocessing pipeline without limitation.


```python
from mimo_keras import MIMODataGenerator

def load_and_preprocess_matrix(feature_values, feature_names):
    parameters = dict(zip(feature_names, feature_values))
    matrix = np.load(parameters.get('matrix_path'))
    if len(np.shape(matrix)) == 2:
        matrix = np.expand_dims(matrix, axis=-1)
    matrix = (matrix - np.mean(matrix)) / np.std(matrix)
    return matrix


data = pd.DaraFrame(columns=['year', 'resolution', 'label', 'location_1', 'location_2', 'matrix_path'])
...
# split datahere
...


# first input with shape (m, n, c,)
matrix_input = ('matrix', # Name of the model IO.
                ['matrix_path'], # These column names and their values are sent to the your function for each sample in batch generation.
                load_and_preprocess_matrix # A function or callable class to load data and preprocessing. Use 'raw' to send values to the model IO directly.
               )
# second input with shape (4,1,)
metadata_input = ('metadata', ['year', 'resolution', 'location_1', 'location_2'], 'raw')
# output (this model has only one output but you can define multiple outputs like inputs)
output = ('metadata', ['label'], 'raw') # binarry classification
train_generator = MIMODataGenerator(data_table = data_train
                                    model_inputs=[matrix_input, metadata_input],
                                    model_outputs=[output],
                                    shuffle=True,
                                    batch_size=batch_size
                                    )

validation_generaetor = MIMODataGenerator(data_table = data_validation
                                          model_inputs=[matrix_input, metadata_input],
                                          model_outputs=[output],
                                          shuffle=False,
                                          batch_size=batch_size
                                          )

model.fit(generator = train_generator,
          validation_data = validation_generator,
          epochs=EPOCHS
          )
```

for more complicated model you can use only sample_id to generate data foe each IO. for example:


```python
from mimo_keras import MIMODataGenerator

def load_user_history(feature_values, feature_names):
    parameters = dict(zip(feature_names, feature_values))
    uid = parameters.get('user_id')
    ...
    return user_history

def product_history(feature_values, feature_names):
    parameters = dict(zip(feature_names, feature_values))
    pid = parameters.get('product_id')
    ...
    return product_history
    
def get_user_product_score(feature_values, feature_names):
    parameters = dict(zip(feature_names, feature_values))
    uid, pid = parameters.get('user_id'), parameters.get('product_id') 
    ...
    return user_product_score

data = pd.DaraFrame(columns=['sample_id', 'user_id', 'product_id', 'label'])

# First input
input_user = ('matrix', ['user_id'], load_user_history)
# Second input
input_product = ('matrix', ['product_id'], load_product_history)
# Output
output = ('matrix', ['user_id', 'product_id'], get_user_product_score)

train_generator = MIMODataGenerator(data_table = data_train
                                    model_inputs=[input_user, input_product],
                                    model_outputs=[output],
                                    shuffle=True,
                                    batch_size=batch_size
                                    )
```

example for multi-dimentional image processing model:


```python
from mimo_keras import MIMODataGenerator

def load_mri_scan(feature_values, feature_names):
    parameters = dict(zip(feature_names, feature_values))
    return normalize_image(cv.imread(parameters.get('image_path')))

def load_pt_scan(feature_values, feature_names):
    parameters = dict(zip(feature_names, feature_values))
    mri_path = parameters.get('image_path')
    return normalize_image(cv.imread(re.replace(mri_path, 'mri', 'pt')))
        
def load_mask(feature_values, feature_names):
    parameters = dict(zip(feature_names, feature_values))
    mri_path = parameters.get('image_path')
    return binarize_image(cv.imread(re.replace(mri_path, 'mri', 'mask')))


data = pd.DaraFrame(columns=['sample_id', 'user_id', 'product_id', 'disease_type'])

# First input
input_mri = ('matrix', ['image_path'], load_mri_scan)
# Second input
input_pt = ('matrix', ['image_path'], load_pt_scan)
# First Output
output_mask = ('matrix', ['image_path'], load_mask)
# Second Output
output_disease = ('matrix', ['disease_type'], 'raw')

train_generator = MIMODataGenerator(data_table = data_train
                                    model_inputs=[input_mri, input_pt],
                                    model_outputs=[output_mask, output_disease],
                                    shuffle=True,
                                    batch_size=batch_size
                                    )
```

#large_dataset #massive_dataset #MRI_keras #data_generator_for_medical_images #fMRI_keras #graph_neural_networrks #deep_learning_with_limited_GPU_memory #TensorFlow
