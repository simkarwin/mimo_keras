[![Latest PyPI version](https://img.shields.io/pypi/v/mimo-keras.svg)](https://pypi.python.org/pypi/mimo-keras)
[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE.md)

#  Custom Data-Generator for multiple-input multiple-output models in TF-Keras
Develop your own Keras DataGenerator in TF-Keras to load and batch every data type with any format from a massive dataset in computers with limited main or GPU memory... mimo-keras is a package that enables feeding models with any format and any number of inputs and outputs.

mimo-keras — One Keras DataGenerator for all model Inputs and Ouputs 
----------------------------------------------------------------------
Have you ever used ImageDataGenerator(), load_form_directory(), or load_from_dataframe() to load batch data and feed your deep model in Keras? mimo-keras makes the data loader quite simple and straightforward even for multiple input/output models



## mimo-keras is a more general version of ImageDataGenerator().load_from_dataframe() and ImageDataGenerator().load_from_directory():



1. Supports all data formats that are not supported by default in Keras.
2. Unlike Keras DataGenerator, you define only one data generator for all inputs and outputs.
3. You can write your own data loader function due to data format.
4. You can use your custom preprocessing pipeline per IO indepently without any limitation. For this pupose, fit your callable class and send it as data loader or preprocessor to generator.
5. You can load data files (images, .numpy, .dicom, .nifti, ...) directly, without needing to copy them to the folders with predefined structures before training.

Installation
------------
```bash
pip install mimo-keras
```

Usage
-----
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
output = ('target', ['label'], 'raw') # binarry classification
train_generator = MIMODataGenerator(data_table=data_train
                                    model_inputs=[matrix_input, metadata_input],
                                    model_outputs=[output],
                                    shuffle=True,
                                    batch_size=BATCH_SIZE
                                    )

validation_generaetor = MIMODataGenerator(data_table=data_validation
                                          model_inputs=[matrix_input, metadata_input],
                                          model_outputs=[output],
                                          shuffle=False,
                                          batch_size=BATCH_SIZE
                                          )

model.fit(generator = train_generator,
          validation_data = validation_generator,
          epochs=EPOCHS
          )
```

for more complicated model you can use only sample_id to generate data for each IO. for example to train a recommender system (DLRM) using a massive dataset:


```python
from mimo_keras import MIMODataGenerator

def load_user_history(feature_values, feature_names):
    parameters = dict(zip(feature_names, feature_values))
    uid = parameters.get('user_id')
    .
    .
    .
    return user_history

def load_product_history(feature_values, feature_names):
    parameters = dict(zip(feature_names, feature_values))
    pid = parameters.get('product_id')
    .
    .
    .
    return product_history
    
def get_user_product_score(feature_values, feature_names):
    parameters = dict(zip(feature_names, feature_values))
    uid, pid = parameters.get('user_id'), parameters.get('product_id') 
    .
    .
    .
    return user_product_score

data = pd.DaraFrame(columns=['sample_id', 'user_id', 'product_id', 'label'])

# First input
input_user = ('user_data', ['user_id'], load_user_history)
# Second input
input_product = ('product_data', ['product_id'], load_product_history)
# Output
output = ('score', ['user_id', 'product_id'], get_user_product_score)

train_generator = MIMODataGenerator(data_table=data_train
                                    model_inputs=[input_user, input_product],
                                    model_outputs=[output],
                                    shuffle=True,
                                    batch_size=BATCH_SIZE
                                    )
```

example for loading .nifti file to train multi-dimentional medical image processing model:

```python
from mimo_keras import MIMODataGenerator
import nibabel as nib

def load_mri_scan(feature_values, feature_names):
    parameters = dict(zip(feature_names, feature_values))
    return normalize_image(nib.load(parameters.get('image_path')).get_fdata())

def load_pet_scan(feature_values, feature_names):
    parameters = dict(zip(feature_names, feature_values))
    mri_path = parameters.get('image_path')
    return normalize_image(nib.load(mri_path.replace('_mri_', '_pet_scan_')).get_fdata())
        
def load_mask(feature_values, feature_names):
    parameters = dict(zip(feature_names, feature_values))
    mri_path = parameters.get('image_path')
    return binarize_image(nib.load(mri_path.replace('_mri_', '_mask_')).get_fdata())


data = pd.DaraFrame(columns=['sample_id', 'image_path', 'disease_type'])

# First input
input_mri = ('mri_scan', ['image_path'], load_mri_scan)
# Second input
input_pet = ('pet_scan', ['image_path'], load_pet_scan)
# First Output
output_mask = ('mask', ['image_path'], load_mask)
# Second Output
output_disease = ('disease_type', ['disease_type'], 'raw')

train_generator = MIMODataGenerator(data_table=data_train
                                    model_inputs=[input_mri, input_pet],
                                    model_outputs=[output_mask, output_disease],
                                    shuffle=True,
                                    batch_size=BATCH_SIZE
                                    )
```

Calculating the metric to evaluate the model:

```python
from mimo_keras import MIMODataGenerator

.
.
.

input = ('input_data', ['s_room', 'n_bedroom', 's_total', 'city', 'floor', 'location'], 'raw')
output = ('target_price', ['price'], 'raw')

test_generaetor = MIMODataGenerator(data_table=data_test
                                    model_inputs=[input],
                                    model_outputs=[output],
                                    shuffle=False,
                                    batch_size=BATCH_SIZE
                                    )
y_pred = model.predict(test_generator)
y_target = test_generator.get_io_data_values_by_name('target_price', 'all')
# or y_target = test_generator.data_table.price.to_list()

mae = mean_absolute_error(y_target, y_pred)
```
If it was useful, please star it :D

#large_dataset #massive_dataset #MRI_keras #data_generator_for_medical_images #fMRI_keras #graph_neural_networks #deep_learning_with_limited_GPU_memory #TensorFlow
