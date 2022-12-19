#  Custom Data-Generator for multiple-input multiple-output models in TF-keras
Develop your own keras Data-Generator in TF-Keras to load and batch every data type with any format from a massive dataset in computers with limited main or GPU memory... mimo-keras is a package that enables feeding models with any format and any number of inputs and outputs.

MIMO-keras — Never use keras imagedatagenerator to load data in batch
Do you ever use image_data_generator() or load_form_directory() to load and feed your deep model in keras? Of course, you do. mimo-keras makes data loader very simple even for multiple input/output models or data with formats that are not supported by default in keras.

#### mimo-keras is like image_data_generator().load_from_directory(), but better:

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
matrix_input = ('matrix', # name for model IO
                ['matrix_path'], # this column_names and their values are sent to the your function for each sample in batch generation
                load_and_preprocess_matrix # function to load processing the data, replace with string 'raw' to send values directly to the model IO
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

model.train(generator = train_generator,
            validation_data = validation_generator,
            epochs=EPOCHS
            )
```



#large_dataset #massive_dataset #MRI_keras #data_generator_for_medical_images #fMRI_keras #graph_neural_networrks #deep_learning_with_limited_GPU_memory #TensorFlow
