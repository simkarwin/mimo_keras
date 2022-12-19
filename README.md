# custome Data-Generator in TF-Keras 
Develop your own keras Data-Generator in TF-Keras to load and batch every data type with any format from a massive dataset in computers with limited main or GPU memory...

I have implemented it for a 3D Deep neural network to process medical MRI images and another project based on graph neural networks. But it is also applicable when you have a multi-input multi-output network with unusual input-data type, or multiple data sources.
 In summary, "you can do what you want on data before sending it to model, even cooking dinner."

 Let's assume that you have a graph network belonging to a social media. Features of each node are in a data-frame, and the edges are accessible in another data-frame... merging them makes many rows by repeated personal data and a high volume Data-frame!!!! But using this technique, you generate the data only for batchs when they are needed...  it is solved. Don't worry about limited memory, un-usual data type, MIMO model data handling and massive dataset in Keras...

Shervin explained a method for sequential models that you can find it in the following link:
https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.
But it is different about functional models, and no sources have been found to describe it clearly. for usual data like images see image_data_generator.load_from_directory or flow_from_dataframe documentations in Keras. 

mimo-keras is a package that enabled feeding model with any number and any type of inputs and outpuuts.

#large_dataset #massive_dataset #MRI_keras #data_generator_for_medical_images #fMRI_keras #graph_neural_networrks #deep_learning_with_limited_GPU_memory #TensorFlow
