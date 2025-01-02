import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from tensorflow.keras.optimizers import RMSprop

data_dir = 'dataset'

datagen = ImageDataGenerator(
	rescale = 1./255,
	validation_split = 0.2
)

train_generator = datagen.flow_from_directory(
	data_dir,
	target_size = (255, 255),
	batch_size = 20,
	class_mode = 'categorical',
	subset = 'training'
)

validation_generator = datagen.flow_from_directory(
	data_dir,
	target_size = (255, 255),
	batch_size = 20,
	class_mode = 'categorical',
	subset = 'validation'
)

def model():
	model = tf.keras.Sequential([
		tf.keras.layers.Conv2D(32,(3,3), activation = tf.keras.layers.LeakyReLU(), input_shape = (255, 255, 3)),
		tf.keras.layers.MaxPooling2D((2,2)),
		
		tf.keras.layers.Conv2D(64, (3,3), activation = tf.keras.layers.LeakyReLU()),
		tf.keras.layers.MaxPooling2D((2,2)),
		
		tf.keras.layers.Conv2D(128, (3, 3), activation = 'relu'),
		tf.keras.layers.MaxPooling2D((2,2)),
		
		tf.keras.layers.Conv2D(256, (3, 3), activation = 'relu'),
		tf.keras.layers.Conv2D(256, (3, 3), activation = 'relu'),
		tf.keras.layers.MaxPooling2D((2,2)),
		
		tf.keras.layers.Flatten(),
		
		tf.keras.layers.Dense(512, activation = 'relu'),
		tf.keras.layers.Dense(len(train_generator.class_indices), activation = 'softmax')
		
		])
		
	model.compile(optimizer = "RMSprop", loss = 'categorical_crossentropy', metrics = ['accuracy'])
	
	model.summary()
	
	history = model.fit(
		train_generator,
		steps_per_epoch = train_generator.samples // train_generator.batch_size,
		epochs = 10,
		validation_data = validation_generator,
		validation_steps = validation_generator.samples // validation_generator.batch_size
	)
	
	validation_loss, validation_accuracy = model.evaluate(validation_generator)
	print(f"Validation Loss: {validation_loss}")
	print(f"Validation Accuracy: {validation_accuracy}")
	
	model.save('PokemonGuesser.keras')
	print('model saved.')
	return model
		
trained_model = model()		
converter = tf.lite.TFLiteConverter.from_saved_model('PokemonGuesser.keras')
tflite_model = converter.convert()

# Save the converted model to a .tflite file
with open('model.tflite', 'wb') as f:
	f.write(tflite_model)
	
		
