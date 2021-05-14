import tensorflow as tf
from keras.models import load_model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.models import Model
from keras.optimizers import Adam
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Dropout, Flatten
from pathlib import Path
import numpy as np

train_generator = ImageDataGenerator(rotation_range=90,
                                     brightness_range=[0.1, 0.7],
                                     width_shift_range=0.5,
                                     height_shift_range=0.5,
                                     horizontal_flip=True,
                                     vertical_flip=True,
                                     validation_split=0.15,
                                     preprocessing_function=preprocess_input) # VGG16 preprocessing

test_generator = ImageDataGenerator(preprocessing_function=preprocess_input) # VGG16 preprocessing
#Param Config
from CNN_configs import CNN_config
img_width, img_height = 150,150
batch_size, epochs = 10, 50
train_data_dir = 'Dataset/training_set'
validation_data_dir = 'Dataset/testing_set'
nb_train_samples, nb_validation_samples = 105,35
n_classes=5
mod01 = CNN_config(img_width, img_height, batch_size, epochs)
input_shape = mod01.input_shape()

train_generator = train_generator.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height), class_mode= 'categorical',
    batch_size=batch_size, color_mode= "rgb", shuffle=True)

validation_generator = test_generator.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height), class_mode= 'categorical',
    batch_size=batch_size,color_mode= 'rgb',shuffle=True)

#%%Transfer Learning
def create_model(input_shape, n_classes, optimizer='rmsprop', fine_tune=0):
    """
    Compiles a model integrated with VGG16 pretrained layers

    input_shape: tuple - the shape of input images (width, height, channels)
    n_classes: int - number of classes for the output layer
    optimizer: string - instantiated optimizer to use for training. Defaults to 'RMSProp'
    fine_tune: int - The number of pre-trained layers to unfreeze.
                If set to 0, all pretrained layers will freeze during training
    """

    # Pretrained convolutional layers are loaded using the Imagenet weights.
    # Include_top is set to False, in order to exclude the model's fully-connected layers.
    conv_base = VGG16(include_top=False,
                      weights='imagenet',
                      input_shape=input_shape)
    # Defines how many layers to freeze during training.
    # Layers in the convolutional base are switched from trainable to non-trainable
    # depending on the size of the fine-tuning parameter.
    if fine_tune > 0:
        for layer in conv_base.layers[:-fine_tune]:
            layer.trainable = False
    else:
        for layer in conv_base.layers:
            layer.trainable = False

    # Create a new 'top' of the model (i.e. fully-connected layers).
    # This is 'bootstrapping' a new top_model onto the pretrained layers.
    top_model = conv_base.output
    top_model = Flatten(name="flatten")(top_model)
    top_model = Dense(4096, activation='relu')(top_model)
    top_model = Dense(1072, activation='relu')(top_model)
    top_model = Dropout(0.2)(top_model)
    output_layer = Dense(n_classes, activation='softmax')(top_model)
    # Group the convolutional base and new fully-connected layers into a Model object.
    model = Model(inputs=conv_base.input, outputs=output_layer)

    # Compiles the model for training.
    import Classification as C
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy', C.f1_score, C.precision_m, C.recall_m, 'AUC'])

    return model

#%%Run Model without Fine ture
optim_1 = Adam(learning_rate=0.001)
n_steps = train_generator.samples // batch_size
n_val_steps = validation_generator.samples // batch_size

# First we'll train the model without Fine-tuning
model1 = create_model(input_shape, n_classes, optim_1, fine_tune=0)

from livelossplot.inputs.keras import PlotLossesCallback
plot_loss_1 = PlotLossesCallback()
# ModelCheckpoint callback - save best weights
tl_checkpoint_1 = ModelCheckpoint(filepath='tl_model_v1.weights.best.hdf5',
                                  save_best_only=True,
                                  verbose=1)
# EarlyStopping
early_stop = EarlyStopping(monitor='val_loss',
                           patience=10,
                           restore_best_weights=True,
                           mode='min')
#Fitting Model
model1_history = model1.fit(train_generator,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_data=validation_generator,
                            steps_per_epoch=n_steps,
                            validation_steps=n_val_steps,
                            callbacks=[tl_checkpoint_1, early_stop, plot_loss_1],
                            verbose=1)
#%%Predict
# Generate predictions
model1.load_weights('tl_model_v1.weights.best.hdf5') # initialize the best trained weights
y_true = validation_generator.classes.astype('int32')
class_indices = train_generator.class_indices
class_indices = dict((v,k) for k,v in class_indices.items())

y_pred1 = model1.predict(validation_generator)
y_pred1_classes = np.argmax(y_pred1, axis=1)

#%%Pre-train to fine tune
train_generator.reset()
validation_generator.reset()

#Set Learning Rate
optim_2 = Adam(lr=0.0001)
model2_ft = create_model(input_shape, n_classes, optim_2, fine_tune=2) #Leaving Last 2 layers
plot_loss_2 = PlotLossesCallback()

# Retrain model with fine-tuning
model2_ft_history = model2_ft.fit(train_generator,
                                  batch_size=batch_size,
                                  epochs=epochs,
                                  validation_data=validation_generator,
                                  steps_per_epoch=n_steps,
                                  validation_steps=n_val_steps,
                                  callbacks=[tl_checkpoint_1, early_stop, plot_loss_2],
                                  verbose=1)

#%%Prediction2
model2_ft.load_weights('tl_model_v1.weights.best.hdf5') # initialize the best trained weights

y_pred2_ft = model2_ft.predict(validation_generator)
y_pred2_classes_ft = np.argmax(y_pred2_ft, axis=1)

#%%Plot Eval
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

label_predict = ['Ichika', 'Itsuki', 'Miku', 'Nino', 'Yotsuba']
#Comparing Models
from sklearn.metrics import accuracy_score
model1_acc = accuracy_score(y_true, y_pred1_classes)
(eval_loss,eval_accuracy, f1_score, precision, recall, auc) = model1.evaluate(validation_generator,batch_size = batch_size,verbose=32)
model2_acc_ft = accuracy_score(y_true, y_pred2_classes_ft)
(eval_loss2,eval_accuracy2, f1_score2, precision2, recall2, auc2) = model2_ft.evaluate(validation_generator,batch_size = batch_size,verbose=32)

acc1, acc2 = model1_history.history['accuracy'], model2_ft_history.history['accuracy']
val_acc1, val_acc2 = model1_history.history['val_accuracy'], model2_ft_history.history['val_accuracy']
loss1, loss2 = model1_history.history['loss'], model2_ft_history.history['loss']
val_loss1, val_loss2 = model1_history.history['val_loss'], model2_ft_history.history['val_loss']
auc_comulate1, auc_comulate2 = model1_history.history['auc'], model2_ft_history.history['auc']

print("VGG16 Model Accuracy with Fine-Tuning: {:.2f}%".format(model2_acc_ft * 100))
print(f"Accuracy: {eval_accuracy*100:.2f}%")
print(f"Loss: {eval_loss:.2f}")
print(f"F1_score: {f1_score*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall: {recall*100:.2f}%")
print("-------------------------")
print("VGG16 Model Accuracy without Fine-Tuning: {:.2f}%".format(model1_acc * 100))
print(f"Accuracy: {eval_accuracy2*100:.2f}%")
print(f"Loss: {eval_loss2:.2f}")
print(f"F1_score: {f1_score2*100:.2f}%")
print(f"Precision: {precision2*100:.2f}%")
print(f"Recall: {recall2*100:.2f}%")

#Classfification Report
print("Classification Report of Unfined tuning/")
print(classification_report(y_true, y_pred1_classes, target_names = label_predict))
print("-------------------------")
print("Classification Report of Fine tuning/")
print(classification_report(y_true, y_pred2_classes_ft, target_names = label_predict))

''''#ACC and Loss
epoch_range = 22#range(epochs)
fig, (ax1,ax2) = plt.subplots(1,2) #Plot1
fig.suptitle('Evaluation Classification of Gotoubun', fontsize=16)
ax1.plot(epoch_range, acc1, label='Training Accuracy')
ax1.plot(epoch_range, val_acc1, label='Validation Accuracy')
ax1.legend(loc = 'lower right')
ax1.title.set_text('Accuracy Evaluation')
plt.title('Training and Validation Accuracy')
ax2.plot(epoch_range, loss1, label = 'Training Loss')
ax2.plot(epoch_range, val_loss1, label = 'Validation Loss')
ax2.legend(loc = 'lower right')
ax2.title.set_text('Loss Evaluation')
plt.show()

fig, (ax1,ax2) = plt.subplots(1,2) #Plot2
fig.suptitle('Evaluation Classification of Gotoubun', fontsize=16)
ax1.plot(epoch_range, acc1, label='Training Accuracy')
ax1.plot(epoch_range, val_acc1, label='Validation Accuracy')
ax1.legend(loc = 'lower right')
ax1.title.set_text('Accuracy Evaluation')
plt.title('Training and Validation Accuracy')
ax2.plot(epoch_range, loss1, label = 'Training Loss')
ax2.plot(epoch_range, val_loss1, label = 'Validation Loss')
ax2.legend(loc = 'lower right')
ax2.title.set_text('Loss Evaluation')
plt.show()
'''
#Confusion Matrix
import seaborn as sns
import matplotlib.pyplot as plt
confusion_metrics1 = confusion_matrix(y_true, y_pred1_classes)
confusion_metrics2 = confusion_matrix(y_true, y_pred2_classes_ft)

plt.figure(figsize=(10,8)) #Plot1
sns.heatmap(confusion_metrics1, cmap = 'Blues', fmt = '.2%').set_title('Confusion Matrix')
plt.title("Model1 Confusion Matrix")
plt.ylabel('Ichika, Itsuki, Miku, Nino, Yotsuba (Predicted)')
plt.xlabel('Ichika, Itsuki, Miku, Nino, Yotsuba (Actual)')
plt.show()

plt.figure(figsize=(10,8)) #Plot2
sns.heatmap(confusion_metrics2, cmap = 'Blues', fmt = '.2%').set_title('Confusion Matrix')
plt.title("Model2 Confusion Matrix")
plt.ylabel('Ichika, Itsuki, Miku, Nino, Yotsuba (Predicted)')
plt.xlabel('Ichika, Itsuki, Miku, Nino, Yotsuba (Actual)')
plt.show()
