
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#Data-Preprocessing
'''number of total dataset = 140, training = 105, testing = 35'''
#Set data Format
from CNN_configs import CNN_config
img_width, img_height = 150,150
mod01 = CNN_config(img_width, img_height, 1, 50)
input_shape = mod01.input_shape()
batch_size, epochs = mod01.get_values()
#Convolution setting
from keras.preprocessing.image import ImageDataGenerator
# this is the augmentation configuration we will use for training

#Setting on Data Convolution
train_data_dir = 'Dataset/training_set'
validation_data_dir = 'Dataset/testing_set'

train_datagen = ImageDataGenerator(
    rotation_range= 40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1. / 255)# only rescaling

#Decode Dataset
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size, color_mode= "rgba")

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,color_mode= 'rgba')

#%%
#Fitting Model
size_kernal = [16,32,64]
function_ac = ['relu', 'relu', 'relu']
last_fn = ['relu', 'sigmoid']
model = mod01.Convolution(num_layer=3, output =5, size_kernal=size_kernal, functions=function_ac,
                          last_fn= last_fn,optimizers = 'sgd')
#Some param from dataset
nb_train_samples = train_generator.labels.shape[0]
nb_validation_samples = validation_generator.labels.shape[0]
#Calling Methods for plotting Classification
from Classification import PerformanceVisualizationCallback

performace_cbk = PerformanceVisualizationCallback(model=model, validation_data= validation_generator,
                                                  image_dir= ' performance vizualization')

log_model1 = model.fit(
    train_generator,
    steps_per_epoch= nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)
    #callbacks = [performace_cbk])


#test_accu = model.evaluate(validation_generator,steps=1)
#try to predict
Y_pred = model.predict(validation_generator,batch_size=batch_size)
y_pred = np.argmax(Y_pred, axis=1)

#Accuracy metrice and Plot
import matplotlib.pyplot as plt
label_predict = ['Ichika', 'Itsuki', 'Miku', 'Nino', 'Yotsuba']
(eval_loss,eval_accuracy, f1_score, precision, recall, auc) = model.evaluate(validation_generator,batch_size = batch_size,verbose=32)
print(f"Loss: {eval_loss:.2f}")
print(f"Accuracy: {eval_accuracy*100:.2f}%")
print(f"F1_score: {f1_score*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall: {recall*100:.2f}%")

acc = log_model1.history['accuracy']
val_acc = log_model1.history['val_accuracy']
loss = log_model1.history['loss']
val_loss = log_model1.history['val_loss']
auc_comulate = log_model1.history['auc']

#%%Plot
#Plot Classification Metrics
from sklearn.metrics import classification_report, confusion_matrix
y_true = validation_generator.classes.astype('int32')
y_pred = y_pred.astype('int32')
print(classification_report(y_true, y_pred, target_names = label_predict))
#ACC and Loss
epoch_range = range(epochs)
fig, (ax1,ax2) = plt.subplots(1,2)
fig.suptitle('Evaluation Classification of Gotoubun', fontsize=16)
ax1.plot(epoch_range, acc, label='Training Accuracy')
ax1.plot(epoch_range, val_acc, label='Validation Accuracy')
ax1.legend(loc = 'lower right')
ax1.title.set_text('Accuracy Evaluation')
plt.title('Training and Validation Accuracy')
ax2.plot(epoch_range, loss, label = 'Training Loss')
ax2.plot(epoch_range, val_loss, label = 'Validation Loss')
ax2.legend(loc = 'lower right')
ax2.title.set_text('Loss Evaluation')
plt.show()

#Confusion Matrix
import seaborn as sns
confusion_metrics = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(confusion_metrics, cmap = 'Blues', fmt = '.2%').set_title('Confusion Matrix')
plt.ylabel('Ichika, Itsuki, Miku, Nino, Yotsuba (Predicted)')
plt.xlabel('Ichika, Itsuki, Miku, Nino, Yotsuba (Actual)')
plt.show()

#Plot Simple ROC
#x_line = np.arange(epoch_range)
plt.figure(1)
plt.plot(epoch_range, auc_comulate, label =f'ROC curve (area = {auc:0.2f})' ,color='deeppink', linestyle = ':', linewidth=4)
#plt.plot([0, 0], [epoch_range, ], 'k--', lw=2)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc = 'lower right')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.show()

#model.save("CNN10b_e100.h5")

