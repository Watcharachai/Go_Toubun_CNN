# Config CNN Model
class CNN_config:
    def __init__(self, img_width, img_height, batch_size, epochs):
        self.img_width = img_width
        self.img_height = img_height
        self.batch_size = batch_size
        self.epochs = epochs

    def get_values(self):
        return (self.batch_size, self.epochs)

    def input_shape(self):
        from keras import backend as K
        if K.image_data_format() == 'channels_first':
            input_shape = (4, self.img_width, self.img_height)
            return input_shape
        else:
            input_shape = (self.img_width, self.img_height, 4)
            return input_shape

    def Convolution(self,num_layer:int, output, size_kernal: list, functions: list, last_fn: list,optimizers):
        from keras.models import Sequential
        from keras.layers import Conv2D, MaxPooling2D
        from keras.layers import Activation, Dropout, Flatten, Dense
        import Classification as C
        try:
            model = Sequential()
            for i in range(0, num_layer):
                model.add(Conv2D(size_kernal[i], (3, 3), input_shape = self.input_shape()))
                model.add(Activation(functions[i]))
                model.add(MaxPooling2D(pool_size=(2, 2)))
            # Output Layer
            model.add(Flatten())  # flatten layer
            model.add(Dense(64))
            model.add(Activation(last_fn[0]))
            model.add(Dropout(0.5))
            model.add(Dense(output))  # Last Dence to 5 points
            model.add(Activation(last_fn[1]))
            model.compile(loss='categorical_crossentropy',
                        optimizer=optimizers,
                        metrics = ['accuracy', C.f1_score, C.precision_m, C.recall_m, 'AUC'])
            return model
        except Exception as e:
            print(e)

