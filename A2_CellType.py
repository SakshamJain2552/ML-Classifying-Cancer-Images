#!/usr/bin/env python
# coding: utf-8

# # Cell-type Classification

# In[1]:


import zipfile
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_recall_fscore_support, classification_report
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from keras_tuner import RandomSearch
from PIL import Image


# Extracting Images from the zip

# In[2]:


with zipfile.ZipFile('./Image_classification_data.zip', 'r') as zip_ref:
    zip_ref.extractall('./')


# Reading the dataframe

# In[3]:


data = pd.read_csv('./data_labels_mainData.csv')


# In[4]:


data.head()


# Check for any null values
# 
# Check if to see any null values exists

# In[5]:


data.cellTypeName.value_counts()


# In[6]:


data.isnull().sum()


# In[7]:


data.shape
data.info()
data.describe()


# In[8]:


plt.figure(figsize=(20,20))
plt.subplot(3, 2, 1)
plt.hist(data['patientID'], facecolor='blue', alpha=0.5)
plt.title('patient ID')


# In[9]:


pd.crosstab(data['cellTypeName'], data['isCancerous'], margins=False).plot.bar(stacked=True,
                                                                                         title='Frequency of cell types',
                                                                                         xlabel='Cell Type',
                                                                                         ylabel='Number of Images')


# All the cancer cells in the main dataset are of a specific kind called Epithelial. This could be a challenge because there are no cell types in the dataset that have a mix of healthy and cancerous cells. This lack of variety could result in information that might be misleading.

# In[10]:


df1 = data.groupby(['cellTypeName', 'cellType'])['InstanceID'].nunique()
df2 = data.groupby(['isCancerous'])['InstanceID'].nunique()
print(df1)
print(df2)


# In[11]:


label_names = {k: v for k, v in df1.keys()}
cellTypes = {v: k for k, v in label_names.items()}
print(label_names)
print(cellTypes)


# In[12]:


img_list = list(data["ImageName"])
types = list(data["cellType"])
target_list = list(data["isCancerous"])
target_labels = {1: "Cancerous", 0: "Non-Cancerous"}

plt.figure(figsize=[25, 12])

for i in np.arange(40):
    plt.subplot(4, 10, i + 1)
    img = Image.open('./patch_images/' + img_list[i])

    plt.imshow(img)
    plt.title(cellTypes[types[i]] + "\n" + target_labels[target_list[i]])


# In[13]:


# Check image sizes
img = Image.open('./patch_images/' + img_list[100])
img_numpy = np.ascontiguousarray(img, dtype=np.float32)
print("Image size: \nH:{} W:{} C:{}".format(img_numpy.shape[0], img_numpy.shape[1], img_numpy.shape[2]))


# ## Train test validation split

# In[14]:


train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.25, random_state=42)

print("Train data : {}, Val Data: {}, Test Data: {}".format(train_data.shape[0], val_data.shape[0], test_data.shape[0]))


# In[15]:


train_data['cellType'] = train_data['cellType'].astype('str')
val_data['cellType'] = val_data['cellType'].astype('str')
test_data['cellType'] = test_data['cellType'].astype('str')


# In[16]:


train_datagen = ImageDataGenerator(rescale=1./255, data_format='channels_last')
val_datagen = ImageDataGenerator(rescale=1./255, data_format='channels_last')
test_datagen = ImageDataGenerator(rescale=1./255, data_format='channels_last')

batch_size = 32

train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_data,
        directory='./patch_images/',
        x_col="ImageName",
        y_col="cellType",
        target_size=(27, 27),
        batch_size=batch_size,
        class_mode='categorical')

validation_generator = val_datagen.flow_from_dataframe(
        dataframe=val_data,
        directory='./patch_images/',
        x_col="ImageName",
        y_col="cellType",
        target_size=(27, 27),
        batch_size=batch_size,
        class_mode='categorical')

test_generator = test_datagen.flow_from_dataframe(
    dataframe = test_data,
    directory = './patch_images/',
    x_col = 'ImageName',
    y_col = "cellType",
    target_size = (27, 27),
    batch_size = batch_size,
    class_mode = 'categorical',
)


# In[17]:


INPUT_DIM = (27,27,3)
HIDDEN_LAYER_DIM = 256
OUTPUT_CLASSES = 4


# In[18]:


AUTOTUNE = tf.data.experimental.AUTOTUNE

tf.__version__


# In[19]:


def plot_learning_curve(train_loss, val_loss, train_metric, val_metric, metric_name='Accuracy'):
    plt.figure(figsize=(10,5))
    
    plt.subplot(1,2,1)
    plt.plot(train_loss, 'r--')
    plt.plot(val_loss, 'b--')
    plt.xlabel("epochs")
    plt.ylabel("Loss")
    plt.legend(['train', 'val'], loc='upper left')

    plt.subplot(1,2,2)
    plt.plot(train_metric, 'r--')
    plt.plot(val_metric, 'b--')
    plt.xlabel("epochs")
    plt.ylabel(metric_name)
    plt.legend(['train', 'val'], loc='upper left')

    plt.show()


# In[20]:


def plotConfusionMatrix(y_test,y_pred):

    conf_data = confusion_matrix(y_test,y_pred)
    df_cm = pd.DataFrame(conf_data, columns=np.unique(y_test), index = np.unique(y_test))
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    plt.figure(figsize = (10,7))
    sns.set(font_scale=1.4)#for label size

    return sns.heatmap(df_cm, cmap="Blues",annot=True,square=True,annot_kws={"size": 16},fmt="g");# font size

early_stopping_monitor = EarlyStopping(patience=10, monitor='val_categorical_accuracy')
def fitModel(model, train_gen=train_generator, val_gen=validation_generator):
    fit_history = model.fit(train_generator, validation_data = validation_generator, 
                            callbacks=[early_stopping_monitor],
                            epochs=10000, verbose=1)
    return fit_history

def predictModel(model, generator):
    batch_size_ = 1
    y_pred = list()
    y_test = list()
    filenames = generator.filenames
    N_images = len(filenames)
    batches = 0

    # iterate through the data generator and predict for each batch
    # hold the predictions and labels
    for x,y in generator:
            yp = model.predict(x, verbose=0)
            yp = np.argmax(yp, axis = 1)
            yt = np.argmax(y, axis = 1)
            y_pred = y_pred + yp.tolist()
            y_test = y_test + yt.tolist()

            batches += 1
            if batches >= N_images / batch_size_:
                break
                
    return (y_test, y_pred)


def showMetrics(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    cm = classification_report(y_test, y_pred, zero_division=0)

    print("Accuracy score: ", accuracy_score(y_test, y_pred))
    print("F1 score: ", f1_score(y_test, y_pred, average='macro'))
    print(cm)
    
    


# ## MLP
#  

# - Initial baseline NN model
# - 4 output classes
# - Input is 27x27 RGB(3 channels) images
# - 1 hidden layer with 256 internal nodes
# - Loss - Categorical Cross Entropy
# - Metric - categorical_accuracy
# - SGD Optimizer

# ### Base MLP

# In[21]:


model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=INPUT_DIM),
    tf.keras.layers.Dense(HIDDEN_LAYER_DIM, activation='relu'),
    tf.keras.layers.Dense(OUTPUT_CLASSES)
])


# In[22]:


model.summary()


# In[23]:


tf.keras.utils.plot_model(model, show_shapes=True)


# In[24]:


model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['categorical_accuracy'])


# In[25]:


history = fitModel(model)


# In[26]:


plot_learning_curve(history.history['loss'], history.history['val_loss'], 
                    history.history['categorical_accuracy'], history.history['val_categorical_accuracy'], 
                    metric_name='Accuracy')


# In[27]:


(train_y, train_pred_y) = predictModel(model, train_generator)


# In[28]:


(val_y, val_pred_y) = predictModel(model, validation_generator)


# In[29]:


(test_y, test_pred_y) = predictModel(model, test_generator)


# In[30]:


showMetrics(train_y, train_pred_y)


# In[31]:


showMetrics(val_y, val_pred_y)


# In[32]:


showMetrics(test_y, test_pred_y)


# In[33]:


plotConfusionMatrix(test_y, test_pred_y)


# - Baseline performed quite well with very slight overfitting. This can be identified from the graph and slight gap between train and val accuracies 
# - The performance on class 3 is consistently lower compared to other classes, suggesting that the model struggles to accurately predict this class.
# - Class Imbalance: It appears that class 2 has the highest number of samples and generally performs better, while class 3 has the fewest samples and consistently has lower performance. Class imbalance might be a factor contributing to the lower performance on class 3.
# - There is room for improvement, particularly in predicting class 3. We may consider exploring techniques such as regularization, adjusting model hyperparameters, or trying more advanced models to improve performance.
# - F1-Score and Weighted Average: The weighted average F1-score is 0.61 on the training set, 0.58 on the validation set, and 0.52 on the test set. The weighted average considers both precision and recall and provides a single metric to assess the overall performance of the model across all classes. The decreasing trend in F1-score from training to test set suggests a potential issue of overfitting.

# ### Trying regularization

# - Trying regularization to generalize and improve accuracy

# In[34]:


reg_lambda = 0.001

reg_model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=INPUT_DIM),
    tf.keras.layers.Dense(HIDDEN_LAYER_DIM, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(reg_lambda)),
    tf.keras.layers.Dense(OUTPUT_CLASSES, kernel_regularizer=tf.keras.regularizers.l2(reg_lambda))
])

reg_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['categorical_accuracy'])


# In[35]:


reg_model.summary()


# In[36]:


reg_history = fitModel(reg_model)


# In[37]:


plot_learning_curve(reg_history.history['loss'], reg_history.history['val_loss'], 
                    reg_history.history['categorical_accuracy'], reg_history.history['val_categorical_accuracy'], 
                    metric_name='Accuracy')


# In[38]:


(train_y, train_reg_pred_y) = predictModel(reg_model, train_generator)


# In[39]:


(val_y, val_reg_pred_y) = predictModel(reg_model, validation_generator)


# In[40]:


(test_reg_y, test_reg_pred_y) = predictModel(reg_model, test_generator)


# In[41]:


showMetrics(train_y, train_reg_pred_y)


# In[42]:


showMetrics(val_y, val_reg_pred_y)


# In[43]:


showMetrics(test_reg_y, test_reg_pred_y)


# In[44]:


plotConfusionMatrix(test_reg_y, test_reg_pred_y)


# The model was regularized but the performance was bad. Hence we can try tuning the hyper parameters to get better performance.

# ### Hyper Parameter Tuning 

# Here we are tuning lambda value for L2 regularization, number of neurons in the hidden layer and the learning rate of the Optimizer

# In[45]:


# Define the model-building function
def build_mlp(hp):
    reg_lambda = hp.Choice('reg_lambda', values=[0.001, 0.01, 0.1])
    num_neurons = hp.Choice('num_neurons', values=[128, 256, 512])
    learning_rate = hp.Choice('learning_rate', values=[0.001, 0.01, 0.1])

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=INPUT_DIM),
        tf.keras.layers.Dense(num_neurons, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(reg_lambda)),
        tf.keras.layers.Dense(OUTPUT_CLASSES, kernel_regularizer=tf.keras.regularizers.l2(reg_lambda))
    ])

    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['categorical_accuracy'])
    return model


# In[48]:


tuner_mlp = RandomSearch(
    build_mlp,
    objective='val_categorical_accuracy',
    max_trials=10,  # Adjust this value as desired
    executions_per_trial=1,
    directory='tuner_directory',  # Provide a directory to store tuner results
    project_name='mlp_hyperparameter_tuning'  # Choose a project name
)

# Perform the hyperparameter search
tuner_mlp.search(train_generator, epochs=25, validation_data=validation_generator)

# Get the best hyperparameters and model architecture
best_mlp_hyperparameters = tuner_mlp.get_best_hyperparameters(num_trials=1)[0]
best_mlp_model = tuner_mlp.get_best_models(num_models=1)[0]


# In[49]:


print(best_mlp_hyperparameters)


# In[50]:


best_mlp_model.summary()


# In[51]:


history_best_mlp = fitModel(best_mlp_model)


# In[52]:


plot_learning_curve(history_best_mlp.history['loss'], history_best_mlp.history['val_loss'], 
                    history_best_mlp.history['categorical_accuracy'], history_best_mlp.history['val_categorical_accuracy'], 
                    metric_name='Accuracy')


# In[53]:


(train_y, train_best_mlp_pred_y) = predictModel(best_mlp_model, train_generator)


# In[54]:


(val_y, val_best_mlp_pred_y) = predictModel(best_mlp_model, validation_generator)


# In[55]:


(test_y, test_best_mlp_pred_y) = predictModel(best_mlp_model, test_generator)


# In[56]:


showMetrics(train_y, train_best_mlp_pred_y)


# In[57]:


showMetrics(val_y, val_best_mlp_pred_y)


# In[58]:


showMetrics(test_y, test_best_mlp_pred_y)


# In[59]:


plotConfusionMatrix(test_y, test_best_mlp_pred_y)


# Hyper paramter tuning worked as we can see the accuracy improved keeping the model regularized. However, it is worth exploring more complicated models to see if the accuracy can be improved

# ## VGG

# ### Base VGG

# Here we are implementing basic 3 block VGG

# In[21]:


model_VGG = tf.keras.Sequential([
    #VGG block 1
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=INPUT_DIM),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    #VGG block 2
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    #VGG block 3
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(OUTPUT_CLASSES, activation='softmax'),
])


# In[22]:


model_VGG.summary()


# In[23]:


tf.keras.utils.plot_model(model_VGG, show_shapes=True)


# In[24]:


model_VGG.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['categorical_accuracy'])


# In[25]:


history_VGG = fitModel(model_VGG)


# In[26]:


plot_learning_curve(history_VGG.history['loss'], history_VGG.history['val_loss'], 
                    history_VGG.history['categorical_accuracy'], history_VGG.history['val_categorical_accuracy'])


# In[27]:


(train_y, train_VGG_pred_y) = predictModel(model_VGG, train_generator)


# In[28]:


(val_y, val_VGG_pred_y) = predictModel(model_VGG, validation_generator)


# In[29]:


(test_y, test_VGG_pred_y) = predictModel(model_VGG, test_generator)


# In[30]:


showMetrics(train_y, train_VGG_pred_y)


# In[31]:


showMetrics(val_y, val_VGG_pred_y)


# In[32]:


showMetrics(test_y, test_VGG_pred_y)


# In[35]:


plotConfusionMatrix(test_y, test_VGG_pred_y)


# - model accuracy was good with lesser loss than MLP
# - shows some overfitting as seen from the graph and gap between accuracies for train and val data
# 
# Trying different techniques to overcome overfitting

# ### Trying Dropout 

# In[21]:


model_VGG_dropout = tf.keras.Sequential([
    #VGG block 1
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=INPUT_DIM),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    #VGG block 2
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    #VGG block 3
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(.2),
    tf.keras.layers.Dense(OUTPUT_CLASSES, activation='softmax'),
])


# In[22]:


model_VGG_dropout.summary()


# In[23]:


model_VGG_dropout.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['categorical_accuracy'])


# In[24]:


history_VGG_dropout = fitModel(model_VGG_dropout)


# In[25]:


plot_learning_curve(history_VGG_dropout.history['loss'], history_VGG_dropout.history['val_loss'], 
                    history_VGG_dropout.history['categorical_accuracy'], history_VGG_dropout.history['val_categorical_accuracy'])


# In[26]:


(train_y, train_VGG_dropout_pred_y) = predictModel(model_VGG_dropout, train_generator)


# In[27]:


(val_y, val_VGG_dropout_pred_y) = predictModel(model_VGG_dropout, validation_generator)


# In[28]:


(test_y, test_VGG_dropout_pred_y) = predictModel(model_VGG_dropout, test_generator)


# In[29]:


showMetrics(train_y, train_VGG_dropout_pred_y)


# In[30]:


showMetrics(val_y, val_VGG_dropout_pred_y)


# In[31]:


showMetrics(test_y, test_VGG_dropout_pred_y)


# In[32]:


plotConfusionMatrix(test_y, test_VGG_dropout_pred_y)


# Seem like Droupout didn't help much with generalizing the model

# ### Regularization + Dropout

# In[20]:


reg_lambda = 0.001

model_VGG_dropout_reg = tf.keras.Sequential([
    #VGG block 1
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(reg_lambda), input_shape=INPUT_DIM),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(reg_lambda)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    #VGG block 2
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(reg_lambda)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(reg_lambda)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    #VGG block 3
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(reg_lambda)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(reg_lambda)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(.2),
    tf.keras.layers.Dense(OUTPUT_CLASSES, activation='softmax'),
])


# In[21]:


model_VGG_dropout_reg.summary()


# In[22]:


model_VGG_dropout_reg.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['categorical_accuracy'])


# In[23]:


history_VGG_dropout_reg = fitModel(model_VGG_dropout_reg)


# In[24]:


plot_learning_curve(history_VGG_dropout_reg.history['loss'], history_VGG_dropout_reg.history['val_loss'], 
                    history_VGG_dropout_reg.history['categorical_accuracy'], history_VGG_dropout_reg.history['val_categorical_accuracy'])


# In[25]:


(train_y, train_VGG_dropout_reg_pred_y) = predictModel(model_VGG_dropout_reg, train_generator)


# In[26]:


(val_y, val_VGG_dropout_reg_pred_y) = predictModel(model_VGG_dropout_reg, validation_generator)


# In[27]:


(test_y, test_VGG_dropout_reg_pred_y) = predictModel(model_VGG_dropout_reg, test_generator)


# In[28]:


showMetrics(train_y, train_VGG_dropout_reg_pred_y)


# In[29]:


showMetrics(val_y, val_VGG_dropout_reg_pred_y)


# In[30]:


showMetrics(test_y, test_VGG_dropout_reg_pred_y)


# In[31]:


plotConfusionMatrix(test_y, test_VGG_dropout_reg_pred_y)


# - model accuracy reduced 
# - very minimal effect on overfitting

# ### Data Augmentation

# Data augmentation is a technique used in machine learning and deep learning that involves creating new training examples by applying various transformations to existing data. These transformations can include rotation, scaling, cropping, flipping, or adding noise to the data. Data augmentation helps in reducing overfitting by increasing the size and diversity of the training dataset, allowing the model to learn more robust and generalized representations. By introducing variations in the data, data augmentation helps the model capture different perspectives and variations present in the real-world data, thus improving its ability to generalize well to unseen examples.

# In[21]:


aug_train_datagen = ImageDataGenerator(rescale=1./255, data_format='channels_last',
                                   rotation_range=90, width_shift_range=0.2,
                                   height_shift_range=0.2, brightness_range=[0.5,1.5])
aug_val_datagen = ImageDataGenerator(rescale=1./255, data_format='channels_last')

batch_size = 32

aug_train_generator = aug_train_datagen.flow_from_dataframe(
    dataframe = train_data,
    directory = './patch_images/',
    x_col = 'ImageName',
    y_col = "cellType",
    target_size = (27, 27),
    batch_size = batch_size,
    class_mode='categorical')

aug_validation_generator = aug_val_datagen.flow_from_dataframe(
    dataframe=val_data,
    directory = './patch_images/',
    x_col = 'ImageName',
    y_col = "cellType",
    target_size = (27, 27),
    batch_size = batch_size,
    class_mode='categorical')


# In[22]:


reg_lambda = 0.001

model_VGG_dropout_reg_aug = tf.keras.Sequential([
    #VGG block 1
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(reg_lambda), input_shape=INPUT_DIM),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(reg_lambda)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    #VGG block 2
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(reg_lambda)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(reg_lambda)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    #VGG block 3
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(reg_lambda)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(reg_lambda)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(.2),
    tf.keras.layers.Dense(OUTPUT_CLASSES, activation='softmax'),
])


# In[23]:


model_VGG_dropout_reg_aug.summary()


# In[24]:


model_VGG_dropout_reg_aug.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['categorical_accuracy'])


# In[25]:


history_VGG_dropout_reg_aug = fitModel(model_VGG_dropout_reg_aug, train_gen = aug_train_generator, val_gen = aug_validation_generator )


# In[26]:


plot_learning_curve(history_VGG_dropout_reg_aug.history['loss'], history_VGG_dropout_reg_aug.history['val_loss'], 
                    history_VGG_dropout_reg_aug.history['categorical_accuracy'], history_VGG_dropout_reg_aug.history['val_categorical_accuracy'])


# In[27]:


(train_y, train_VGG_dropout_reg_aug_pred_y) = predictModel(model_VGG_dropout_reg_aug, train_generator)


# In[28]:


(val_y, val_VGG_dropout_reg_aug_pred_y) = predictModel(model_VGG_dropout_reg_aug, validation_generator)


# In[29]:


(test_y, test_VGG_dropout_reg_aug_pred_y) = predictModel(model_VGG_dropout_reg_aug, test_generator)


# In[30]:


showMetrics(train_y, train_VGG_dropout_reg_aug_pred_y)


# In[31]:


showMetrics(val_y, val_VGG_dropout_reg_aug_pred_y)


# In[32]:


showMetrics(test_y, test_VGG_dropout_reg_aug_pred_y)


# In[33]:


plotConfusionMatrix(test_y, test_VGG_dropout_reg_aug_pred_y)


# Very minimal effect on overfitting

# ### Hyper Paramter Tuning

# In[22]:


def build_model(hp):
    reg_lambda = hp.Choice('reg_lambda', values=[0.001, 0.01])
    dropout_rate = hp.Choice('dropout_rate',values=[0.1, 0.2])
    
    model = tf.keras.Sequential([
        # VGG block 1
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(reg_lambda), input_shape=INPUT_DIM),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(reg_lambda)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # VGG block 2
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(reg_lambda)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(reg_lambda)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        #VGG block 3
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(reg_lambda)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(reg_lambda)),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(.2),
        tf.keras.layers.Dense(OUTPUT_CLASSES, activation='softmax'),
    ])

    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=hp.Choice('learning_rate', values=[0.001, 0.01])),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['categorical_accuracy'])
    return model


# In[23]:


tuner = RandomSearch(
    build_model,
    objective='val_categorical_accuracy',
    max_trials=6,  # Adjust this value as desired
    executions_per_trial=1,
    directory='tuner_directory',  # Provide a directory to store tuner results
    project_name='vgg_hyperparameter_tuning'  # Choose a project name
)

# Perform the hyperparameter search
tuner.search(aug_train_generator, epochs=25, validation_data=aug_validation_generator)

# Get the best hyperparameters and model architecture
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
best_model = tuner.get_best_models(num_models=1)[0]


# In[24]:


print(best_hyperparameters)


# In[25]:


best_model.summary()


# In[26]:


history_best = fitModel(best_model, train_gen = aug_train_generator, val_gen = aug_validation_generator )


# In[27]:


plot_learning_curve(history_best.history['loss'], history_best.history['val_loss'], 
                    history_best.history['categorical_accuracy'], history_best.history['val_categorical_accuracy'], 
                    metric_name='Accuracy')


# In[28]:


(train_y, train_best_pred_y) = predictModel(best_model, train_generator)


# In[29]:


(val_y, val_best_pred_y) = predictModel(best_model, validation_generator)


# In[30]:


(test_y, test_best_pred_y) = predictModel(best_model, test_generator)


# In[31]:


showMetrics(train_y, train_best_pred_y)


# In[32]:


showMetrics(val_y, val_best_pred_y)


# In[33]:


showMetrics(test_y, test_best_pred_y)


# In[34]:


plotConfusionMatrix(test_y, test_best_pred_y)


# - The model achieved the highest accuracy compared to other models tested.
# - It demonstrates better generalization, as evidenced.
# - The model shows improved performance in terms of accuracy and F1 score, indicating its effectiveness in correctly classifying instances.
# - The regularization technique applied has helped mitigate overfitting, resulting in improved generalization.
# - The model's ability to generalize well is supported by its performance on unseen data, suggesting it can handle real-world scenarios.

# ## Lenet

# ### Base Lenet

# In[21]:


model_leNet = tf.keras.Sequential([
    tf.keras.layers.Input(shape=INPUT_DIM),
    tf.keras.layers.Lambda(lambda x: tf.expand_dims(x[:,:,:,0], -1, name=None)),
    
    
    tf.keras.layers.Conv2D(32, (3, 3)),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
    
    tf.keras.layers.Conv2D(32, (3, 3)),
    tf.keras.layers.Activation('relu'),
    
    tf.keras.layers.Conv2D(64, (3, 3)),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
    
    tf.keras.layers.Flatten(),
    
    tf.keras.layers.Dense(64),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dense(OUTPUT_CLASSES, activation='softmax')
])


# In[22]:


model_leNet.summary()


# In[23]:


tf.keras.utils.plot_model(model_leNet, show_shapes=True)


# In[24]:


model_leNet.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['categorical_accuracy'])


# In[25]:


history_leNet = fitModel(model_leNet)


# In[26]:


plot_learning_curve(history_leNet.history['loss'], history_leNet.history['val_loss'], 
                    history_leNet.history['categorical_accuracy'], history_leNet.history['val_categorical_accuracy'], 
                    metric_name='Accuracy')


# In[29]:


(train_y, train_leNet_pred_y) = predictModel(model_leNet, train_generator)


# In[31]:


(val_y, val_leNet_pred_y) = predictModel(model_leNet, validation_generator)


# In[27]:


(test_y, test_leNet_pred_y) = predictModel(model_leNet, test_generator)


# In[30]:


showMetrics(train_y, train_leNet_pred_y)


# In[32]:


showMetrics(val_y, val_leNet_pred_y)


# In[28]:


showMetrics(test_y, test_leNet_pred_y)


# In[33]:


plotConfusionMatrix(test_y, test_leNet_pred_y)


# 
# - Accuracy is worse than VGG
# - better fitted than VGG

# ### Regularization + Dropout

# In[34]:


reg_lambda = 0.001

model_leNet_reg = tf.keras.Sequential([
    tf.keras.layers.Input(shape=INPUT_DIM),
    tf.keras.layers.Lambda(lambda x: tf.expand_dims(x[:,:,:,0], -1, name=None)),
    
    
    tf.keras.layers.Conv2D(32, (3, 3), kernel_regularizer=tf.keras.regularizers.l2(reg_lambda)),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
    
    tf.keras.layers.Conv2D(32, (3, 3), kernel_regularizer=tf.keras.regularizers.l2(reg_lambda)),
    tf.keras.layers.Activation('relu'),
    
    tf.keras.layers.Conv2D(64, (3, 3)),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
    
    tf.keras.layers.Flatten(),
    
    tf.keras.layers.Dense(64),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(OUTPUT_CLASSES, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(reg_lambda))
])


# In[35]:


model_leNet_reg.summary()


# In[36]:


model_leNet_reg.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['categorical_accuracy'])


# In[37]:


history_leNet_reg = fitModel(model_leNet_reg)


# In[38]:


plot_learning_curve(history_leNet_reg.history['loss'], history_leNet_reg.history['val_loss'], 
                    history_leNet_reg.history['categorical_accuracy'], history_leNet_reg.history['val_categorical_accuracy'], 
                    metric_name='Accuracy')


# In[39]:


(train_y, train_leNet_reg_pred_y) = predictModel(model_leNet_reg, train_generator)


# In[40]:


(val_y, val_leNet_reg_pred_y) = predictModel(model_leNet_reg, validation_generator)


# In[41]:


(test_y, test_leNet_reg_pred_y) = predictModel(model_leNet_reg, test_generator)


# In[42]:


showMetrics(train_y, train_leNet_reg_pred_y)


# In[43]:


showMetrics(val_y, val_leNet_reg_pred_y)


# In[44]:


showMetrics(test_y, test_leNet_reg_pred_y)


# In[45]:


plotConfusionMatrix(test_y, test_leNet_reg_pred_y)


# Still the accuracy is very less compared to the tuned VGG model.
# 
# <b>Overall, our best model for this task is the VGG - 3 Block + Dropout + Regularisation + Data Augmentation + Hyperparameter Tuned Model </b> 

# ## Improving Cell-type classification using Pseudo labelling

# Pseudo labelling is a technique used in semi-supervised learning, where a model trained on labeled data is used to predict labels for unlabeled data. These predicted labels are then combined with the original labeled data, creating a larger dataset with both labeled and pseudo-labeled examples

# In[30]:


# Import extra dataset
extra_data = pd.read_csv('./data_labels_extraData.csv')
extra_data


# In[31]:


# Check for missing values in main dataset
extra_data.isna().sum()


# In[32]:


extra_datagen = ImageDataGenerator(rescale=1./255, data_format='channels_last')
batch_size = 32

extra_datagen = extra_datagen.flow_from_dataframe(
        dataframe=extra_data,
        directory='./patch_images/',
        x_col="ImageName",
        target_size=(27, 27),
        batch_size=batch_size,
       class_mode=None)


# Predicting the labels for extra data

# In[34]:


y_pred = best_model.predict(extra_datagen)


# In[36]:


y_pred = np.argmax(y_pred, axis = 1)


# In[37]:


print(y_pred)


# In[38]:


extra_data['cellType'] = y_pred


# In[39]:


extra_data.head()


# In[40]:


merged_data = pd.concat([train_data, extra_data], axis=0)


# In[41]:


merged_data


# In[42]:


extra_data_train_gen = ImageDataGenerator(rescale=1./255, data_format='channels_last',
                                   rotation_range=90, width_shift_range=0.2,
                                   height_shift_range=0.2, brightness_range=[0.5,1.5])
batch_size = 32

extra_data_train_generator = extra_data_train_gen.flow_from_dataframe(
        dataframe=merged_data,
        directory='./patch_images/',
        x_col="ImageName",
        target_size=(27, 27),
        batch_size=batch_size,
       class_mode=None)


# Training our best model i.e tuned VGG-3block with maindata + extradata

# In[43]:


history_extra_data = fitModel(best_model, train_gen=extra_data_train_generator , val_gen=aug_validation_generator)


# In[44]:


(test_y, test_extra_data_pred) = predictModel(best_model, test_generator)


# In[45]:


showMetrics(test_y, test_extra_data_pred)


# After employing the VGG model as our best performing model, we decided to explore the technique of pseudo labelling in order to potentially enhance its performance. Further information can be found in the report.

# --------------------
