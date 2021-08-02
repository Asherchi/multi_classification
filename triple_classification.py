import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import keras as K
import numpy as np
import matplotlib.pyplot as plt


def load_data(excel_file_path):
    classification = pd.read_excel(excel_file_path, sheet_name=0, header=0)
    target_var = "label"
    features = list(classification.columns)
    features.remove(target_var)
    class_type = classification[target_var].unique()
    class_dict = dict(zip(class_type, range(len(class_type))))
    classification['target'] = classification[target_var].apply(lambda x: class_dict[x])
    lb = LabelBinarizer()
    lb.fit(list(class_dict.values()))
    transformed_labers = lb.transform(classification['target'])
    y_bin_labels = []
    for i in range(transformed_labers.shape[1]):
        y_bin_labels.append('y' + str(i))
        classification['y' + str(i)] = transformed_labers[:, i]

    train_x, test_x, train_y, test_y = train_test_split(classification[features], classification[y_bin_labels],
                                                        train_size=0.7, test_size=0.3, random_state=0)
    return train_x, test_x, train_y, test_y, class_dict


train_x, test_x, train_y, test_y, class_dict = load_data(r'C:\Users\Asher\Desktop\self_\SVM_three_data.xlsx')

init = K.initializers.glorot_uniform(seed=1)
simple_adam = K.optimizers.Adam()
model = K.models.Sequential()
model.add(K.layers.Dense(units=5, input_dim=3, kernel_initializer=init, activation='relu'))
model.add(K.layers.Dense(units=6, kernel_initializer=init, activation="relu"))
model.add(K.layers.Dense(units=3,kernel_initializer=init, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer=simple_adam, metrics=["accuracy"])

batch_size = 1
epochs = 1
print("startring training")
history = model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=1,
                    validation_data=(test_x, test_y))
print("training finished \n")

eval = model.evaluate(test_x, test_y, verbose=0)
unknown = np.array([[0.6024, 0.03579, 0.7795]], dtype=np.float32)
predicted = model.predict(unknown)
print("predicted softmax vector is:", predicted)
specied_dict = {v:k for k,v in class_dict.items()}
print("predicted species is:", specied_dict[np.argmax(predicted)])
