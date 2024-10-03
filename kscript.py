# pip3 install pandas numpy scikit-learn tensorflow
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import sys

def load_and_preprocess_data():
    # Load training data from two separate files
    train_data1 = pd.read_csv(sys.argv[1])
    train_data2 = pd.read_csv(sys.argv[2])
    
    # Concatenate the training data
    data = pd.concat([train_data1, train_data2], ignore_index=True)
    
    # Assuming the last column is the target variable
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

def create_improved_model(input_dim, num_classes):
    model = Sequential([
        Dense(256, activation='relu', input_dim=input_dim),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

def train_and_evaluate(model, X, y):
    # Convert categorical labels to numerical labels
    y_cat = pd.Categorical(y)
    y_num = y_cat.codes
    
    # Calculate class weights
    class_weights = class_weight.compute_class_weight('balanced',
                                                      classes=np.unique(y_num),
                                                      y=y_num)
    
    # Create a dictionary with class indices
    class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
    
    # Update the model's output layer to match the number of classes
    num_classes = len(np.unique(y_num))
    model.layers[-1] = Dense(num_classes, activation='softmax')
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    early_stopping = EarlyStopping(monitor='loss', patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=0.0001)
    
    history = model.fit(
        X, y_num,
        epochs=500,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr],
        class_weight=class_weights_dict,
        verbose=1
    )
    
    # Evaluate the model on the same data
    y_pred = model.predict(X)
    y_pred_classes = np.argmax(y_pred, axis=1)
    accuracy = accuracy_score(y_num, y_pred_classes)
    print(f'Accuracy: {accuracy:.4f}')
    print(classification_report(y_num, y_pred_classes))

def main():
    print("Loading and preprocessing data...")
    X, y = load_and_preprocess_data()
    
    print("Creating and training the model...")
    num_classes = len(np.unique(y))
    model = create_improved_model(X.shape[1], num_classes)
    
    print("Model summary:")
    model.summary()
    
    print("\nTraining and evaluating the model...")
    train_and_evaluate(model, X, y)

if __name__ == "__main__":
    main()