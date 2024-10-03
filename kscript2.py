import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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
    train_data = pd.concat([train_data1, train_data2], ignore_index=True)
    
    # Load testing data
    test_data = pd.read_csv(sys.argv[3])
    
    # Assuming the last column is the target variable
    X = pd.concat([train_data.iloc[:, :-1], test_data.iloc[:, :-1]], ignore_index=True)
    y = pd.concat([train_data.iloc[:, -1], test_data.iloc[:, -1]], ignore_index=True)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

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

def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    # Convert categorical labels to numerical labels
    y_train_cat = pd.Categorical(y_train)
    y_test_cat = pd.Categorical(y_test)
    
    y_train_num = y_train_cat.codes
    y_test_num = y_test_cat.codes
    
    # Calculate class weights
    class_weights = class_weight.compute_class_weight('balanced',
                                                      classes=np.unique(y_train_num),
                                                      y=y_train_num)
    
    # Create a dictionary with class indices
    class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
    
    # Update the model's output layer to match the number of classes
    num_classes = len(np.unique(y_train_num))
    model.layers[-1] = Dense(num_classes, activation='softmax')
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
    
    history = model.fit(
        X_train, y_train_num,
        epochs=200,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr],
        class_weight=class_weights_dict,
        verbose=1
    )
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    accuracy = accuracy_score(y_test_num, y_pred_classes)
    print(f'Accuracy: {accuracy:.4f}')
    print(classification_report(y_test_num, y_pred_classes))

def main():
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    print("Creating and training the model...")
    num_classes = len(np.unique(y_train))
    model = create_improved_model(X_train.shape[1], num_classes)
    
    print("Model summary:")
    model.summary()
    
    print("\nTraining the model...")
    train_and_evaluate(model, X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()