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
from tensorflow.keras.regularizers import l2
import sys

def load_and_preprocess_data(train_file1, train_file2, test_file):
    train_data1 = pd.read_csv(train_file1)
    train_data2 = pd.read_csv(train_file2)
    test_data = pd.read_csv(test_file)
    
    train_data = pd.concat([train_data1, train_data2], ignore_index=True)
    
    X_train = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1]
    
    X_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, y_train, X_test_scaled, y_test

def create_improved_model(input_dim, num_classes):
    model = Sequential([
        Dense(512, activation='relu', input_dim=input_dim, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    y_train_cat = pd.Categorical(y_train)
    y_train_num = y_train_cat.codes
    y_test_cat = pd.Categorical(y_test)
    y_test_num = y_test_cat.codes
    
    class_weights = class_weight.compute_class_weight('balanced',
                                                      classes=np.unique(y_train_num),
                                                      y=y_train_num)
    class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
    
    num_classes = len(np.unique(y_train_num))
    model.layers[-1] = Dense(num_classes, activation='softmax')
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.00001)
    
    # Split the data into train and validation sets
    val_split = 0.10
    split_index = int(len(X_train) * (1 - val_split))
    X_train_split, X_val = X_train[:split_index], X_train[split_index:]
    y_train_split, y_val = y_train_num[:split_index], y_train_num[split_index:]
    
    history = model.fit(
        X_train_split, y_train_split,
        validation_data=(X_val, y_val),
        epochs=500,
        batch_size=64,
        callbacks=[early_stopping, reduce_lr],
        class_weight=class_weights_dict,
        verbose=1
    )
    
    # Get final validation metrics
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    
    # Get final test metrics
    test_loss, test_accuracy = model.evaluate(X_test, y_test_num, verbose=0)
    
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    total_params = model.count_params()
    non_trainable_params = sum([tf.size(w).numpy() for w in model.non_trainable_weights])
    non_trainable_percentage = (non_trainable_params / total_params) * 100
    
    print(f'Final Validation Accuracy: {val_accuracy:.4f}')
    print(f'Final Validation Loss: {val_loss:.4f}')
    print(f'Final Test Accuracy: {test_accuracy:.4f}')
    print(f'Final Test Loss: {test_loss:.4f}')
    print(f'Non-trainable Parameters Percentage: {non_trainable_percentage:.2f}%')
    print(classification_report(y_test_num, y_pred_classes))
    
    return history, val_accuracy, val_loss, test_accuracy, test_loss, non_trainable_percentage

def main():
    if len(sys.argv) != 4:
        print("Usage: python script.py <train_file1.csv> <train_file2.csv> <test_file.csv>")
        sys.exit(1)

    train_file1 = sys.argv[1]
    train_file2 = sys.argv[2]
    test_file = sys.argv[3]

    print("Loading and preprocessing data...")
    X_train, y_train, X_test, y_test = load_and_preprocess_data(train_file1, train_file2, test_file)
    
    print("Creating and training the model...")
    num_classes = len(np.unique(y_train))
    model = create_improved_model(X_train.shape[1], num_classes)
    
    print("Model summary:")
    model.summary()
    
    print("\nTraining and evaluating the model...")
    history, val_accuracy, val_loss, test_accuracy, test_loss, non_trainable_percentage = train_and_evaluate(model, X_train, X_test, y_train, y_test)
    
    print("\nFinal Results:")
    print(f"Final Validation Accuracy: {val_accuracy:.4f}")
    print(f"Final Validation Loss: {val_loss:.4f}")
    print(f"Final Test Accuracy: {test_accuracy:.4f}")
    print(f"Final Test Loss: {test_loss:.4f}")
    print(f"Non-trainable Parameters Percentage: {non_trainable_percentage:.2f}%")

if __name__ == "__main__":
    main()