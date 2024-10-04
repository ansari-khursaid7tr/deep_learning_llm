import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import sys

def load_and_preprocess_data():
    train_data1 = pd.read_csv(sys.argv[1])
    train_data2 = pd.read_csv(sys.argv[2])
    data = pd.concat([train_data1, train_data2], ignore_index=True)
    
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split the data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def create_improved_model(input_dim, num_classes):
    model = Sequential([
        Dense(2048, activation='relu', input_dim=input_dim, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1024, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.4),
        Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.4),
        Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(16, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=RMSprop(learning_rate=0.0003, rho=0.9, epsilon=1e-08), 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

def train_and_evaluate(model, X_train, X_val, X_test, y_train, y_val, y_test):
    y_train_cat = pd.Categorical(y_train)
    y_train_num = y_train_cat.codes
    y_val_cat = pd.Categorical(y_val)
    y_val_num = y_val_cat.codes
    y_test_cat = pd.Categorical(y_test)
    y_test_num = y_test_cat.codes
    
    class_weights = class_weight.compute_class_weight('balanced',
                                                      classes=np.unique(y_train_num),
                                                      y=y_train_num)
    class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
    
    num_classes = len(np.unique(y_train_num))
    model.layers[-1] = Dense(num_classes, activation='softmax')
    model.compile(optimizer=RMSprop(learning_rate=0.0003, rho=0.9, epsilon=1e-08),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.00001)
    
    history = model.fit(
        X_train, y_train_num,
        validation_data=(X_val, y_val_num),
        epochs=500,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr],
        class_weight=class_weights_dict,
        verbose=1
    )
    
    # Evaluate on validation set
    y_val_pred = model.predict(X_val)
    y_val_pred_classes = np.argmax(y_val_pred, axis=1)
    val_accuracy = accuracy_score(y_val_num, y_val_pred_classes)
    
    # Evaluate on test set
    y_test_pred = model.predict(X_test)
    y_test_pred_classes = np.argmax(y_test_pred, axis=1)
    test_accuracy = accuracy_score(y_test_num, y_test_pred_classes)
    
    final_loss = history.history['val_loss'][-1]
    loss_percentage = final_loss * 100
    
    total_params = model.count_params()
    non_trainable_params = sum([tf.size(w).numpy() for w in model.non_trainable_weights])
    non_trainable_percentage = (non_trainable_params / total_params) * 100
    
    print(f'Validation Accuracy: {val_accuracy:.4f}')
    print(f'Test Accuracy: {test_accuracy:.4f}')
    print(f'Validation Loss Percentage: {loss_percentage:.2f}%')
    print(f'Non-trainable Parameters Percentage: {non_trainable_percentage:.2f}%')
    print("Classification Report (Validation Set):")
    print(classification_report(y_val_num, y_val_pred_classes))
    print("Classification Report (Test Set):")
    print(classification_report(y_test_num, y_test_pred_classes))
    
    return history, val_accuracy, test_accuracy, loss_percentage, non_trainable_percentage

def main():
    print("Loading and preprocessing data...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess_data()
    
    print("Creating and training the model...")
    num_classes = len(np.unique(y_train))
    model = create_improved_model(X_train.shape[1], num_classes)
    
    print("Model summary:")
    model.summary()
    
    print("\nTraining and evaluating the model...")
    history, val_accuracy, test_accuracy, loss_percentage, non_trainable_percentage = train_and_evaluate(model, X_train, X_val, X_test, y_train, y_val, y_test)
    
    print("\nFinal Results:")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Validation Loss Percentage: {loss_percentage:.2f}%")
    print(f"Non-trainable Parameters Percentage: {non_trainable_percentage:.2f}%")

if __name__ == "__main__":
    main()