from tensorflow.keras.models import Sequential, load_model as keras_load_model
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import os, time

class BiLSTMMODEL:
    @staticmethod
    def build(max_words=10000, max_len=150, embedding_dim=128, lstm_units=64, dropout=0.3, num_classes=2):
        model = Sequential([
            Embedding(max_words, embedding_dim, input_length=max_len),
            Bidirectional(LSTM(lstm_units, dropout=dropout, recurrent_dropout=dropout)),
            Dropout(dropout),
            Dense(1 if num_classes==2 else num_classes, activation="sigmoid" if num_classes==2 else "softmax")
        ])
        model.compile(loss="binary_crossentropy" if num_classes==2 else "sparse_categorical_crossentropy",
                      optimizer=Adam(1e-3), metrics=["accuracy"])
        return model

    @staticmethod
    def train(model, X_train, y_train, X_val, y_val, epochs=20, batch_size=32):
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5)
        start = time.time()
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                            epochs=epochs, batch_size=batch_size, callbacks=[early_stop, reduce_lr], verbose=1)
        end = time.time()
        return history, end-start

    @staticmethod
    def save(model, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        model.save(path)

    @staticmethod
    def load(path):
        return keras_load_model(path)

