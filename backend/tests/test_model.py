import numpy as np
from src.model import BiLSTMMODEL

def test_bilstm_model_forward_pass_outputs_probabilities():
    model = BiLSTMMODEL.build(max_words=100, max_len=10, embedding_dim=8, lstm_units=4, num_classes=2)

    X_dummy = np.random.randint(0, 100, (2, 10))
    y_pred = model.predict(X_dummy)

    assert y_pred.shape[0] == 2
    assert y_pred.shape[1] == 1 or y_pred.shape[1] == 2
    assert np.all(y_pred >= 0.0) and np.all(y_pred <= 1.0)


