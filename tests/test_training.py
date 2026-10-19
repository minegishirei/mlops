from train.training import fit_classifier


def test_fit_classifier_returns_accurate_model():
    model, accuracy = fit_classifier()

    assert accuracy >= 0.8
    assert model.predict([[5.1, 3.5, 1.4, 0.2]]).shape == (1,)