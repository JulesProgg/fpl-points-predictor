from src.model import predict_points

def test_predict_points_length():
    assert len(predict_points(3)) == 3

def test_predict_points_values():
    assert all(p == 4.0 for p in predict_points(2))