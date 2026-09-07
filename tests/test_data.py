from sklearn.datasets import load_breast_cancer

df = load_breast_cancer(as_frame=True).frame


def test_schema():
    """คอลัมน์ต้องครบ 31 ฟีเจอร์ + target"""
    assert df.shape[1] == 31
    assert "target" in df.columns


def test_no_missing():
    assert df.isnull().sum().sum() == 0


def test_two_classes():
    """ต้องมี 2 คลาส (malignant และ benign)"""
    assert df["target"].nunique() == 2


def test_mean_radius_range():
    """ค่าที่หลุดช่วงนี้แปลว่าข้อมูลต้นทางผิดปกติ"""
    assert df["mean radius"].between(5.0, 35.0).all()