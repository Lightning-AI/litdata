from sklearn.preprocessing import LabelEncoder
import joblib

if __name__ == "__main__":
    data = ["Cancelation", "IBAN Change", "Damage Report"]
    # Create an instance of LabelEncoder
    label_encoder = LabelEncoder()
    label_encoder.fit(data)
    print("Classes:", label_encoder.classes_)
    encoded_labels = label_encoder.transform(data)
    print("Encoded labels:", encoded_labels)
    original_data = label_encoder.inverse_transform(encoded_labels)
    print("Original data:", original_data)
    joblib_file = "labelencoder.joblib"
    joblib.dump(label_encoder, joblib_file)
    print(f"Label encoder saved to {joblib_file}")