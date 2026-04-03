import tensorflow as tf
import xgboost as xgb
import h5py
import numpy as np
from PIL import Image

class AccidentPredictor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.model_type = None
        self.class_names = ['None', 'Minor', 'Moderate', 'Severe']
        self.img_size = (224, 224)
        
        # Detect and load model
        self._load_model()
    
    def _load_model(self):
        """Detect model type and load accordingly"""
        try:
            # Try loading as Keras model first
            self.model = tf.keras.models.load_model(self.model_path)
            self.model_type = 'keras'
            print(f"✓ Loaded Keras model from {self.model_path}")
        except Exception as keras_error:
            # Try loading as XGBoost model
            try:
                with h5py.File(self.model_path, 'r') as f:
                    if 'xgboost_model' in f:
                        # Load XGBoost model from HDF5
                        model_bytes = bytes(f['xgboost_model'][()])
                        
                        # Save to temporary file for XGBoost to load
                        import tempfile
                        with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='wb') as tmp:
                            tmp.write(model_bytes)
                            tmp_path = tmp.name
                        
                        # Load with XGBoost
                        self.model = xgb.Booster()
                        self.model.load_model(tmp_path)
                        self.model_type = 'xgboost'
                        
                        # Clean up temp file
                        import os
                        os.unlink(tmp_path)
                        
                        # Load feature extractor for XGBoost
                        self._load_feature_extractor()
                        
                        print(f"✓ Loaded XGBoost model from {self.model_path}")
                    else:
                        raise Exception("Unknown model format")
            except Exception as xgb_error:
                raise Exception(f"Failed to load model: Keras error: {keras_error}, XGBoost error: {xgb_error}")
    
    def _load_feature_extractor(self):
        """Load MobileNetV2 for feature extraction (needed for XGBoost)"""
        from tensorflow.keras.applications import MobileNetV2
        
        # Load pre-trained MobileNetV2 without top layer
        self.feature_extractor = MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            pooling='avg',  # Global average pooling
            weights='imagenet'
        )
        print("✓ Loaded feature extractor for XGBoost")
    
    def preprocess_image(self, image_path):
        """Preprocess image for prediction"""
        img = Image.open(image_path)
        img = img.resize(self.img_size)
        img_array = np.array(img)
        
        # Ensure RGB
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        return img_array
    
    def predict(self, image_path):
        """Predict accident severity from image"""
        img_array = self.preprocess_image(image_path)
        
        if self.model_type == 'keras':
            # Direct Keras prediction
            predictions = self.model.predict(img_array, verbose=0)
            probabilities = predictions[0]
            
        elif self.model_type == 'xgboost':
            # Extract features first
            features = self.feature_extractor.predict(img_array, verbose=0)
            
            # Convert to DMatrix for XGBoost
            dmatrix = xgb.DMatrix(features)
            
            # Predict with XGBoost
            probabilities = self.model.predict(dmatrix)[0]
        
        else:
            raise Exception("Unknown model type")
        
        # Get predicted class
        predicted_class = np.argmax(probabilities)
        confidence = probabilities[predicted_class]
        
        result = {
            'severity': int(predicted_class),
            'class_name': self.class_names[predicted_class],
            'confidence': float(confidence),
            'probabilities': probabilities.tolist(),
            'model_type': self.model_type
        }
        
        return result
    
    def predict_severity_type(self, image_path):
        """Get accident type (0-3) for travel time calculation"""
        result = self.predict(image_path)
        return result['severity']