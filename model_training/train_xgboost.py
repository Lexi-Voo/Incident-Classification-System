"""
Enhanced XGBoost Training with Data Augmentation and Class Weights
"""

from pathlib import Path
from integration.accident_predictor import XGBoostTrainer
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

DATASET_PATH = Path(r"C:\Users\60103\Downloads\AI_Assignmnet_2B\AI_Assignmnet_2B\test_images\traffic_images")
MODEL_SAVE_PATH = "models/model3_xgboost.h5"


def collect_dataset_with_split(dataset_path, test_size=0.15):
    """
    Collect dataset and create train/test split like your friend's code
    """
    from sklearn.model_selection import train_test_split
    
    image_paths = []
    labels = []
    
    label_map = {'none': 0, 'minor': 1, 'moderate': 2, 'severe': 3}
    
    print("\n" + "="*70)
    print("  Dataset Collection (with Train/Test Split)")
    print("="*70)
    
    # Collect from training folder only
    train_dir = dataset_path / 'training'
    
    if not train_dir.exists():
        print(f"❌ Training folder not found: {train_dir}")
        return [], [], [], []
    
    print(f"\nCollecting from: {train_dir}")
    
    for class_name, label in label_map.items():
        class_dir = train_dir / class_name
        
        if class_dir.exists():
            img_files = list(class_dir.glob('*.jpg')) + \
                       list(class_dir.glob('*.jpeg')) + \
                       list(class_dir.glob('*.png'))
            
            for img_file in img_files:
                image_paths.append(str(img_file))
                labels.append(label)
            
            print(f"  {class_name:10s}: {len(img_files):4d} images")
    
    # Split into train and test (like your friend)
    X_train, X_test, y_train, y_test = train_test_split(
        image_paths, labels,
        test_size=test_size,
        random_state=42,
        stratify=labels  # Maintain class distribution
    )
    
    print(f"\nSplit (test_size={test_size}):")
    print(f"  Training: {len(X_train)} images")
    print(f"  Testing:  {len(X_test)} images")
    
    return X_train, X_test, y_train, y_test


def calculate_class_weights(labels):
    """Calculate class weights to handle imbalance"""
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    
    # Convert to dict format for XGBoost
    weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    
    print("\n" + "="*70)
    print("  Class Weights (handling imbalance)")
    print("="*70)
    class_names = ['None', 'Minor', 'Moderate', 'Severe']
    for i, weight in weight_dict.items():
        print(f"  {class_names[i]:10s}: {weight:.3f}")
    
    # Convert labels to sample weights
    sample_weights = np.array([weight_dict[label] for label in labels])
    
    return sample_weights


def plot_xgboost_history(results, title_prefix="XGBoost", save_path=None):
    """
    Plot training and validation metrics for XGBoost
    Matching CNN plotting style exactly
    """
    plt.figure(figsize=(12, 5))
    
    # Extract metrics
    train_metrics = results['validation_0']
    val_metrics = results['validation_1']
    
    # Calculate accuracy from mlogloss (approximate)
    # Using exp(-loss) as accuracy proxy, normalized to 0-1 range
    train_loss = np.array(train_metrics['mlogloss'])
    val_loss = np.array(val_metrics['mlogloss'])
    
    # Convert loss to accuracy-like metric
    train_acc = np.exp(-train_loss)
    val_acc = np.exp(-val_loss)
    
    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(train_acc, linewidth=2, label='Train Accuracy')
    plt.plot(val_acc, linewidth=2, label='Val Accuracy')
    plt.title(f"{title_prefix} Accuracy", fontsize=14, fontweight='bold')
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Accuracy", fontsize=11)
    plt.legend(fontsize=10)
    plt.grid(False)
    
    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(train_loss, linewidth=2, label='Train Loss')
    plt.plot(val_loss, linewidth=2, label='Val Loss')
    plt.title(f"{title_prefix} Loss", fontsize=14, fontweight='bold')
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Loss", fontsize=11)
    plt.legend(fontsize=10)
    plt.grid(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Plot saved to {save_path}")
    else:
        plt.show()
    
    return plt


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """
    Plot confusion matrix for XGBoost predictions
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names,
                cbar_kws={'label': ''})
    
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=11)
    plt.ylabel('True Label', fontsize=11)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to {save_path}")
    else:
        plt.show()
    
    return plt


if __name__ == '__main__':
    print("\n╔" + "="*68 + "╗")
    print("║" + " "*15 + "Enhanced XGBoost Training" + " "*29 + "║")
    print("║" + " "*15 + "With Data Split & Class Weights" + " "*21 + "║")
    print("╚" + "="*68 + "╝")
    
    # Validate dataset path
    if not DATASET_PATH.exists():
        print(f"\n❌ ERROR: Dataset path not found: {DATASET_PATH}")
        sys.exit(1)
    
    # Collect dataset with train/test split
    X_train, X_test, y_train, y_test = collect_dataset_with_split(DATASET_PATH, test_size=0.15)
    
    if len(X_train) == 0:
        print("\n❌ ERROR: No images found!")
        sys.exit(1)
    
    # Calculate class weights
    sample_weights = calculate_class_weights(y_train)
    
    # Display distribution
    print("\n" + "="*70)
    print("  Training Set Distribution")
    print("="*70)
    class_names = ['None', 'Minor', 'Moderate', 'Severe']
    for label in range(4):
        count = y_train.count(label)
        percentage = (count / len(y_train)) * 100
        print(f"  {class_names[label]:10s}: {count:4d} ({percentage:5.1f}%)")
    
    # Train with class weights
    print("\n" + "="*70)
    print("  Training XGBoost with Enhanced Features")
    print("="*70)
    
    try:
        # Modify XGBoostTrainer to accept sample weights
        trainer = XGBoostTrainer()
        
        # Update train method to use sample weights
        trainer.setup_feature_extractor()
        
        print(f"\nStep 1: Feature Extraction")
        X_train_features = trainer.extract_features_batch(X_train)
        
        print(f"\nStep 2: Training XGBoost with Class Weights...")
        
        # Prepare validation features for eval_set
        print(f"  Extracting validation features...")
        X_test_features = trainer.extract_features_batch(X_test)
        
        # Train with sample weights and evaluation set
        trainer.model = XGBClassifier(
            n_estimators=150,  # More trees
            max_depth=8,       # Deeper trees
            learning_rate=0.05,  # Lower learning rate
            use_label_encoder=False,
            eval_metric='mlogloss',
            random_state=42,
            scale_pos_weight=1,  # For imbalanced data
        )
        
        # Fit with sample weights and evaluation set for tracking
        eval_set = [(X_train_features, np.array(y_train)), 
                    (X_test_features, np.array(y_test))]
        
        trainer.model.fit(
            X_train_features,
            np.array(y_train),
            sample_weight=sample_weights,  # ← Key difference!
            eval_set=eval_set,
            verbose=True
        )
        
        print(f"✓ Training complete")
        
        # Plot training history
        print(f"\nStep 2.5: Generating training plots...")
        results = trainer.model.evals_result()
        plot_xgboost_history(
            results, 
            title_prefix="XGBoost",
            save_path="xgboost_training_history.png"
        )
        
        # Save model
        print(f"\nStep 3: Saving model...")
        trainer.save_model_h5(MODEL_SAVE_PATH)
        print(f"✓ Model saved to {MODEL_SAVE_PATH}")
        
        # Evaluate on test set
        print("\n" + "="*70)
        print("  Evaluating on Test Set")
        print("="*70)
        
        from integration.accident_predictor import AccidentPredictor
        from sklearn.metrics import accuracy_score, classification_report
        
        predictor = AccidentPredictor(MODEL_SAVE_PATH)
        
        print("\nPredicting...")
        y_pred = []
        for i, img_path in enumerate(X_test):
            if (i + 1) % 50 == 0:
                print(f"  Progress: {i + 1}/{len(X_test)}")
            result = predictor.predict(img_path)
            y_pred.append(result['severity'])
        
        accuracy = accuracy_score(y_test, y_pred)
        
        # Calculate F1 scores
        from sklearn.metrics import f1_score
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        
        print("\n" + "="*70)
        print("  Results")
        print("="*70)
        print(f"\n📊 Test Accuracy: {accuracy:.2%}")
        print(f"📊 Macro F1-Score: {macro_f1:.4f}")
        print(f"\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=class_names))
        
        # Generate confusion matrix
        print("\n" + "="*70)
        print("  Generating Confusion Matrix")
        print("="*70)
        plot_confusion_matrix(
            y_test, 
            y_pred, 
            class_names=['minor', 'moderate', 'none', 'severe'],
            save_path='xgboost_confusion_matrix.png'
        )
        
        print("\n✓ Training Complete!\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)