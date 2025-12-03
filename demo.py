"""
Demo application for face verification using trained models.
Provides a GUI with camera feed and verification options.
"""
import sys
import os
import cv2
import numpy as np
import torch
import pickle
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QComboBox, QPushButton, QTextEdit)
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap, QFont
from facenet_pytorch import InceptionResnetV1, MTCNN
from scipy.spatial.distance import cosine

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
STRANGER_CLASS = 'stranger'


def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors"""
    return 1 - cosine(vec1, vec2)


def extract_embedding(image, mtcnn, resnet, device=DEVICE):
    """Extract embedding from an image"""
    try:
        # Convert BGR to RGB for facenet
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face = mtcnn(img_rgb)
        if face is None:
            return None
        
        face = face.unsqueeze(0).to(device)
        emb = resnet(face).detach().cpu().numpy()[0]
        return emb
    except Exception as e:
        print(f"Error extracting embedding: {e}")
        return None


def load_cosine_model(model_path='models/cosine_similarity_model.pkl'):
    """Load cosine similarity model"""
    if not os.path.exists(model_path):
        return None
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data


def load_mlp_model(model_path, device=DEVICE):
    """Load MLP model"""
    if not os.path.exists(model_path):
        return None
    
    from experiment_mlp import SimpleMLP
    
    model_data = torch.load(model_path, map_location=device)
    model = SimpleMLP(
        input_dim=model_data['input_dim'],
        num_classes=model_data['num_classes']
    )
    model.load_state_dict(model_data['model_state_dict'])
    model.eval().to(device)
    
    return {
        'model': model,
        'label2idx': model_data['label2idx'],
        'idx2label': model_data['idx2label'],
        'threshold': model_data.get('threshold', None)
    }


class CameraThread(QThread):
    """Thread for camera capture"""
    frame_ready = pyqtSignal(np.ndarray)
    
    def __init__(self):
        super().__init__()
        self.running = False
        self.cap = None
    
    def start_capture(self):
        self.running = True
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open camera")
            return
        self.start()
    
    def stop_capture(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.wait()
    
    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.frame_ready.emit(frame)
            self.msleep(33)  # ~30 FPS


class VerificationDemo(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_frame = None
        self.mtcnn = None
        self.resnet = None
        self.cosine_model = None
        self.mlp3_model = None
        self.mlp4_model = None
        self.current_model_type = None
        self.camera_thread = None
        
        self.init_ui()
        self.init_models()
        self.start_camera()
    
    def init_ui(self):
        """Initialize the UI"""
        self.setWindowTitle("Face Verification Demo")
        self.setGeometry(100, 100, 1200, 700)
        
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        # Left panel - Controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setFixedWidth(300)
        left_panel.setStyleSheet("""
            QWidget {
                background-color: #f5f5f5;
                border-radius: 5px;
            }
        """)
        
        # Title
        title = QLabel("Face Verification")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #2c3e50; padding: 10px;")
        left_layout.addWidget(title)
        
        # Model selection
        model_label = QLabel("Select Model:")
        model_label.setFont(QFont("Arial", 11, QFont.Bold))
        model_label.setStyleSheet("color: #34495e; padding-top: 10px;")
        left_layout.addWidget(model_label)
        
        self.model_combo = QComboBox()
        self.model_combo.addItems(["Cosine Similarity", "MLP3", "MLP4"])
        self.model_combo.setStyleSheet("""
            QComboBox {
                padding: 8px;
                border: 2px solid #3498db;
                border-radius: 5px;
                background-color: white;
                font-size: 11pt;
            }
            QComboBox:hover {
                border-color: #2980b9;
            }
        """)
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        left_layout.addWidget(self.model_combo)
        
        # Verification mode selection
        mode_label = QLabel("Verification Mode:")
        mode_label.setFont(QFont("Arial", 11, QFont.Bold))
        mode_label.setStyleSheet("color: #34495e; padding-top: 15px;")
        left_layout.addWidget(mode_label)
        
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Verify Kinga", "Verify Paweł", "Verify Piotr", "Verify Group Member"])
        self.mode_combo.setStyleSheet("""
            QComboBox {
                padding: 8px;
                border: 2px solid #3498db;
                border-radius: 5px;
                background-color: white;
                font-size: 11pt;
            }
            QComboBox:hover {
                border-color: #2980b9;
            }
        """)
        left_layout.addWidget(self.mode_combo)
        
        # Verify button
        self.verify_button = QPushButton("Verify Identity")
        self.verify_button.setFont(QFont("Arial", 12, QFont.Bold))
        self.verify_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                padding: 12px;
                border: none;
                border-radius: 5px;
                font-size: 12pt;
                margin-top: 20px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #21618c;
            }
        """)
        self.verify_button.clicked.connect(self.verify_face)
        left_layout.addWidget(self.verify_button)
        
        # Status/Result display
        self.status_label = QLabel("Ready to verify")
        self.status_label.setFont(QFont("Arial", 12))
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("""
            QLabel {
                background-color: #ecf0f1;
                padding: 15px;
                border-radius: 5px;
                color: #2c3e50;
                margin-top: 20px;
            }
        """)
        self.status_label.setWordWrap(True)
        left_layout.addWidget(self.status_label)
        
        # Confidence/Similarity display
        self.confidence_label = QLabel("")
        self.confidence_label.setFont(QFont("Arial", 10))
        self.confidence_label.setAlignment(Qt.AlignCenter)
        self.confidence_label.setStyleSheet("""
            QLabel {
                background-color: #ecf0f1;
                padding: 10px;
                border-radius: 5px;
                color: #7f8c8d;
                margin-top: 10px;
            }
        """)
        left_layout.addWidget(self.confidence_label)
        
        left_layout.addStretch()
        
        # Right panel - Camera feed
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        camera_label = QLabel("Camera Feed")
        camera_label.setFont(QFont("Arial", 12, QFont.Bold))
        camera_label.setAlignment(Qt.AlignCenter)
        camera_label.setStyleSheet("color: #2c3e50; padding: 5px;")
        right_layout.addWidget(camera_label)
        
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setStyleSheet("""
            QLabel {
                background-color: #2c3e50;
                border: 2px solid #34495e;
                border-radius: 5px;
            }
        """)
        self.camera_label.setMinimumSize(640, 480)
        right_layout.addWidget(self.camera_label)
        
        # Instructions
        instructions = QLabel("Click 'Verify Identity' button or press SPACE\nPress ESC to exit")
        instructions.setFont(QFont("Arial", 10))
        instructions.setAlignment(Qt.AlignCenter)
        instructions.setStyleSheet("color: #7f8c8d; padding: 10px;")
        right_layout.addWidget(instructions)
        
        # Add panels to main layout
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel, stretch=1)
        
        # Set main window style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #ecf0f1;
            }
        """)
    
    def init_models(self):
        """Initialize face detection and recognition models"""
        print("Loading face detection and recognition models...")
        self.mtcnn = MTCNN(image_size=160, margin=0, keep_all=False, device=DEVICE)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)
        
        # Load trained models
        print("Loading trained models...")
        self.cosine_model = load_cosine_model()
        if self.cosine_model:
            print("✓ Cosine similarity model loaded")
        else:
            print("⚠ Cosine similarity model not found")
        
        self.mlp3_model = load_mlp_model('models/mlp3_model.pth', DEVICE)
        if self.mlp3_model:
            print("✓ MLP3 model loaded")
        else:
            print("⚠ MLP3 model not found")
        
        self.mlp4_model = load_mlp_model('models/mlp4_model.pth', DEVICE)
        if self.mlp4_model:
            print("✓ MLP4 model loaded")
        else:
            print("⚠ MLP4 model not found")
        
        # Set default model
        self.on_model_changed(self.model_combo.currentText())
    
    def on_model_changed(self, model_name):
        """Handle model selection change"""
        self.current_model_type = model_name.lower().replace(' ', '_')
        print(f"Selected model: {model_name}")
    
    def start_camera(self):
        """Start camera capture"""
        self.camera_thread = CameraThread()
        self.camera_thread.frame_ready.connect(self.update_frame)
        self.camera_thread.start_capture()
    
    def update_frame(self, frame):
        """Update camera frame display"""
        self.current_frame = frame.copy()
        
        # Convert to RGB for display
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(
            self.camera_label.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        self.camera_label.setPixmap(scaled_pixmap)
    
    def verify_face(self):
        """Perform face verification on current frame"""
        if self.current_frame is None:
            self.status_label.setText("No frame available")
            return
        
        # Extract embedding
        embedding = extract_embedding(self.current_frame, self.mtcnn, self.resnet, DEVICE)
        if embedding is None:
            self.status_label.setText("❌ No face detected")
            self.status_label.setStyleSheet("""
                QLabel {
                    background-color: #f8d7da;
                    padding: 15px;
                    border-radius: 5px;
                    color: #721c24;
                    margin-top: 20px;
                    border: 2px solid #f5c6cb;
                }
            """)
            self.confidence_label.setText("")
            return
        
        # Get verification mode
        mode = self.mode_combo.currentText()
        target_person = None
        if "Kinga" in mode:
            target_person = "Kinga"
        elif "Paweł" in mode or "Pawel" in mode:
            target_person = "Pawel"
        elif "Piotr" in mode:
            target_person = "Piotr"
        elif "Group Member" in mode:
            target_person = "any"  # Any of the three
        
        # Perform verification based on model type
        if self.current_model_type == "cosine_similarity":
            result = self.verify_cosine(embedding, target_person)
        elif self.current_model_type == "mlp3":
            result = self.verify_mlp3(embedding, target_person)
        elif self.current_model_type == "mlp4":
            result = self.verify_mlp4(embedding, target_person)
        else:
            self.status_label.setText("❌ Model not loaded")
            self.status_label.setStyleSheet("""
                QLabel {
                    background-color: #f8d7da;
                    padding: 15px;
                    border-radius: 5px;
                    color: #721c24;
                    margin-top: 20px;
                    border: 2px solid #f5c6cb;
                }
            """)
            self.confidence_label.setText("")
            return
        
        # Display result
        if result['verified']:
            self.status_label.setText(f"✅ VERIFICATION PASSED\n{result['message']}")
            self.status_label.setStyleSheet("""
                QLabel {
                    background-color: #d4edda;
                    padding: 15px;
                    border-radius: 5px;
                    color: #155724;
                    margin-top: 20px;
                    border: 2px solid #c3e6cb;
                }
            """)
        else:
            self.status_label.setText(f"❌ VERIFICATION FAILED\n{result['message']}")
            self.status_label.setStyleSheet("""
                QLabel {
                    background-color: #f8d7da;
                    padding: 15px;
                    border-radius: 5px;
                    color: #721c24;
                    margin-top: 20px;
                    border: 2px solid #f5c6cb;
                }
            """)
        
        # Display confidence/similarity
        if 'value' in result:
            if self.current_model_type == "cosine_similarity":
                self.confidence_label.setText(f"Cosine Similarity: {result['value']:.4f}")
            else:
                self.confidence_label.setText(f"Confidence: {result['value']:.4f}")
        else:
            self.confidence_label.setText("")
    
    def verify_cosine(self, embedding, target_person):
        """Verify using cosine similarity model"""
        if self.cosine_model is None:
            return {'verified': False, 'message': 'Model not loaded'}
        
        class_means = self.cosine_model['class_means']
        threshold = self.cosine_model['threshold']
        
        # Find best match
        best_similarity = -1
        best_class = STRANGER_CLASS
        
        for class_name, mean_emb in class_means.items():
            similarity = cosine_similarity(embedding, mean_emb)
            if similarity > best_similarity:
                best_similarity = similarity
                best_class = class_name
        
        # Check if above threshold
        if best_similarity < threshold:
            return {
                'verified': False,
                'message': f'Similarity too low ({best_similarity:.4f} < {threshold:.4f})',
                'value': best_similarity
            }
        
        # Check if matches target
        if target_person == "any":
            # Any group member is acceptable
            if best_class in ["Kinga", "Pawel", "Piotr"]:
                return {
                    'verified': True,
                    'message': f'Identified as {best_class}',
                    'value': best_similarity
                }
            else:
                return {
                    'verified': False,
                    'message': f'Not a group member (identified as {best_class})',
                    'value': best_similarity
                }
        else:
            # Specific person verification
            if best_class == target_person:
                return {
                    'verified': True,
                    'message': f'Verified as {target_person}',
                    'value': best_similarity
                }
            else:
                return {
                    'verified': False,
                    'message': f'Expected {target_person}, got {best_class}',
                    'value': best_similarity
                }
    
    def verify_mlp3(self, embedding, target_person):
        """Verify using MLP3 model (threshold approach)"""
        if self.mlp3_model is None:
            return {'verified': False, 'message': 'Model not loaded'}
        
        model = self.mlp3_model['model']
        idx2label = self.mlp3_model['idx2label']
        threshold = self.mlp3_model['threshold']
        
        # Get prediction
        with torch.no_grad():
            emb_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            outputs = model(emb_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            max_prob, predicted = torch.max(probabilities, 1)
        
        predicted_class = idx2label[predicted.item()]
        confidence = max_prob.item()
        
        # Apply threshold
        if confidence < threshold:
            predicted_class = STRANGER_CLASS
        
        # Check verification
        if target_person == "any":
            if predicted_class in ["Kinga", "Pawel", "Piotr"]:
                return {
                    'verified': True,
                    'message': f'Identified as {predicted_class}',
                    'value': confidence
                }
            else:
                return {
                    'verified': False,
                    'message': f'Not a group member (identified as {predicted_class})',
                    'value': confidence
                }
        else:
            if predicted_class == target_person:
                return {
                    'verified': True,
                    'message': f'Verified as {target_person}',
                    'value': confidence
                }
            else:
                return {
                    'verified': False,
                    'message': f'Expected {target_person}, got {predicted_class}',
                    'value': confidence
                }
    
    def verify_mlp4(self, embedding, target_person):
        """Verify using MLP4 model (4-class approach)"""
        if self.mlp4_model is None:
            return {'verified': False, 'message': 'Model not loaded'}
        
        model = self.mlp4_model['model']
        idx2label = self.mlp4_model['idx2label']
        
        # Get prediction
        with torch.no_grad():
            emb_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            outputs = model(emb_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            max_prob, predicted = torch.max(probabilities, 1)
        
        predicted_class = idx2label[predicted.item()]
        confidence = max_prob.item()
        
        # Check verification
        if target_person == "any":
            if predicted_class in ["Kinga", "Pawel", "Piotr"]:
                return {
                    'verified': True,
                    'message': f'Identified as {predicted_class}',
                    'value': confidence
                }
            else:
                return {
                    'verified': False,
                    'message': f'Not a group member (identified as {predicted_class})',
                    'value': confidence
                }
        else:
            if predicted_class == target_person:
                return {
                    'verified': True,
                    'message': f'Verified as {target_person}',
                    'value': confidence
                }
            else:
                return {
                    'verified': False,
                    'message': f'Expected {target_person}, got {predicted_class}',
                    'value': confidence
                }
    
    def keyPressEvent(self, event):
        """Handle key presses"""
        if event.key() == Qt.Key_Space:
            self.verify_face()
        elif event.key() == Qt.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)
    
    def closeEvent(self, event):
        """Clean up on close"""
        if self.camera_thread:
            self.camera_thread.stop_capture()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern style
    
    window = VerificationDemo()
    window.show()
    
    sys.exit(app.exec_())

