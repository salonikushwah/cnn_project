import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

class MyCustomModel(nn.Module):
    def __init__(self):
        super(MyCustomModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)

        self.fc1 = nn.Linear(128 * 18 * 18, 256)
        self.fc2 = nn.Linear(256, 6)  # 6 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 150 -> 75
        x = self.pool(F.relu(self.conv2(x)))  # 75 -> 37
        x = self.pool(F.relu(self.conv3(x)))  # 37 -> 18
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyCustomModel()
model.load_state_dict(torch.load("C:\\Users\\Hp\\OneDrive\\Desktop\\model\\custom_cnn_model.pth", map_location=device))
model.to(device)
model.eval()

# Define class names (MUST match model output dimension)
class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Streamlit UI
st.title("🧠 Image Classifier")
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption='📷 Uploaded Image')

        # Predict
        img_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img_tensor)
            _, pred = torch.max(output, 1)

            st.write("Model output shape:", output.shape)
            st.write("Predicted index:", pred.item())
            st.write("Available classes:", class_names)

            predicted_class = class_names[pred.item()]
            st.markdown(f"### 🔍 Predicted Class: **{predicted_class}**")

    except IndexError:
        st.error("⚠️ Index out of range: Update class_names to match model output.")
    except Exception as e:
        st.error(f"❌ Error: {e}")
