import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image

from app.disease_info import disease_info

device = "cpu"

model = models.mobilenet_v2(weights=None)
model.classifier[1] = torch.nn.Linear(1280, 9)

checkpoint = torch.load(
    "mobilenetv2_chatbot.pth",
    map_location=device
)

model.load_state_dict(checkpoint["model_state"])
class_names = checkpoint["class_names"]

model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

CONF_THRESHOLD = 85  # %

def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    confidence = conf.item() * 100

    if confidence < CONF_THRESHOLD:
        return None, confidence, None

    disease = class_names[pred.item()]
    detail = disease_info.get(disease, "")

    return disease, confidence, detail
