import torch
import gradio as gr

from PIL import Image

from torchvision import transforms

from model import get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_model(2)  # 2 classes: Cat and Dog

model.load_state_dict(
    torch.load(
        "best_resnet_model.pth",
        map_location=device
    )
)

model.to(device)

model.eval()

train_mean = [0.4888 , 0.4557 , 0.4175]
train_std = [0.2228 , 0.2182 , 0.2185]

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(
        train_mean,
        train_std
    )
])

def predict(image):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
    classes = ['Cat', 'Dog']
    predicted_class = classes[predicted.item()]
    return f"{predicted_class} ({confidence.item()*100:.2f}%)"

app = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Text(label="Prediction"),
    title="Cat vs Dog Classifier",
    description="Upload an image of a cat or a dog."
)


app.launch()
