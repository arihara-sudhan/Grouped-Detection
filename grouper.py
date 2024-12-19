import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class TEmbeddingNet(nn.Module):
    def __init__(self, modelt):
        super(TEmbeddingNet, self).__init__()
        self.modelt = modelt
        self.modelt.head = nn.Identity()
        
    def forward(self, x):
        x = self.modelt(x)
        return x

    def get_embedding(self, x):
        return self.forward(x)

class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.enet = embedding_net

    def forward(self, x1, x2=None, x3=None):
        if x2 is None and x3 is None:
            return self.enet.get_embedding(x1)
        return (
            self.enet.get_embedding(x1),
            self.enet.get_embedding(x2),
            self.enet.get_embedding(x3),
        )

    def get_embedding(self, x):
        return self.enet.get_embedding(x)

def get_classifier(model_path):
    model = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
    model.eval()
    model.fc = nn.Identity() 
    tmodel = TEmbeddingNet(model)
    model = TripletNet(tmodel)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    try:
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    except Exception as e:
        print(f"ERROR WHILE LOADING MODEL: {e}")
    
    return model

def get_embedding_for_image(model, image):
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    elif not isinstance(image, Image.Image):
        raise ValueError("IMPUT SHOULD BE AN IMAGE")
    image = transform(image).unsqueeze(0)
    device = next(model.parameters()).device
    image = image.to(device)
    with torch.no_grad():
        embedding = model(image)
    
    return embedding

