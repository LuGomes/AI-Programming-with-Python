import argparse
from torchvision import models
from torch import nn, optim
import torch
from PIL import Image
import json
import numpy as np

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

def get_input_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_path', default='flowers/test/100/image_07926.jpg',
                        help='Path to image to classify')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of most likely classes to display')
    
    return parser.parse_args()


def process_image(image):
    """
    Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    """

    # Process a PIL image for use in a PyTorch model
    im = Image.open(image)
    [width,height] = im.size
    # resize so shortest side is 256px, maintain original aspect ratio
    shortest_side_size = 256
    if width <= height:
        new_width = shortest_side_size
        new_height = int(new_width * height / width)
    else:
        new_height = shortest_side_size
        new_width = int(new_height * width / height)
    im = im.resize(size=(new_width,new_height))

    #Crop center as square of 224px
    upper = (new_height - 224)//2
    left = (new_width - 224)//2
    lower = upper + 224
    right = left + 224
    im = im.crop((left,upper,right,lower))
    np_im = np.array(im,dtype='float64')

    for i in range(224):
        for j in range(224):
            np_im[i][j] /= np.array([255,255,255])
            np_im[i][j] -= np.array([0.485, 0.456, 0.406])
            np_im[i][j] /= np.array([0.229, 0.224, 0.225])
    np_im = np_im.transpose((2,0,1))
    return np_im


# Load a checkpoint and rebuild the model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.vgg16(pretrained=False)
    classifier = nn.Sequential(nn.Linear(25088, 256),
                           nn.ReLU(),
                           nn.Dropout(0.5),
                           nn.Linear(256, 102),
                           nn.LogSoftmax(dim=1))

    model.classifier = classifier
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    model.load_state_dict(checkpoint['state_model_dict'])
    optimizer.load_state_dict(checkpoint['state_optimizer_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

def main():
    # Use GPU if it's available
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    in_arg = get_input_args()
    trained_model = load_checkpoint('checkpoint.pth')
    image = torch.from_numpy(process_image(in_arg.image_path)).type(torch.FloatTensor).unsqueeze_(0)
    log_ps = trained_model(image)
    ps = torch.exp(log_ps)
    probs, classes = ps.topk(k=in_arg.top_k,dim=1)
    cls = np.zeros(in_arg.top_k)
    idx_to_class = {v: k for k, v in trained_model.class_to_idx.items()}
    cls = [idx_to_class[cl] for cl in classes[0].numpy()]
    cats = [cat_to_name[cat] for cat in cls]
    print(probs,cats)
    return probs, cats

if __name__ == "__main__":
    main()
