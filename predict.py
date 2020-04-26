def prediction(path, model_path, top, catergory_names, gpu):
    
    #Imports
    from torchvision import models
    import torch
    from torch import nn
    #Loading Checkpoint
    checkpoint = torch.load(model_path)
    if checkpoint['arch'] == 'vgg13':
        model = models.vgg13(pretrained=True)
    elif checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
        
    for param in model.parameters():
        param.requires_grad = False
        
    model.class_to_idx = checkpoint['class_to_idx']
    inputs = model.classifier[0].in_features
    
    classifier = nn.Sequential(nn.Linear(inputs,checkpoint['hidden_units']),
                           nn.ReLU(),
                           nn.Dropout(p=0.5),
                           nn.Linear(checkpoint['hidden_units'], checkpoint['output_units']),
                           nn.LogSoftmax(dim=1)
                           )
    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    
    device = 'cpu'
    if ((gpu) and torch.cuda.is_available()):
        device= 'cuda'
    model.to(device);
    
    #Prediction
    img = process_image(path)
    
    imgn = torch.from_numpy(img).type(torch.FloatTensor)
    final_img = imgn.unsqueeze(0)
    final_img = final_img.to(device)
    log_output = model.forward(final_img)
    output = torch.exp(log_output)
    ps, labels = output.topk(top, dim=1)
    ps = ps.cpu().detach().numpy().tolist()[0] 
    labels = labels.cpu().detach().numpy().tolist()[0]
    idxclass = {val: key for key, val in model.class_to_idx.items()}
    labels  = [idxclass[label] for label in labels]
    
    return ps, labels

def process_image(path):
    
    #Imports
    from PIL import Image
    import numpy  as np
    
    img = Image.open(path)
    if img.size[0] > img.size[1]:
        img.thumbnail((10000, 256))
    else:
        img.thumbnail((256, 10000))
    left = (img.width-224)/2
    bottom = (img.height-224)/2
    right = left + 224
    top = bottom + 224
    img = img.crop((left, bottom, right, top))
    img = np.array(img)/255
    mean = np.array([0.485, 0.456, 0.406]) 
    std = np.array([0.229, 0.224, 0.225]) 
    img = (img - mean)/std
    img = img.transpose((2, 0, 1))

    return img

def Main():
    import argparse
    import json 
    
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Enter your image path", action='store')
    parser.add_argument("checkpoint", help="Enter model path", action='store')
    parser.add_argument("--top_k", action='store', default=3, type=int)
    parser.add_argument("--category_names", action='store', type=str, default="cat_to_name.json")
    parser.add_argument("--gpu", action="store_true", default=False)
    arg = parser.parse_args()
    prob, lab = prediction(arg.path, arg.checkpoint, arg.top_k, arg.category_names, arg.gpu)
    
    with open(arg.category_names, 'r') as f:
        cat_to_name = json.load(f)
  
    
    print("Prediction: ")
    for p, l in zip(prob, lab):
        print("Name: {} Probability: {}".format(cat_to_name[str(l)],p))
       
    
if __name__ == '__main__':
    Main()