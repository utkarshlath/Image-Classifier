def training(data_dir,checkpoint, model_name, gpu, learning_rate, hidden_units, epochs):
    #Imports
    import torch
    import numpy as np
    from torch import nn, optim
    import torch.nn.functional as F
    from torchvision import datasets, transforms, models
    
    #Transforming and Loading data
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    
    image_data =    {
                    'train_data' :datasets.ImageFolder(train_dir, transform = train_transforms),
                    'test_data'  :datasets.ImageFolder(test_dir, transform = test_transforms),
                    'valid_data' :datasets.ImageFolder(valid_dir, transform = valid_transforms)
                    }
    dataloader =    {
                    'trainloader' : torch.utils.data.DataLoader(image_data['train_data'], batch_size=64, shuffle=True),
                    'testloader'  : torch.utils.data.DataLoader(image_data['test_data'], batch_size=64),
                    'validloader' : torch.utils.data.DataLoader(image_data['valid_data'], batch_size=64)
                    }
    
    class_names = image_data['train_data'].classes
    
    #Building and Training the Classifier
    if model_name == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif model_name == 'vgg13':
        model = models.vgg13(pretrained = True)
    else:
        print("Model not recognized, using vgg13.")
        model = models.vgg13(pretrained = True)
        
    inpu = model.classifier[0].in_features
    
    for param in model.parameters():
        param.requires_grad = False
    
    classifier = nn.Sequential(nn.Linear(inpu,hidden_units),
                               nn.ReLU(),
                               nn.Dropout(p=0.5),
                               nn.Linear(hidden_units, len(class_names)),
                               nn.LogSoftmax(dim=1)
                              )
    model.classifier = classifier
    
    device = 'cpu'
    if ((gpu) and torch.cuda.is_available()):
        device= 'cuda'
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    model.to(device);
    
    #Training
    print_in = 5
    steps = 0
    running_loss = 0

    for epoch in range(epochs):
        
        for images, labels in (dataloader['trainloader']):
            
            steps+= 1
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            log_output = model.forward(images)
            loss = criterion(log_output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps%print_in == 0:
                t_loss = 0
                acc = 0
                model.eval()
                with torch.no_grad():
                    for images, labels in dataloader['testloader']:
                        
                        images, labels = images.to(device), labels.to(device)
                        log_output = model.forward(images)
                        batchloss = criterion(log_output, labels)
                        t_loss = batchloss.item()

                        output = torch.exp(log_output)
                        top_ps, top_class = output.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        acc += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_in:.3f}.. "
                      f"Test loss: {t_loss/len(dataloader['testloader']):.3f}.. "
                      f"Test accuracy: {acc/len(dataloader['testloader']):.3f}")
                running_loss = 0
                model.train()
                
    #Model Validation
    t_loss = 0
    acc = 0
    model.eval()
    with torch.no_grad():
        for images, labels in dataloader['validloader']:
            model.cuda()
            images, labels = images.to(device), labels.to(device)
            log_output = model.forward(images)
            batchloss = criterion(log_output, labels)
            t_loss = batchloss.item()

            output = torch.exp(log_output)
            top_ps, top_class = output.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            acc += torch.mean(equals.type(torch.FloatTensor)).item()
    print(
          f"Validation loss: {t_loss/len(dataloader['validloader']):.3f}.. "
          f"Validation accuracy: {acc/len(dataloader['validloader']):.3f}"
         )

    model.train() 
    
    #Saving Checkpoint
    model.class_to_idx = image_data['train_data'].class_to_idx
    model.cpu()
    torch.save({'arch': model_name,
                'state_dict': model.state_dict(), 
                'hidden_units': hidden_units,
                'output_units': len(class_names),
                'class_to_idx': model.class_to_idx}, 
                checkpoint)
    

def Main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("data_directory", help="Enter your images directory", action='store')
    parser.add_argument("--save_dir", action='store', default='classifier.pth' )
    parser.add_argument("--arch", action='store', default='vgg13')
    parser.add_argument("--learning_rate", action='store', type=float, default=0.001)
    parser.add_argument("--hidden_units", action='store', type=int, default=4096)
    parser.add_argument("--epoch", action='store', type=int, default=2)
    parser.add_argument("--gpu", action="store_true", default=False)
    arg = parser.parse_args()
    training(arg.data_directory, arg.save_dir, arg.arch, arg.gpu, arg.learning_rate, arg.hidden_units, arg.epoch)
    
if __name__ == '__main__':
    Main()
    
    