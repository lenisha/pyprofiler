import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as T
import torchvision.datasets as datasets
import torchvision.models as models
from torch_ort import ORTModule 

import torch.profiler

# model in pytorch repo with weights 
model = models.resnet50(pretrained=True)
model = ORTModule(model)

model.cuda() # load in GPU
cudnn.benchmark = True #? needed for profiler? 

# pre-process images
transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
# Dataset load 
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                        download=True, transform=transform)
# Loading ( parallel workers processes - GIL problem global lock - not running in threads)                                     
trainloader = torch.utils.data.DataLoader(trainset, batch_size=8,
                                          shuffle=True)
# calc loss (target and training) - and minimize it
criterion = nn.CrossEntropyLoss().cuda()
# back propagation 
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

device = torch.device("cuda:0")
# switch to training mode
model.train()

def output_fn(p):
    p.export_chrome_trace("./trace/resnet50_record_ort/worker0.pt.trace.json")
    

# add context manager around training loop
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(
        wait=2, # skip first 2 training steps
        warmup=2, # reach steady and skip few layers, profiling happens ignores results
        active=6), # only profile 6 steps - allows to focus and skip some layers for reducing overhead(even in prod)
    on_trace_ready=output_fn,
    record_shapes=True,
    with_stack=True
) as p:
    for step, data in enumerate(trainloader, 0):
        print("step:{}".format(step))
        inputs, labels = data[0].to(device=device), data[1].to(device=device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        p.step()
        
        if step + 1 >= 10:
            break
