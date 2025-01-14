#import the python libraries
import cv2
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import video
import torch.nn as nn


# Load the pre-trained model and sift to eval mode
#model = models.video.r3d_18(pretrained=True)
model = video.r3d_18(weights=video.R3D_18_Weights.DEFAULT)
model.eval()
#Defining the hook for the output
def output_hook(desired_layers):
    layer_outputs = {}
     # Gather the output for modules for the specified layers
    def hook_function(module, input, output):
        for name, mod in model.named_modules():
            if mod == module:
                module_name = name
        if module_name in desired_layers:
            layer_outputs[module_name] = output.detach()
    # Registering the hook for the layer in desired_layer
    for name, module in model.named_modules():
        if name in desired_layers:
            module.register_forward_hook(hook_function)
    return layer_outputs

#Sliding Window Technique with window_size and step
def sliding_window(frames, window_size, step):
    slided_frames = []
    total_frames = len(frames)
    #loop through every window of size 32
    for i in range(0, total_frames - window_size +1, step):
        current_slide =[]
        for j in range(window_size):
            current_slide.append(frames[i+j])
        slided_frames.append(current_slide)
    return slided_frames

def layer3_feature(video_file_path):    
    #Capture the video
    desired_layers = ['layer3']
    layer_outputs = output_hook(desired_layers)
    video_frames = []
    cap = cv2.VideoCapture(video_file_path)
    
    if not cap.isOpened():
        print("Could not read the video")
        return
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        video_frames.append(frame)
    
    # Release and destroy the windows
    cap.release()
    cv2.destroyAllWindows()    
    #Defining the maximum frames to 32 and step as 16
    max_frames = 32
    step = 16
    sliding_frames = []
    # Get sequences using sliding window
    if len(video_frames)>=32:
        sliding_frames = sliding_window(video_frames, max_frames, step)
    else:
        #print("Skipping the video", video_file_path)
        #return
        diff = 32-len(video_frames)
        last_frame = video_frames[-1]
        for i in range(0, diff):
            video_frames.append(video_frames[i])
        sliding_frames = sliding_window(video_frames, max_frames, step)
    
    # Transformation pipeline
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((112, 112)),
        transforms.ToTensor()
    ])
    #Gathering all layer outputs ffrom the sliding window function
    slided_output_layer = []
    transformed_frame=[]
    for frames in sliding_frames:
        transformed_frame=[]
        for frame in frames:
            transform_frame=transform(frame)
            transformed_frame.append(transform_frame)
        tensor_frames = torch.stack(transformed_frame)
        tensor_frames = tensor_frames.permute(1, 0, 2, 3).unsqueeze(0)
        # Append the layers
        with torch.no_grad():
            output = model(tensor_frames)
        for layer_name, output in layer_outputs.items():
            slided_output_layer.append(output)
    # Stack outputs along a new dimension
    stack_layers = torch.stack(slided_output_layer, dim=0)
    #print("The output size: is",stack_layers.shape)
    # Apply max pooling across the windows (dim=0)
    maxpooling_output = torch.max(stack_layers, dim=0).values
    maxpooling_output = maxpooling_output.squeeze(0)
    maxpooling_output = maxpooling_output.view(256, 2, 4 ,14, 14)
    output_current_layer = maxpooling_output.mean(dim=[2,3,4])
    output_current_layer = output_current_layer.view(512)
    #print("The output is: ", output_current_layer)
    #print("The output size: is",output_current_layer.shape)
    return output_current_layer.tolist()

def layer4_feature(video_file_path):       
    #Capture the video
    video_frames = []
    desired_layers = ['layer4']
    layer_outputs = output_hook(desired_layers)
    cap = cv2.VideoCapture(video_file_path)
    if not cap.isOpened():
        print("Could not read the video")
        return
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        video_frames.append(frame)
        # Visualizing the video frame using OpenCV
        #cv2.imshow("The captured frame is: ", frame)
        #if cv2.waitKey(30) & 0xFF == ord('q'):
        #    break
    
    # Release and destroy the windows
    cap.release()
    cv2.destroyAllWindows()    
    #Defining the maximum frames to 32 and step as 16
    max_frames = 32
    step = 16
    sliding_frames = []
    # Get sequences using sliding window
    if len(video_frames)>=32:
        sliding_frames = sliding_window(video_frames, max_frames, step)
    else:
        #print("Skipping the video", video_file_path)
        #return
        diff = 32-len(video_frames)
        last_frame = video_frames[-1]
        for i in range(0, diff):
            video_frames.append(video_frames[i])
        sliding_frames = sliding_window(video_frames, max_frames, step)
    
    # Transformation pipeline
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((112, 112)),
        transforms.ToTensor()
    ])
    #Gathering all layer outputs ffrom the sliding window function
    slided_output_layer = []
    transformed_frame=[]
    for frames in sliding_frames:
        transformed_frame=[]
        for frame in frames:
            transform_frame=transform(frame)
            transformed_frame.append(transform_frame)
        tensor_frames = torch.stack(transformed_frame)
        tensor_frames = tensor_frames.permute(1, 0, 2, 3).unsqueeze(0)
        # Append the layers
        with torch.no_grad():
            output = model(tensor_frames)
        for layer_name, output in layer_outputs.items():
            slided_output_layer.append(output)
    # Stack outputs along a new dimension
    stack_layers = torch.stack(slided_output_layer, dim=0)
    # Apply max pooling across the windows (dim=0)
    maxpooling_output = torch.max(stack_layers, dim=0).values
    maxpooling_output = maxpooling_output.squeeze(0)
    output_current_layer = maxpooling_output.mean(dim=[1,2,3])
    return output_current_layer.tolist()

# Function to preprocess the video and apply sliding window approach  
def avgpool_feature(video_file_path):       
    #Capture the video
    video_frames = []
    desired_layers = ['avgpool']
    layer_outputs = output_hook(desired_layers)
    cap = cv2.VideoCapture(video_file_path)
    if not cap.isOpened():
        print("Could not read the video")
        return
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        video_frames.append(frame)
        # Visualizing the video frame using OpenCV
        #cv2.imshow("The captured frame is: ", frame)
        #if cv2.waitKey(30) & 0xFF == ord('q'):
        #    break
    
    # Release and destroy the windows
    cap.release()
    cv2.destroyAllWindows()    
    #Defining the maximum frames to 32 and step as 16
    max_frames = 32
    step = 16
    sliding_frames = []
    # Get sequences using sliding window
    if len(video_frames)>=32:
        sliding_frames = sliding_window(video_frames, max_frames, step)
    else:
        #print("Skipping the video", video_file_path)
        #return
        diff = 32-len(video_frames)
        last_frame = video_frames[-1]
        for i in range(0, diff):
            video_frames.append(video_frames[i])
        sliding_frames = sliding_window(video_frames, max_frames, step)
    
    # Transformation pipeline
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((112, 112)),
        transforms.ToTensor()
    ])
    #Gathering all layer outputs ffrom the sliding window function
    slided_output_layer = []
    transformed_frame=[]
    for frames in sliding_frames:
        transformed_frame=[]
        for frame in frames:
            transform_frame=transform(frame)
            transformed_frame.append(transform_frame)
        tensor_frames = torch.stack(transformed_frame)
        tensor_frames = tensor_frames.permute(1, 0, 2, 3).unsqueeze(0)
        # Append the layers
        with torch.no_grad():
            output = model(tensor_frames)
        for layer_name, output in layer_outputs.items():
            slided_output_layer.append(output)
    # Stack outputs along a new dimension
    stack_layers = torch.stack(slided_output_layer, dim=0)
    # Apply max pooling across the windows (dim=0)
    maxpooling_output = torch.max(stack_layers, dim=0).values
    output_current_layer = maxpooling_output.view(512)
    return output_current_layer.tolist()
