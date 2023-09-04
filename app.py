# import torch, torchvision
# from torchvision import transforms
# import numpy as np
# from PIL import Image
# from pytorch_grad_cam import GradCAM
# from pytorch_grad_cam.utils.image import show_cam_on_image
# from custom_resnet import ResNetLightningModel
# import gradio as gr
# import os

# model = ResNetLightningModel()
# model.load_state_dict(torch.load("model_checkpoint.pt", map_location=torch.device('cpu')), strict=False)

# inv_normalize = transforms.Normalize(
#     mean=[-0.50/0.23, -0.50/0.23, -0.50/0.23],
#     std=[1/0.23, 1/0.23, 1/0.23]
# )
# classes = ('plane', 'car', 'bird', 'cat', 'deer',
#            'dog', 'frog', 'horse', 'ship', 'truck')

# def greet(name):
#     return f"Hello {name}!"

# def grad_cam_view(input_img, transparency = 0.5, target_layer_number = -1, num_classes=3):
#     transform = transforms.ToTensor()
#     org_img = input_img
#     input_img = transform(input_img)
#     input_img = input_img
#     input_img = input_img.unsqueeze(0)
#     outputs = model(input_img)
#     softmax = torch.nn.Softmax(dim=0)
#     o = softmax(outputs.flatten())
#     confidences = {classes[i]: float(o[i]) for i in range(10)}
#     print (confidences)
#     _, prediction = torch.max(outputs, 1)
#     target_layers = [model.resnet2[target_layer_number]]
#     cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
#     grayscale_cam = cam(input_tensor=input_img, targets=None)
#     grayscale_cam = grayscale_cam[0, :]
#     img = input_img.squeeze(0)
#     img = inv_normalize(img)
#     rgb_img = np.transpose(img, (1, 2, 0))
#     rgb_img = rgb_img.numpy()
#     visualization = show_cam_on_image(org_img/255, grayscale_cam, use_rgb=True, image_weight=transparency)
#     return  visualization,confidences


# title = "CIFAR10 trained on ResNet18 Model with GradCAM"
# description = "A simple Gradio interface to infer on ResNet model, and get GradCAM results"
# path = "example"
# examples = [["example/cat.jpg", 0.5, -1], 
#             ["example/dog.jpg", 0.1, -3],
#             ["example/deer.jpg", 0.3, -1],
#             ["example/bird.jpg", 0.6, -2],
#             ["example/deer.jpg", 0.2, -1],
#             ["example/frog.jpg", 0.7, -3],
#             ["example/horse.jpg", 0.7, -1],
#             ["example/ship.jpg", 0.9, -2],
#             ["example/tejas.jpg", 0.2, -3],
#             ["example/toy_car.jpg", 0.3, -1],
#             ["example/truck.jpg", 0.4, -2]]


# demo = gr.Interface(fn=greet, inputs="text", outputs="text")


# demo = gr.Interface(
#     grad_cam_view, 
#     inputs = [gr.Image(shape=(32, 32), label="Input Image"), 
#             gr.Slider(0, 1, value = 0.5, label="Opacity of GradCAM"), 
#             gr.Slider(-3, -1, value = -2, step=1, label="Which Layer?"),
#             gr.Slider(1, 10, value = 3, step=1, label="Number of Class confidences")], 
#     outputs = [gr.Image(shape=(64, 64), label="Output").style(width=256, height=256), 
#                 gr.Label(num_top_classes=4)], 
#     title = title,
#     description = description,
#     examples = examples,
# )

# with gr.Blocks() as demo:
    
    
#     gr.Interface(
#     grad_cam_view, 
#     inputs = [gr.Image(shape=(32, 32), label="Input Image"), 
#             gr.Slider(0, 1, value = 0.5, label="Opacity of GradCAM"), 
#             gr.Slider(-3, -1, value = -2, step=1, label="Which Layer?"),
#             gr.Slider(1, 10, value = 3, step=1, label="Number of Class confidences")], 
#     outputs = [gr.Image(shape=(64, 64), label="Output").style(width=256, height=256), 
#                 gr.Label(num_top_classes=4)], 
#     title = title,
#     description = description,
#     examples = examples,
#                 )
    
    
#     gr.Markdown("Look at me...")

# demo.launch(server_name="0.0.0.0",server_port=os.environ.get('6024'), share=True)



#######################################################################


import numpy as np
import gradio as gr
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch
from torchvision import datasets, transforms
from custom_resnet import ResNetLightningModel
import random
import os
import random


model = ResNetLightningModel()
model.load_from_checkpoint('model_checkpoint.pt')
model.eval()


classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')

images = []

def run_model(input_img, transparency = 0.5, target_layer = -1, input_slider_classes = 3):
       
    mean=[0.49139968, 0.48215827, 0.44653124]
    std=[0.24703233, 0.24348505, 0.26158768]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    orginal_img = input_img
    input_img = transform(input_img)
    input_img = input_img.unsqueeze(0)
    outputs = model(input_img)
    # softmax = torch.nn.Softmax(dim=0)
    # o = softmax(outputs.flatten())
    o = outputs.flatten()
    confidences = {classes[i]: float(o[i]) for i in range(10)}
    _, prediction = torch.max(outputs, 1)
    target_layers = [model.resnet2[target_layer]]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    grayscale_cam = cam(input_tensor=input_img, targets=None)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(orginal_img/255, grayscale_cam, use_rgb=True, image_weight=transparency)
    print (input_slider_classes, type(input_slider_classes))
    print  (confidences, type(confidences))
    return confidences, visualization

def inference(input_img, transparency = 0.5, target_layer =-2,input_slider_classes=10):
    confidences, visualization = run_model(input_img, transparency, target_layer, input_slider_classes)
    top_classes = dict(sorted(confidences.items(), key=lambda x: x[1], reverse=True)[:input_slider_classes])
    return top_classes, visualization

def change_missclassified_view(choice):
    if choice == "Yes":
        return misclassified_dialog_box.update(visible=True)
    else:
        return misclassified_dialog_box.update(visible=False)


def get_images():
  counter = 29
  if images == []:
    while counter>0:
      image_path = f'misclassified_images/{counter}.jpg'
      images.append(image_path)
      counter -=1
  return images


# def show_misclassified_images(number_of_missclassified):
#     images = get_images()
#     output_gallery = []
#     for image_path in images:
#         image = Image.open(image_path)
#         image_array = np.asarray(image)
#         visualization = inference(image_array, gradcam, transparency, target_layer)[-1]
#         output_gallery.append(visualization)
    
#     return {
#         gallery: output_gallery[:number_of_missclassified]
#     }
    


def show_missclassified(input_radio_misclassification,input_slider_misclassified):
    if input_radio_misclassification  == "Yes":
        images = [
            (f"misclassified_images/{i}.jpg", f"label {i}") for i in range(input_slider_misclassified+1)
        ]
        return images
    else:
        return None

with gr.Blocks() as demo:
    gr.Markdown("# Lighting DavidNet")
    gr.Markdown("### CIFAR 10 Classifier with GradCAM with DavidNet")
    with gr.Tab('Grad Cam Images'):
        with gr.Row():
            with gr.Column(scale=0.25):
                input_image = gr.Image(shape=(32, 32), label="Input Image")
                with gr.Row():
                    clear_btn_main = gr.ClearButton()
                    submit_btn_main = gr.Button("Submit")

                with gr.Column(visible=True) as gradcam_dialog_box:
                    input_slider1 = gr.Slider(0, 1, value = 0.5, label="Opacity of GradCAM")
                    input_slider2 = gr.Slider(-3, 0, value = -1, step=-1, label="Which Layer?")
                    input_slider_classes = gr.Slider(1, 10, value = 10, step=1, label="How Many Classes you want to see?")
                
            with gr.Column(scale=0.25):
                output_classes = gr.Label(label="Output Labels")
            with gr.Column(scale=0.5):
                output_image = gr.Image(shape=(32, 32), label="Classification Output(Default: Without GradCAM)").style(width=512, height=512)

            submit_btn_main.click(
                                    fn=inference, 
                                    inputs=[input_image, input_slider1, input_slider2,input_slider_classes], 
                                    outputs=[output_classes, output_image]
                                )
            
            clear_btn_main.click(
                lambda: [None, "No", 0.5, 3, 3,"No",3,3, None,None], 
                outputs=[input_image, input_slider1, input_slider2, input_slider_classes, output_classes, output_image])
            
        with gr.Row(scale=0.5):
            with gr.Column(scale=0.5):
                gr.Markdown("## Examples")
                gr.Examples(examples = [["example/cat.jpg", 0.5, -1,2], 
                                        ["example/dog.jpg", 0.1, -3,4],
                                        ["example/deer.jpg", 0.3, -1,5],
                                        ["example/bird.jpg", 0.6, -2,6],
                                        ["example/frog.jpg", 0.7, -3,3],
                                        ["example/horse.jpg", 0.7, -1,2],
                                        ["example/ship.jpg", 0.9, -2,4],
                                        ["example/tejas.jpg", 0.2, -3,2],
                                        ["example/toy_car.jpg", 0.3, -1,3],
                                        ["example/truck.jpg", 0.4, -2,3]],
                inputs=[input_image, input_slider1, input_slider2, input_slider_classes],
                outputs=[output_classes, output_image],
                fn=inference,
                cache_examples=True,
                examples_per_page=5
                )
        
            with gr.Column(scale=0.5):
                gr.Markdown("## Misclassified Images")
                input_radio_misclassification = gr.Radio(choices = ["Yes", "No"], value="Yes",label="Do you want to see misclassified images?")
                # if input_radio_misclassification =="Yes":
                    # images = get_images()               
                
                # #with gr.Column(visible=False) as misclassified_dialog_box:
                input_slider_misclassified = gr.Slider(0, 29, value = 29, step=1, label="Number of misclassified images to view?")
            
            #with gr.Column(visible=True) as misclassified_output_box:
            #gallery =  gr.Gallery(label="Misclassified Gallery", show_label=False, elem_id="gallery").style(columns=[5], rows=[6], object_fit="contain", height="auto")
                
                gallery = gr.Gallery(
                        label="Generated images", show_label=False, elem_id="gallery"
                    , columns=[5], rows=[5], object_fit="contain", height="auto")
                
                with gr.Row():
                    gr.Button("Click to See Missclassifed images", scale=1).click(show_missclassified,inputs=[input_radio_misclassification,input_slider_misclassified], outputs=[gallery])

demo.launch(server_name="0.0.0.0",server_port=os.environ.get('6024'), share=True)
