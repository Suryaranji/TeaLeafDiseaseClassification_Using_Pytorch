
import torch
import model
import gradio as gr
from model import model
import os
from pathlib import Path
from timeit import default_timer as timer
class_names=['Anthracnose',
  'algal leaf',
  'bird eye spot',
  'brown blight',
  'gray light',
  'healthy',
  'red leaf spot',
  'white spot']


effnet_v2_model,transform=model(len(class_names))


effnet_v2_model.load_state_dict(torch.load(f="efficientnet_v2_s.pth",
                                           map_location=torch.device("cpu")))

def predict(img):
    start=timer()
    transformed_image=transform(img).unsqueeze(0)
    effnet_v2_model.eval()
    with torch.inference_mode():
        predictions=torch.softmax(effnet_v2_model(transformed_image),dim=1)
        pred_labels_and_probs={class_names[i]:float(predictions[0][i]) for i in range(len(class_names))}

        pred_time=round(timer()-start,5
                        )
    return pred_labels_and_probs,pred_time
examples=list(Path("examples").glob("*.jpg"))
title="Tea_Leaf_Disease_Classification_UsingPytorch"
description="Classify the Input images of diseased leaves into their respective classes"
article="[Model](https://github.com/Suryaranji/TeaLeafDiseaseClassification_Using_Pytorch.git)"
demo =gr.Interface(
    fn=predict,
    inputs=gr.Image(type='pil'),
    outputs=[gr.Label(num_top_classes=3,label="predictions"),
             gr.Number(label="Time")],
    examples=examples,
    title=title,
    description=description,
    article=article
)
demo.launch()