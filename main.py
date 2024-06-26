import streamlit as st
import numpy as np
from PIL import Image
import torch
from torch import nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
# Function
collected_img = 0
model_path = 'model.pth'
# Neural Network


class PotatoNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.ReLU()
        )

    def forward(self, x):
        x = nn.Flatten()(x)
        logits = self.relu_stack(x)
        return logits


# Header
st.header("Potato classifier", divider="violet")
'''
This is a potato image classifier that analyzes an image of a potato and verifies if it has blight on its leaf

### Classifing 👀

When you put the image, the model will return three possible labels:

- (0): Healthy
- (1): Early Blight
- (2): Late Blight

'''

st.header("Let's try ⭐", divider='violet')
'''
Send your image above to scan and return the result
'''

# left_column, right_column = st.columns(2, gap="medium")

# with left_column:
#     left_column.subheader("Send image here:")
#     collected_img = left_column.file_uploader(
#         label="Your image", type=['png', 'jpg'])
# with right_column:
#     if collected_img:
#         right_column.image(collected_img, caption="Your image before analyse")

# analy_button = st.button("Analyse image", use_container_width=True)


# def collectImgAndReturnPred(img):
#     t = transforms.Compose([
#         transforms.Grayscale(),
#         transforms.Resize((28, 28)),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5), (0.5))
#     ])

#     labels = {
#         0: 'Healthy',
#         1: 'Early',
#         2: 'Late'
#     }

#     open_img = Image.open(img)
#     manipulated_img = t(open_img)

#     model = PotatoNetwork()
#     model.load_state_dict(torch.load(model_path))
#     model.eval()

#     pred = labels[model(manipulated_img).argmax(1).tolist()[0]]

#     return pred, open_img


# if analy_button:
#     if collected_img:

#         prediction, img = collectImgAndReturnPred(collected_img)

#         fig, ax = plt.subplots()
#         ax.imshow(img)
#         ax.axis('off')
#         ax.text(0.5, -0.1, prediction, transform=ax.transAxes,
#                 fontsize=12, ha='center', color='blue')
#         st.pyplot(fig)
#     else:
#         st.write("Nenhuma imagem alocada❗")

'''
# Do you like it?
## Follow me!

- Github: https://github.com/JonatasMSS
- LinkedIn: https://www.linkedin.com/in/jonatasmss/
'''
