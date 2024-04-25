from seg_style.style_transfer import StyleTransfer

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from scipy.ndimage import binary_dilation, distance_transform_edt

def main():
    style_transfer = StyleTransfer()
    img = Image.open('samples/cat_standing.jpg')
    img_numpy = np.array(img)

    mask = Image.open('samples/cat_standing_binary.png')
    mask_numpy = np.array(mask)

    # Step 2: Calculate the Euclidean distances from the False points to the nearest True point
    distances = distance_transform_edt(mask_numpy)  # ~ inverts the mask from True to False and vice versa
    
    # Step 3: Normalize the distances to the range [0, 1]
    mask_numpy = (distances / distances.max())**0.1


    style = Image.open('samples/neural_style_transfer_5_1.jpg')
    #for mask, style in tqdm(zip(masks, [style, style])):
        #styled_img = style_transfer(img_numpy, mask, style)
    
    
    styled_img = style_transfer(img_numpy, mask_numpy, style)

    with torch.no_grad():
        output_right_size = styled_img.squeeze().cpu().numpy()
        output_right_size = output_right_size.transpose((1, 2, 0)).copy()
        output_right_size = (output_right_size * 255).astype(np.uint8)
    Image.fromarray(output_right_size).save('test.png')




if __name__=='__main__':
    main()
