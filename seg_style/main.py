from seg_style.style_transfer import StyleTransfer as StyleTransfer
from seg_style.style_transfer_multiple import StyleTransfer as StyleTransferMultiple
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from scipy.ndimage import binary_dilation, distance_transform_edt

def main():

    #style_normal()
    style_background()
    #create_video()


def style_normal():
    style_transfer = StyleTransfer()
    img = Image.open('samples/cat_standing.jpg')
    img_numpy = np.array(img)

    mask = Image.open('samples/cat_standing_binary.png')
    mask_numpy = np.array(mask)

    # Step 2: Calculate the Euclidean distances from the False points to the nearest True point
    #distances = distance_transform_edt(mask_numpy)  # ~ inverts the mask from True to False and vice versa
    
    # Step 3: Normalize the distances to the range [0, 1]
    #mask_numpy = (distances / distances.max())**0.1


    style = Image.open('samples/neural_style_transfer_5_1.jpg')
    #for mask, style in tqdm(zip(masks, [style, style])):
        #styled_img = style_transfer(img_numpy, mask, style)
    
    
    styled_img = style_transfer(img_numpy, mask_numpy, style)

    with torch.no_grad():
        output_right_size = styled_img.squeeze().cpu().numpy()
        output_right_size = output_right_size.transpose((1, 2, 0)).copy()
        output_right_size = (output_right_size * 255).astype(np.uint8)

    Image.fromarray(output_right_size).save('test.png')
    # create_video()
    


def style_background():
    style_transfer = StyleTransferMultiple()
    img = Image.open('samples/CR7_Messi.jpg')
    img_numpy = np.array(img)

    mask = Image.open('samples/CR7_Messi_sam_mask1.png')
    mask_numpy = np.array(mask)


    # Euclidean distances from the True points to the nearest False point
    #distances = distance_transform_edt(mask_numpy)
    
    # Increase the distances to make the mask more smooth
    #mask_numpy = (distances / distances.max())**0.1


    
    mask1 = Image.open('samples/CR7_Messi_sam_mask1.png')
    mask1_numpy = np.array(mask1)
    mask1_numpy = mask1_numpy[:, :, 0] > 0

    mask2 = Image.open('samples/CR7_Messi_sam_mask2.png')
    mask2_numpy = np.array(mask2)
    mask2_numpy = mask2_numpy[:, :, 0] > 0

    mask_numpy = mask1_numpy | mask2_numpy

    # background mask is inverse of mask_numpy
    mask_background = np.logical_not(mask_numpy)

    style_obj = Image.open('samples/great_wave.jpg')

    style_background = Image.open('samples/wassily-kandinsky.jpg')


    styled_img = style_transfer(img_numpy, mask_numpy, mask_background,
                                style_obj, style_background, num_steps=200)

    with torch.no_grad():
        output_right_size = styled_img.squeeze().cpu().numpy()
        output_right_size = output_right_size.transpose((1, 2, 0)).copy()
        output_right_size = (output_right_size * 255).astype(np.uint8)

    Image.fromarray(output_right_size).save('test.png')



def create_video():
    import os
    import moviepy.video.io.ImageSequenceClip
    image_folder = './video'
    fps = 30

    image_files = [os.path.join(image_folder, img)
                   for img in sorted(os.listdir(image_folder))
                   if img.endswith(".png")]
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
    clip.write_videofile('Style_Video.mp4')



if __name__=='__main__':
    main()
