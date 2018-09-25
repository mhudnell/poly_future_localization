import cv2
import numpy as np
import data_extract_1obj

def drawFrameRects(sample_set, frame, objId, bb, isGen, folder_dir):
    '''
    Draws the provided bounding boxes onto the specified frame and outputs the drawn image to 'folder_dir'.

    Args:
        sample_set (string): The set to get the specified frame from.
        frame (string): The frame to draw boxes on.
        objId (string): The id of the object being tracked.
        bb ((n,4) array): An array with each row being LTWH values describing a bounding box.
        isGen (boolean): Indicates whether the concluding bounding box in 'bb' was generated, or if it is the target.
        folder_dir: The folder to output the drawn images to.
    '''
    img_file = 'F:\\Car data\\kitti\\data_tracking\\training\\image_02\\' + sample_set + '\\' + frame + '.png'

    # Load img to draw on.
    img = cv2.imread(img_file)

    # 
    proposal = data_extract_1obj.transform(bb[-2], bb[-1])
    bb[-1] = proposal

    data_extract_1obj.unnormalize_sample(bb, sample_set)
    print(proposal)

    # Convert bb (LTWH values) to ints.
    bb_int = np.zeros((len(bb), 4), dtype='int32')
    for i in range(len(bb_int)):
        nums = bb[i]
        bb_int[i][0] = int(float(nums[0]))
        bb_int[i][1] = int(float(nums[1]))
        bb_int[i][2] = int(float(nums[2]))
        bb_int[i][3] = int(float(nums[3]))

    # bb_int[-1][0] = proposal[0]
    # bb_int[-1][1] = proposal[1]
    # bb_int[-1][2] = proposal[2]
    # bb_int[-1][3] = proposal[3]

    # Draw target box in green if ground truth, blue if generated (color is specified in (b,g,r) format)
    target_color = (255, 0, 0) if isGen else (0, 255, 0)

    # Draw each bounding box on the image.
    for i in range(len(bb_int)):
        if i < 10:
            img = cv2.rectangle(img, (bb_int[i][0], bb_int[i][1]), (bb_int[i][0] + bb_int[i][2], bb_int[i][1] + bb_int[i][3]), (61, 165, 244, 0.5), 5)     # draw past frames in orange
        else:
            img = cv2.rectangle(img, (bb_int[i][0], bb_int[i][1]), (bb_int[i][0] + bb_int[i][2], bb_int[i][1] + bb_int[i][3]), target_color, 5)            # draw future frame in color: 'target_color'

    # Write image to file.
    suffix = "generated" if isGen else "target"
    cv2.imwrite(folder_dir + sample_set + '_' + frame + '_' + objId + '_' + suffix + '.png', img)

    return
