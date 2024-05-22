import numpy as np
import cv2

# Load YOLO model
def load_yolo_model():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return net, output_layers

# Draw bounding boxes on the image
def draw_boxes(image, boxes):
    for (x, y, w, h) in boxes:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return image

# tranfer the block data structure into a np array(image)
def retrieve_bounding_box_image(blocks):
    block, (y, x) = blocks
    block_h, block_w = block.shape
    compensated_image = np.zeros((block_h, block_w), dtype=np.uint8)
    compensated_image[0:block_h, 0:block_w] = block

    return compensated_image

# Detect objects in an image
def detect_objects(image, net, output_layers, conf_threshold=0.2, nms_threshold=0.3):
    if len(image.shape) == 2 or image.shape[2] == 1:
        image_bgr = (cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)).copy()

    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image_bgr, 0.00392, (608, 608), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Ensure the bounding box fits within the image
                if x < 0:
                    w += x
                    x = 0
                if y < 0:
                    h += y
                    y = 0
                if x + w > width:
                    w = width - x
                if y + h > height:
                    h = height - y

                if w<50 and h<50:
                    continue # discard too small objects
                    
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    return [(boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]) for i in indexes.flatten()]


# for each target obj, find its corresponding obj in ref
def find_corresponding_blocks(target_blocks, ref_blocks, threshold=10):
    matched_indices = []

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    for target_block in target_blocks:
        kp_target, des_target = sift.detectAndCompute(target_block, None)
        best_match_index = -1
        max_good_matches = 0

        for idx, ref_block in enumerate(ref_blocks):
            kp_ref, des_ref = sift.detectAndCompute(ref_block, None)
            if des_target is None or len(des_target) < 2:
                matched_indices.append(-1)
                continue

            if des_ref is None or len(des_ref) < 2:
                continue

            matches = flann.knnMatch(des_target, des_ref, k=2)

            good_matches = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)

            if len(good_matches) > max_good_matches:
                max_good_matches = len(good_matches)
                best_match_index = idx

        if max_good_matches >= threshold:
            matched_indices.append(best_match_index)
        else:
            matched_indices.append(-1)

    return matched_indices