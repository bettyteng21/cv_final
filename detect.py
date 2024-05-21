import numpy as np
import cv2

# Draw bounding boxes on the image
def draw_boxes(image, boxes):
    for (x, y, w, h) in boxes:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return image

def retrieve_bounding_box_image(blocks):
    block, (y, x) = blocks
    block_h, block_w = block.shape
    compensated_image = np.zeros((block_h, block_w), dtype=np.uint8)
    compensated_image[0:block_h, 0:block_w] = block

    return compensated_image

# Load YOLO model
def load_yolo_model():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, output_layers

# Detect objects in an image
def detect_objects(image, net, output_layers, conf_threshold=0.3, nms_threshold=0.3):
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
                    
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    return [(boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]) for i in indexes.flatten()]

# Find corresponding blocks for target objects with a distance threshold
def find_corresponding_blocks(target_boxes, ref_boxes, dist_threshold=200.0):
    target_blocks = []
    for tbox in target_boxes:
        tx, ty, tw, th = tbox
        tcenter = (tx + tw // 2, ty + th // 2)
        ref_index = -1
        min_dist = float('inf')
        for i, rbox in enumerate(ref_boxes):
            rx, ry, rw, rh = rbox
            rcenter = (rx + rw // 2, ry + rh // 2)
            dist = np.linalg.norm(np.array(tcenter) - np.array(rcenter))
            if dist < min_dist:
                min_dist = dist
                ref_index = i
        if min_dist <= dist_threshold:
            target_blocks.append(ref_index)
        else:
            target_blocks.append(-1)
    return target_blocks

