import cv2
import numpy as np
import math
import functools
import pytesseract
import wordsegment as ws
import re
import num2words as n2w
import string

def debug_boxes(window_name, boxes, img):
    cv2.namedWindow(window_name)
    timg = img.copy()
    for box in boxes:
        cv2.rectangle(timg, (box[0],box[1]), (box[0]+box[2],box[1]+box[3]), (255,0,255), 1)
    cv2.imshow(window_name, timg)
    cv2.waitKey()

def num2words(text):
    return n2w.num2words(int(text.group(1)))

def fix_text(text):
    text = text.replace("|, I", text)
    text = re.sub(r"1([A-Z]| )", "I\g<1>", text)
    text = re.sub(r"([A-Z]*)\$([A-Z]+| )", "\g<1>S\g<2>", text)
    text = re.sub(r"([0-9]+)", num2words, text, flags=re.I)
    # text = text[:-1].translate(str.maketrans("","", string.punctuation))
    # tmp = []
    # for w in text.split(" "):
    #     tmp.append(" ".join(ws.segment(w)))
    # return " ".join(tmp)
    return " ".join(ws.segment(text))

def filter_boxes_normal_distribution_adaptive(boxes, num_sds):
    mean = functools.reduce(lambda x,y: x + y[3], boxes, 0)/len(boxes)
    standard_deviation = math.sqrt(functools.reduce(lambda x,y: x + (y[3]-mean)**2, boxes, (boxes[0][3]-mean)**2)/len(boxes))
    if standard_deviation >= 3:
        num_sds -= 1
    if standard_deviation != 0:
       return list(filter(lambda x: abs(x[3] - mean) < num_sds * standard_deviation, boxes)), mean
    return boxes, mean

def filter_boxes_normal_distribution(boxes, num_sds):
    mean = functools.reduce(lambda x,y: x + y[3], boxes, 0)/len(boxes)
    standard_deviation = math.sqrt(functools.reduce(lambda x,y: x + (y[3]-mean)**2, boxes, (boxes[0][3]-mean)**2)/len(boxes))
    if standard_deviation != 0:
       return list(filter(lambda x: abs(x[3] - mean) < num_sds * standard_deviation, boxes)), mean
    return boxes, mean

def filter_boxes_from_histogram(boxes, min_bucket_size, img_height):
    histogram = gen_histogram_from_boxes(boxes, img_height)
    tmp = []
    for box in boxes:
        x,y,w,h = box
        if sum(histogram[max(int(y+(h/2))-1, 0): min(int(y+(h/2))+1, img_height)]) > min_bucket_size and w*h > 4 and h>4:
            tmp.append([x,y,w,h])
    return tmp

def gen_histogram_from_boxes(boxes, img_height):
    histogram = [0] * img_height
    for box in boxes:
        x,y,w,h = box
        histogram[int(y+(h/2))] += 1
    return histogram

#assumes b1 is ontop of b2
def dist_between_boxes_v(b1, b2):
    return (b2[1] + b2[3]/2) - (b1[1] + b1[3]/2)

# is b1 inside b2
def is_inside(b1, b2):
    return b1[0] > b2[0] and b1[1] > b2[1] and (b1[0] + b1[2]) < (b2[0] + b2[2]) and (b1[1] + b1[3]) < (b2[1] + b2[3])

def sort_into_lines(boxes):
    boxes = sorted(boxes, key=lambda x: int(x[1]+x[3]/2))
    lines = []
    curr = 0
    lines.append([boxes[0].copy()])
    for box in boxes[1:]:
        if int(box[1]+box[3]/2) - int(lines[curr][-1][1] + lines[curr][-1][3]/2) <= 4:
            lines[curr].append(box.copy())
        else:
            lines.append([box.copy()])
            curr += 1

    tmp = []
    for line in lines:
        line = sorted(line, key=lambda x: x[0])
        tmp.append(line)
    return tmp

def dice_img(img, indices_r, indices_c, asa):
    ret = []
    for i,ind in enumerate(indices_r[:-1]):
        ret.append(img[ind:indices_r[i+1], int(indices_c[i][0] - asa):int(indices_c[i][1] + asa)])
    return ret

def draw_boxes(img, DEBUG_MODE):
    (height, width) = img.shape[0:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, gray = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)

    boxes = []
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        boxes.append([x,y,w,h])
    
    if DEBUG_MODE:
        debug_boxes("initial detection boxes", boxes, img)

    boxes = filter_boxes_from_histogram(boxes, 4, height)
    boxes, mean = filter_boxes_normal_distribution(boxes, 3)

    if DEBUG_MODE:
        debug_boxes("1 pass m/sd + histogram", boxes, img)

    tmp = []
    for box in boxes:
        is_inside_another = False
        for obox in boxes:
            if is_inside(box, obox):
                is_inside_another = True
        if not is_inside_another:
            tmp.append(box.copy())
    boxes = tmp

    if DEBUG_MODE:
        debug_boxes("1 pass m/sd + histogram + remove enclosing", boxes, img)
    
    boxes, mean = filter_boxes_normal_distribution_adaptive(boxes, 3)
    boxes = filter_boxes_from_histogram(boxes, 2, height)

    lines = sort_into_lines(boxes)

    if DEBUG_MODE:
        debug_boxes("1 pass m/sd + histogram + remove enclosing + another m/sd + histogram", boxes, img)
    
    word_row = []
    word_col = []
    word_row.append(int(lines[0][0][1]))
    average_slice_amnt = 0.1*mean
    count = 0
    if len(lines) > 1:
        average_slice_amnt = 0
        for i,line in enumerate(lines[1:]):
            line = sorted(line, key=lambda x: x[0])
            if (line[0][1] + line[0][3]) + mean > height:
                word_row.append(int((lines[i][0][1] + mean + line[0][1])/2))
                word_col.append([lines[i][0][0], lines[i][-1][0] + lines[i][-1][2]])
                lines = lines[0:i+2]
                break
            else:
                word_row.append(int((lines[i][0][1] + mean + line[0][1])/2))
                word_col.append([lines[i][0][0], lines[i][-1][0] + lines[i][-1][2]])
                average_slice_amnt += (line[0][1] - (lines[i][0][1] + mean))/2
                count += 1
        if (count == 0):
                average_slice_amnt = 0.1*mean
        else:
                average_slice_amnt /= count
                
    word_row[0] = max(int(word_row[0] - average_slice_amnt),0)
    word_row.append(min(int(lines[-1][0][1] + lines[-1][0][3] + average_slice_amnt), height))
    word_col.append([lines[-1][0][0], lines[-1][-1][0] + lines[-1][-1][2]])

    strips = dice_img(img, word_row, word_col, average_slice_amnt)
    return strips

def meme_to_text(img, DEBUG_MODE=False):
    img_top = img[0:int(img.shape[0]/2)]
    img_bot = img[int(img.shape[0]/2):img.shape[0]]

    img_top = draw_boxes(img_top, DEBUG_MODE=DEBUG_MODE)
    img_bot = draw_boxes(img_bot, DEBUG_MODE=DEBUG_MODE)

    config = "-l eng --oem 1 --psm 7" # language english, using the LSTM nerural net (oem 1), page segmentation mode 7 (single line)
    text = ["",""]
    for strip in img_top:
        try:
            strip = cv2.bitwise_not(cv2.inRange(strip, (230,230,230), (255,255,255)))
            # strip = cv2.GaussianBlur(strip, (3,3), 0)

            if DEBUG_MODE:
                cv2.namedWindow("strips")
                cv2.imshow("strips", strip)
                cv2.waitKey()

            text[0] += pytesseract.image_to_string(strip, config=config) + " "
        except Exception as e:
            print("Error, quietly ignoring: {}. Shape of strip: {}".format(e, strip.shape))
    for strip in img_bot:
        try:
            strip = cv2.bitwise_not(cv2.inRange(strip, (230,230,230), (255,255,255)))
            # strip = cv2.GaussianBlur(strip, (3,3), 0)

            if DEBUG_MODE:
                cv2.namedWindow("strips")
                cv2.imshow("strips", strip)
                cv2.waitKey()

            text[1] += pytesseract.image_to_string(strip, config=config) + " "
        except Exception as e:
            print("Error, quietly ignoring: {}. Shape of strip: {}".format(e, strip.shape))
    tmp = []
    for caption in text:
        tmp.append(fix_text(caption))
    return tmp

def run_tests(img_name, all=False, debug=False):
    ws.load()
    img = cv2.imread("test_images/" + img_name + ".jpg", cv2.IMREAD_COLOR)
    if img is None:
        img = cv2.imread("test_images/" + img_name + ".png", cv2.IMREAD_COLOR)
    print(meme_to_text(img, DEBUG_MODE=debug))
    if all:
        print("TODO")