import json
import urllib.request
import time
import tesseract_better
import cv2
import wordsegment as ws
import csv

ws.load()
output = open("output.csv", "a", newline='')
csvwriter = csv.writer(output, delimiter="|")
postlist = open("search.json", 'r')
postlist = json.loads(postlist.read())

for i,post in enumerate(postlist["data"]):
    if "jpg" in post["url"][-3:] or "png" in post["url"][-3:]:
        ext = post["url"][-4:]
        try:
            urllib.request.urlretrieve(post["url"], filename="./images/adviceanimals" + str(i) + ext)
            img = cv2.imread("./images/adviceanimals" + str(i) + ext, cv2.IMREAD_COLOR)
            txt = tesseract_better.meme_to_text(img)
            csvwriter.writerow(txt)
            print(i, txt)
        except Exception as e:
            print("Quietly ignoring error: ", i, e)
    if (i+1) % 60 == 0:
        print("Sleeping to avoid ban...")
        output.seek(0)
        time.sleep(20)