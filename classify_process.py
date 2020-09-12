import base64
import io
import json
import time
from io import BytesIO

import redis as redis
import requests as req
from PIL import Image

import efficientService as service

RedisCache = redis.StrictRedis(host="localhost", port=6379, db=0)
IMAGE_QUEUE = "imageQueue"
BATCH_SIZE = 32
SERVER_SLEEP = 0.1
SERVER_SLEEP_IDLE = 0.5


def classify_process():
    while True:
        # 从redis中获取预测图像队列
        queue = RedisCache.lrange(IMAGE_QUEUE, 0, BATCH_SIZE - 1)
        print("classify_process is running")
        if len(queue) < 1:
            time.sleep(SERVER_SLEEP)
            continue

        # 遍历队列
        for item in queue:
            # step 1. 获取队列中的图像信息
            item = json.loads(item);
            image_key = item.get("imageKey")
            image_link = item.get("imageUrl")
            response = req.get(image_link)
            image = Image.open(BytesIO(response.content))

            # step 2. detect image
            image_array = service.detect(image)

            # step 3. convert image_array to byte_array
            img = Image.fromarray(image_array, 'RGB')
            img_byte_array = io.BytesIO()
            img.save(img_byte_array, format='JPEG')

            # step 4. return image_info to page
            image_info = base64.b64encode(img_byte_array.getvalue()).decode('ascii')

            RedisCache.hset(name=image_key, key="consultOut", value=image_info)

        RedisCache.ltrim(IMAGE_QUEUE, BATCH_SIZE, -1)

        time.sleep(SERVER_SLEEP)


if __name__ == '__main__':
    print("start classify_process")
    classify_process()
