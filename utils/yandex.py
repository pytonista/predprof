"""
Code by @skair72
link: https://github.com/skair72/scrapImagesYandex
"""

import json
import os
import urllib.request

import cv2
import numpy as np
import requests
from bs4 import BeautifulSoup


UA = 'Opera/9.80 (Series 60; Opera Mini/7.1.32444/34.861; U; en) Presto/2.8.119 Version/11.10'


def get_length(side):
    x = side[0][0] - side[1][0]
    y = side[0][1] - side[1][1]
    return x**2 + y**2


def get_hash(image):
    dots = get_dots(image)
    checking_array = list()
    array = list()

    for dot_first in dots:
        for dot_second in dots:
            for dot_third in dots:
                dot = (dot_first, dot_second, dot_third)
                sorted_dot = sorted(dot)
                if dot_first != dot_second and dot_first != dot_third and \
                   dot_second != dot_third and sorted_dot not in checking_array:

                    checking_array.append(sorted_dot)
                    array.append(dot)
    del checking_array

    triangle_arr = list()
    for i in array:
        sides = [get_length((i[0], i[1])), get_length((i[1], i[2])), get_length((i[0], i[2]))]
        triangle_arr.append(sorted(sides, reverse=True))

    return triangle_arr


def get_dots(img):
    if isinstance(img, str):
        img = cv2.imread(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)

    trash = np.sort(dst.ravel())[::-1][10]

    # Threshold for an optimal value, it may vary depending on the image.
    operation = dst > trash

    must_append = 10 - len(dst[operation])

    img[operation] = [0, 0, 255]

    for i, row in enumerate(dst):
        for j, line in enumerate(row):
            if must_append == 0:
                break

            if line == trash:
                img[i, j] = [0, 0, 255]
                must_append -= 1

    dots = list()
    c = 0
    for y, i in enumerate(img):
        for x, j in enumerate(i):
            if j.tolist() == [0, 0, 255]:
                dots.append((x, y))
                c += 1

    return dots


def write_hashes(path):
    a = list(os.walk(path))[0][2]

    hashes = dict()
    for c, i in enumerate(a):
        print(f'{c + 1}/{len(a)}')
        if i.endswith('.jpg'):
            h = get_hash(os.path.join(path, i))
            hashes.update({i: h})
    json.dump(hashes, open(os.path.join(path, 'config.json'), 'w+'))


def compare_hashes(hash1, hash2):
    c = 0
    c1 = 0
    while True:
        if c1 > len(hash1) - 1:
            break
        co1 = hash1[c1]
        c2 = 0
        while True:
            if c2 > len(hash2) - 1:
                c1 += 1
                break

            co2 = hash2[c2]
            if co1[0] / co2[0] == co1[1] / co2[1] == co1[2] / co2[2]:
                c += 1
                del hash1[c1]
                del hash2[c2]
                break

            c2 += 1

    return c


def compare_images(dir_name):
    json_config1 = json.load(open(os.path.join(dir_name, 'config.json')))
    json_config2 = json_config1.copy()
    start_length = len(json_config2)

    images = dict()
    for name1, hash1 in json_config1.items():
        images.update({name1: list()})
        for name2, hash2 in json_config2.items():
            if name1 != name2:
                comparing = compare_hashes(hash1[:], hash2[:])
                images[name1].append({name2: comparing})
                if comparing > 30:
                    print(name1, name2, comparing)
        del json_config2[name1]
        print(f'{len(json_config2) + 1}/{start_length}')
        json.dump(images, open(os.path.join(dir_name, 'results.json'), 'w+'))


def show_results(dir_name):
    results = json.load(open(os.path.join(dir_name, 'results.json'), 'r+'))

    for name, r in results.items():
        new_results = sorted(r, key=lambda x: next(iter(x.values())), reverse=True)
        if len(new_results) > 0:
            if next(iter(new_results[0].values())) > 10:
                print(name)
            else:
                continue
        else:
            continue
        for i in new_results:
            key, value = i.popitem()
            if value > 10:
                print(key, value, end=', ')
        print('\n')


def do_with_results(dir_name):
    results = json.load(open(os.path.join(dir_name, 'results.json'), 'r+'))
    for name, r in results.items():
        for i in r:
            key, value = i.popitem()
            if value > 43:
                print(value, name, key, 'is a copy')
            elif value > 34:
                print(value, name, key, 'maybe a copy')
            elif value > 24:
                print(value, name, key, 'maybe not a copy')

class SearchQuery:
    def __init__(self, search_text, num_of_pic, path):
        self.search_params = {
            'source': 'collections',
            'text': search_text
        }
        self.search_text = search_text
        self.num_of_pic = num_of_pic
        self.current_page = 0

        self.S = requests.session()
        self.S.headers = {
            'User-Agent': UA
        }

        self.root_dir = path
        os.makedirs(self.root_dir, exist_ok=True)
        self.images_dir = os.path.join(self.root_dir, 'images')
        os.makedirs(self.images_dir, exist_ok=True)
        self.pages_dir = os.path.join(self.root_dir, 'pages')
        os.makedirs(self.pages_dir, exist_ok=True)

        if os.path.exists(os.path.join(self.root_dir, 'hashes.json')):
            self.pictures_hashes = json.load(open(os.path.join(self.root_dir, 'hashes.json'), 'r+'))
            self.good_pic_found = int(list(self.pictures_hashes.keys())[-1][:-4])
        else:
            self.good_pic_found = 0
            self.pictures_hashes = dict()

        self.last_request = None

    def search_pictures(self):
        while self.num_of_pic > self.good_pic_found:
            if self.current_page != 0:
                self.search_params.update({'p': str(self.current_page)})

            r = self.S.get('https://m.yandex.ru/images/smart/search', params=self.search_params)

            self.last_request = r
            print(f'page {self.current_page + 1}')
            with open(os.path.join(self.pages_dir, f'{self.current_page}.html'), 'w+') as f:
                f.write(r.text)
                f.close()
            self.download_thumbnails(r)
            self.current_page += 1

    def download_thumbnails(self, page):
        soup = BeautifulSoup(page.text, 'html.parser')
        hrefs = soup.findAll('img', {'class': 'serp-item__image'})

        for img in hrefs:
            pic_url = 'https:' + str(img.get('src')[:-2] + '32')
            image = url_to_image(pic_url)

            image_hash, number, name_of_original = self.is_copy(image)
            if number != 0:
                print(f'{pic_url} is copy of {name_of_original}, {number}')
                continue

            self.good_pic_found += 1
            cv2.imwrite(os.path.join(self.images_dir, f'{self.good_pic_found}.jpg'), image)
            self.pictures_hashes.update({f'{self.good_pic_found}.jpg': image_hash})
            json.dump(self.pictures_hashes, open(os.path.join(self.root_dir, 'hashes.json'), 'w+'))
            print(f'{self.good_pic_found}/{self.num_of_pic}')

    def is_copy(self, image):
        target_hash = get_hash(image)
        if len(self.pictures_hashes) == 0:
            return target_hash, 0, ''

        for name, h in self.pictures_hashes.items():
            n = compare_hashes(h, target_hash)
            if n > 30:
                return target_hash, n, name
        return target_hash, 0, ''

def url_to_image(url):
    resp = urllib.request.urlopen(url)
    arr = np.asarray(bytearray(resp.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)
    return img
