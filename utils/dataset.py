import os
import shutil
from pathlib import Path

try:
    from .yandex import SearchQuery
except ImportError:
    from yandex import SearchQuery

#BASE_DIR = 'downloaded'
#@click.command()
#@click.option('--count', '-c', default=12, help='number of images')
#@click.option('--folder', '-f', default='unnamed', help='folder to save images')
#@click.option('--search', '-s', default='test', help='search request')
#def main(count, folder, search):
#    S = SearchQuery(search, count, os.path.join(BASE_DIR, folder, search))
#    S.search_pictures()

BASE_DIRECTORY = "dataset"

def download_dataset(theme, dest) -> int:
    """
    Download images for dataset.

    theme: `str`
        name of theme to search

    dest: `str`
        dataset directory
    """

    path = f'{BASE_DIRECTORY}/{dest}/'
    temp_path = f'.request/{dest}'

    theme = f'{theme} памятник'

    if not os.path.exists(path):
        os.makedirs(path)

    if not os.path.exists(temp_path):
        os.makedirs(temp_path)
    
    s = SearchQuery(search_text=theme, num_of_pic=400, path=temp_path)
    s.search_pictures()

    files = os.listdir(f'{temp_path}/images')
    for file in files:
        Path(f'{temp_path}/images/{file}').rename(f'{path}/{file}')
    
    shutil.rmtree(temp_path)

def rename_images(dest) -> int:
    """
    Change images' name to number.

    dest: `str`
        dataset directory
    """

    path = f'{BASE_DIRECTORY}/{dest}' 
    if not os.path.exists(path):
        return -1

    result = 1
    objs = os.listdir(path)
    for obj in objs:
        if obj.endswith(".jpg") or obj.endswith(".png"):
            old_file = "{}/{}".format(path, obj)
            new_file = "{}/{}.jpg.temp".format(path, result)
            os.rename(old_file, new_file)
            result += 1

    objs = os.listdir(path)
    for obj in objs:
        if obj.endswith(".temp"):
            old_file = "{}/{}".format(path, obj)
            new_file = "{}/{}".format(path, obj.rsplit('.', 1)[0])
            os.rename(old_file, new_file)
            result += 1

    return result-1

if __name__ == '__main__':
    search = [
        "памятник ленину",
        "памятник циолковскому",
        "статуя свободы",
        "памятник тельмана аэропорт",
        "памятник минину и пожарскому"
    ]

    for i, s in enumerate(search):
        download_dataset(s, str(i+1))
        rename_images(str(i+1))
