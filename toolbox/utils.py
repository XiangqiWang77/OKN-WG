class BaseLogger:
    def __init__(self) -> None:
        self.info = print


def extract_title_and_question(input_string):
    lines = input_string.strip().split("\n")

    title = ""
    question = ""
    is_question = False  # flag to know if we are inside a "Question" block

    for line in lines:
        if line.startswith("Title:"):
            title = line.split("Title: ", 1)[1].strip()
        elif line.startswith("Question:"):
            question = line.split("Question: ", 1)[1].strip()
            is_question = (
                True  # set the flag to True once we encounter a "Question:" line
            )
        elif is_question:
            # if the line does not start with "Question:" but we are inside a "Question" block,
            # then it is a continuation of the question
            question += "\n" + line.strip()

    return title, question

import requests
from bs4 import BeautifulSoup
import os
import base64
from io import BytesIO

from IPython.display import HTML, display
from PIL import Image


class ImageDownloader:
    def __init__(self, folder_path='images'):
        self.folder_path = folder_path
        # Create a folder to save images if it doesn't exist
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)

    def download_image(self, url):
        try:
            # Get the image content
            response = requests.get(url)
            response.raise_for_status()  # Check if the request was successful
            # Get the image name
            image_name = os.path.join(self.folder_path, url.split('/')[-1])
            # Write the image to a file
            with open(image_name, 'wb') as f:
                f.write(response.content)
            print(f"Image downloaded: {image_name}")
        except requests.exceptions.RequestException as e:
            print(f"Failed to download image: {url}\n{e}")

    def find_and_download_images(self, webpage_url):
        try:
            # Send a request to the web page
            response = requests.get(webpage_url)
            response.raise_for_status()  # Check if the request was successful
            # Parse the web page content
            soup = BeautifulSoup(response.text, 'html.parser')
            # Find all image tags
            img_tags = soup.find_all('meta')
            # Download images containing 'jpg' in the URL
            #print(img_tags)
            for img in img_tags:
                img_url = img.get('content')
                if img_url and 'jpg' in img_url:
                    if not img_url.startswith('https'):
                        img_url = webpage_url + img_url
                    #print(img_url)
                    self.download_image(img_url)
        except requests.exceptions.RequestException as e:
            print(f"Failed to retrieve web page: {webpage_url}\n{e}")

def convert_to_base64(pil_image):
    """
    Convert PIL images to Base64 encoded strings

    :param pil_image: PIL image
    :return: Re-sized Base64 string
    """

    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")  # You can change the format if needed
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str
