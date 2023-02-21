from sklearn.datasets import fetch_olivetti_faces

data = fetch_olivetti_faces()

images = data.images

target = data.target



from PIL import Image as im

image = im.fromarray(array, "RGB")

image.save('file_name.png')

