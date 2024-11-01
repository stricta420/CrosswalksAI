# test.py
from ai_model import training

labels_map = {
    0: "circle",
    1: "square",
    2: "triangle"
}

t_model = training.Model()
t_model.set_labels_map(labels_map)
t_model.load_eval()
image_path = "root/label/valid/kwadrat/kwadrat_1.png"
resoult = t_model.predict_from_path(image_path)
print(f"prob: {resoult[0]}, res: {resoult[1]}")