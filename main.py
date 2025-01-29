from src.full_model import FullModel

fullModel = FullModel(model="src/detection/best.pt")

fullModel.run("./dataset/output", "./dataset/target")

