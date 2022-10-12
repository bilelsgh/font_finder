from model import Model


if __name__ == "__main__":
    model = Model("D:/BILEL/font_finder/split_dataset")
    model.train_model()
    model.eval()