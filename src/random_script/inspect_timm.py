import timm


if __name__ == "__main__":
    print(timm.list_models("*efficient*", pretrained=True))
