import os
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import sys

saved_file = "./trash_classifier_40_model.pth"


class TrashNet(torch.nn.Module):
    def __init__(self):
        super(TrashNet, self).__init__()
        self.INDEX_MAP = [
            0,
            1,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            2,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            3,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            4,
            5,
            6,
            7,
            8,
            9,
        ]
        self.NAME_MAP = {
            "0": "其他垃圾/一次性快餐盒",
            "1": "其他垃圾/污损塑料",
            "2": "其他垃圾/烟蒂",
            "3": "其他垃圾/牙签",
            "4": "其他垃圾/破碎花盆及碟碗",
            "5": "其他垃圾/竹筷",
            "6": "厨余垃圾/剩饭剩菜",
            "7": "厨余垃圾/大骨头",
            "8": "厨余垃圾/水果果皮",
            "9": "厨余垃圾/水果果肉",
            "10": "厨余垃圾/茶叶渣",
            "11": "厨余垃圾/菜叶菜根",
            "12": "厨余垃圾/蛋壳",
            "13": "厨余垃圾/鱼骨",
            "14": "可回收物/充电宝",
            "15": "可回收物/包",
            "16": "可回收物/化妆品瓶",
            "17": "可回收物/塑料玩具",
            "18": "可回收物/塑料碗盆",
            "19": "可回收物/塑料衣架",
            "20": "可回收物/快递纸袋",
            "21": "可回收物/插头电线",
            "22": "可回收物/旧衣服",
            "23": "可回收物/易拉罐",
            "24": "可回收物/枕头",
            "25": "可回收物/毛绒玩具",
            "26": "可回收物/洗发水瓶",
            "27": "可回收物/玻璃杯",
            "28": "可回收物/皮鞋",
            "29": "可回收物/砧板",
            "30": "可回收物/纸板箱",
            "31": "可回收物/调料瓶",
            "32": "可回收物/酒瓶",
            "33": "可回收物/金属食品罐",
            "34": "可回收物/锅",
            "35": "可回收物/食用油桶",
            "36": "可回收物/饮料瓶",
            "37": "有害垃圾/干电池",
            "38": "有害垃圾/软膏",
            "39": "有害垃圾/过期药物",
        }
        self.model_ft = torchvision.models.mobilenet_v2()
        self.model_ft.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0), torch.nn.Linear(self.model_ft.last_channel, 40),
        )
        self.model_ft.load_state_dict(torch.load(saved_file))

    def transform(self, image):
        image_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image = image_transform(image)
        return image

    def forward(self, image):
        image = self.transform(image)
        image = image.view(1, image.size()[0], image.size()[1], image.size()[2])
        output = self.model_ft(image)
        _, pred = torch.max(output, 1)
        real_index = self.INDEX_MAP[pred.item()]
        return self.NAME_MAP[str(real_index)]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("请输入图片路径！")
    else:
        image_path = sys.argv[1]
        image = Image.open(image_path)
        trash_classifier = TrashNet()
        trash_classifier.model_ft.eval()
        ans = trash_classifier(image)
        print(f"识别结果：{ans}")
