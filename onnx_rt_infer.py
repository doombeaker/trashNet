import sys
import onnxruntime as ort
from PIL import Image
import numpy as np


class TrashClassifier:
    def __init__(self):
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
        self.ort_sess = ort.InferenceSession("trash40.onnx")

    def find_max(self, list_obj):
        max_index = -1
        idx = 0
        max_value = -1e8
        for item in list_obj:
            if item > max_value:
                max_value = item
                max_index = idx
            idx += 1
        return max_index

    def image_process(self, image):
        # resize with keeping ratio
        larger_edge = image.height
        smaller_edge = image.width
        if image.width > image.height:
            larger_edge = image.width
            smaller_edge = image.height
            image = image.resize((int(256 * larger_edge / smaller_edge), 256))
        else:
            image = image.resize((256, int(256 * larger_edge / smaller_edge)))

        # crop
        DST_WIDTH = 224
        DST_HEIGHT = 224
        center_x = image.width / 2
        center_y = image.height / 2

        image = image.crop(
            (
                center_x - DST_WIDTH / 2,
                center_y - DST_HEIGHT / 2,
                center_x + DST_WIDTH / 2,
                center_y + DST_HEIGHT / 2,
            )
        )

        # normalize
        ary = np.array(image, dtype=np.float32)
        ary = ary / 255.0
        ary = np.transpose(ary, (2, 0, 1))

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        for i in range(0, ary.shape[0]):
            ary[i] = (ary[i] - mean[i]) / std[i]
        return ary

    def predict(self, image):
        image = self.image_process(image)
        image = image.reshape(1, *image.shape)
        output = self.ort_sess.run(None, {"input": image})
        output = output[0]
        predict_idx = self.find_max(output[0].tolist())
        real_index = self.INDEX_MAP[predict_idx]
        return self.NAME_MAP[str(real_index)]


def make_prediction(image_path):
    image = Image.open(image_path)
    clf = TrashClassifier()
    ans = clf.predict(image)
    return f"{ans}"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("请输入图片路径！")
    else:
        image_path = sys.argv[1]
        image = Image.open(image_path)
        clf = TrashClassifier()
        ans = clf.predict(image)
        print(f"{ans}")
