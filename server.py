from flask import Flask, render_template, request, url_for, send_file
from werkzeug.utils import secure_filename
import os
import random
from onnx_rt_infer import make_prediction
import json

app = Flask(__name__)
app.config["SECRET_KEY"] = "12345"
app.config["UPLOAD_FOLDER"] = "./temp"
base_dir = os.environ.get('BASE_DIR', '')

@app.route(f"{base_dir}/v1/index", methods=["GET", "POST"])
def home():
    if request.method == "GET":
        return render_template("home.html", base_dir=base_dir)
    if request.method == "POST":
        if "image" not in request.files:
            json_obj = {"image_url": "", "prediction": "文件选择错误"}
            return json.dumps(json_obj)
        image_file = request.files["image"]

        if image_file.filename == "":
            json_obj = {"image_url": "", "prediction": "上传文件为空"}
            return json.dumps(json_obj)

        if image_file and is_allowed_file(image_file.filename):
            try:
                filename = generate_filenames(image_file.filename)
                filePath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                image_file.save(filePath)
                return predict(filename)
            except Exception:
                json_obj = {"image_url": "", "prediction": "后台异常"}
                return json.dumps(json_obj)


def is_allowed_file(filename):
    VALID_EXTENSIONS = ["jpg", "jpeg"]
    is_valid_ext = filename.rsplit(".", 1)[1].lower() in VALID_EXTENSIONS
    return "." in filename and is_valid_ext


def generate_filenames(filename):
    LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    ext = filename.split(".")[-1]
    random_indexes = [random.randint(0, len(LETTERS) - 1) for _ in range(10)]
    random_chars = "".join([LETTERS[index] for index in random_indexes])
    new_name = "{name}.{extension}".format(name=random_chars, extension=ext)
    return secure_filename(new_name)


def predict(filename):
    image_url = url_for("images", filename=filename)
    image_file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    prediction = make_prediction(image_file_path)
    json_obj = {"image_url": image_url, "prediction": prediction}
    return json.dumps(json_obj)


@app.route(f"{base_dir}/images/<filename>", methods=["GET"])
def images(filename):
    return send_file(os.path.join(app.config["UPLOAD_FOLDER"], filename))


@app.errorhandler(500)
def server_error(error):
    return render_template("error.html"), 500


if __name__ == "__main__":
    print("server is running .....")
    app.run("0.0.0.0")
