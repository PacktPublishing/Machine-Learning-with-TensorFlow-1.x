import os
import requests
import zipfile
from datetime import datetime
from flask import Flask, render_template, request, redirect, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from client import process_image
from database import db_session, init_db
from dbmodels import Image, Model

app = Flask(__name__)
app.config["UPLOAD_DIR"] = "uploads"
app.config['LABEL_FILE'] = "labels.txt"
app.config["MODEL_DIR"] = "/home/ubuntu/productions/"

init_db()

if not os.path.exists(app.config["UPLOAD_DIR"]):
    os.mkdir(app.config["UPLOAD_DIR"])


def get_labels():
    label_data = [line.strip() for line in open(app.config['LABEL_FILE'], 'r')]
    return label_data


@app.teardown_appcontext
def shutdown_session(exception=None):
    db_session.remove()


@app.route("/uploads/<filename>")
def uploads(filename):
    return send_from_directory(app.config["UPLOAD_DIR"], filename)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/upload_image", methods=["POST"])
def upload_image():
    if 'image' not in request.files:
        return "No file found!"
    image = request.files['image']
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = "%s_%s" % (timestamp, secure_filename(image.filename))
    filepath = os.path.join(app.config['UPLOAD_DIR'], filename)
    image.save(filepath)
    outputs = process_image(filepath, get_labels())

    image = Image(filename)
    image.set_model_output(outputs)
    db_session.add(image)
    db_session.commit()
    return redirect("/results/%s" % image.id)


@app.route("/results/<result_id>", methods=["GET"])
def view_results(result_id):
    image = Image.query.filter(Image.id == result_id).first()
    return render_template("result.html",
                           label_data=get_labels(),
                           model_result=image.get_model_results(),
                           user_label=image.get_user_label(),
                           image_path=image.get_image_path())


@app.route("/results/<result_id>", methods=["POST"])
def set_label(result_id):
    label_index = int(request.form['label'])
    image = Image.query.filter(Image.id == result_id).first()
    image.set_user_label(label_index, get_labels()[label_index])
    db_session.commit()
    return redirect("/results/%s" % image.id)


@app.route("/user-labels", methods=["GET"])
def get_user_labels():
    list_images = Image.query.filter(Image.user_label != "").all()
    list_labels = get_labels()
    outputs = list()
    for image in list_images:
        outputs.append({
            "id": image.id,
            "url": image.get_image_path(),
            "label": list_labels.index(image.get_user_label()),
            "name": image.get_user_label()
        })
    return jsonify(outputs)


def download_file(link):
    response = requests.get(link)
    file_name = link.split("/")[-1]
    temp_file_path = "/tmp/%s" % file_name
    with open(temp_file_path, 'wb') as f:
        for chunk in response:
            f.write(chunk)
    return temp_file_path


@app.route("/model", methods=["POST"])
def upgrade_model():
    if "link" in request.form:
        link = request.form["link"]
    else:
        link = ""
    version = request.form["version"]
    ckpt_name = request.form["ckpt_name"]
    name = request.form["name"]

    model = Model(version=version, link=link, ckpt_name=ckpt_name, name=name)
    db_session.add(model)
    db_session.commit()

    if len(link) > 0:
        print("Start downloading", link)
        file_path = download_file(link)
        extract_path = os.path.join(app.config["MODEL_DIR"], version)
        print("Downloaded file at", file_path)
        with zipfile.ZipFile(file_path, 'r') as z:
            z.extractall(extract_path)
        print("Extracted at", extract_path)
        os.remove(file_path)

    return jsonify(model.export_data())


@app.route("/model", methods=["GET"])
def get_model():
    model = Model.query.order_by("-id").first()
    return jsonify(model.export_data())


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, threaded=True)
