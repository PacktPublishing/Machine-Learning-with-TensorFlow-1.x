import json
import os
from flask import url_for
from sqlalchemy import Column, Integer, String
from database import Base


class Image(Base):
    __tablename__ = "images"
    id = Column(Integer, primary_key=True)
    image_path = Column(String(128), unique=True)
    model_result = Column(String(256), default="")
    user_label = Column(String(128), default="")

    def __init__(self, image_path, model_result="", user_label=""):
        self.image_path = image_path
        self.model_result = model_result
        self.user_label = user_label

    def get_model_results(self):
        return json.loads(self.model_result)

    def get_image_path(self):
        return url_for('uploads', filename=self.image_path)

    def get_user_label(self):
        if len(self.user_label) > 0:
           return self.user_label.split(" ")[-1]
        return ""

    def set_model_output(self, outputs):
        self.model_result = json.dumps([obj.__dict__ for obj in outputs])

    def set_user_label(self, label_idx, label_name):
        self.user_label = "%s %s" % (label_idx, label_name)

    def __repr__(self):
        if self.id:
            return "<Image %d>: %s %s %s" % (self.id, self.image_path, self.model_result, self.user_label)
        else:
            return "<Image None>: %s %s %s" % (self.image_path, self.model_result, self.user_label)


class Model(Base):
    __tablename__ = "models"
    id = Column(Integer, primary_key=True)
    version = Column(Integer)
    link = Column(String(128), default="")
    name = Column(String(128), default="")
    ckpt_name = Column(String(128), default="")

    def __init__(self, version, link, name, ckpt_name):
        self.version = version
        self.link = link
        self.name = name
        self.ckpt_name = ckpt_name

    def __repr__(self):
        if self.id:
            return "<Model %d>: %d %s" % (self.id, self.version, self.ckpt_name)
        else:
            return "<Model None>: %d %s" % (self.version, self.ckpt_name)

    def export_data(self):

        return {
            "id": self.id,
            "version": self.version,
            "name": self.name,
            "ckpt_name": self.ckpt_name,
            "link": self.link
        }
