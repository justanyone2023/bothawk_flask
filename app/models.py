#-*- coding=utf-8 -*-
from . import db


class Role(db.Model):
    __tablename__ = 'roles'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64), unique=True)
    users = db.relationship('User', backref='role', lazy='dynamic')

    def __repr__(self):
        return '<Role %r>' % self.name


class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, index=True)
    role_id = db.Column(db.Integer, db.ForeignKey('roles.id'))

    def __repr__(self):
        return '<User %r>' % self.username


class TaskStatus(db.Model):
    __tablename__ = 'task_status'
    id = db.Column(db.String(length=255), primary_key=True)
    status = db.Column(db.String(length=255), nullable=False)
    user = db.Column(db.String(length=5000), nullable=False)
    result = db.Column(db.String(length=255), nullable=True)
    error = db.Column(db.String(length=255), nullable=True)
    progress = db.Column(db.Integer, nullable=False)
    # 自动设置当前时间
    datetime = db.Column(db.DateTime, server_default=db.func.now())

