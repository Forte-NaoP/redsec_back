from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import relationship, backref

from database import Base


class Question(Base):
    __tablename__ = "question"

    id = Column(Integer, primary_key=True)
    subject = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    create_date = Column(DateTime, nullable=False)
    user_id = Column(Integer, ForeignKey("user.id"), nullable=True)
    user = relationship("User", backref="question_users")
    modify_date = Column(DateTime, nullable=True)
    answers = relationship("Answer", back_populates="question", cascade="all, delete, delete-orphan")


class Answer(Base):
    __tablename__ = "answer"

    id = Column(Integer, primary_key=True)
    content = Column(Text, nullable=False)
    create_date = Column(DateTime, nullable=False)
    question_id = Column(Integer, ForeignKey("question.id"))
    question = relationship("Question", back_populates="answers")
    user_id = Column(Integer, ForeignKey("user.id"), nullable=True)
    user = relationship("User", backref="answer_users")
    modify_date = Column(DateTime, nullable=True)


class User(Base):
    __tablename__ = "user"

    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    user_models = relationship("Model", back_populates="user", cascade="all, delete, delete-orphan")


class Model(Base):
    __tablename__ = "model"

    id = Column(Integer, primary_key=True)
    ML_model_name = Column(String, nullable=False)
    ML_model_uuid = Column(String, unique=True, nullable=False)
    user_id = Column(Integer, ForeignKey("user.id"), nullable=False)
    user = relationship("User", back_populates="user_models")
    model_history = relationship("History", back_populates="model", cascade="all, delete, delete-orphan")


class History(Base):
    __tablename__ = "history"

    id = Column(Integer, primary_key=True)
    model_id = Column(Integer, ForeignKey("model.id"), nullable=False)
    model = relationship("Model", back_populates="model_history")
    description = Column(String, nullable=True)
    file_name = Column(String, nullable=True)
    pending = Column(Boolean, nullable=False, default=True)
