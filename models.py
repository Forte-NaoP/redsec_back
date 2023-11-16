from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import relationship, backref

from database import Base


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
