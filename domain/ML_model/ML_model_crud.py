from domain.ML_model.ML_model_schema import ModelCreate
from models import Model, User
from sqlalchemy.orm import Session


def save_model(db: Session, user: User, name: str, path: str):
    db_model = Model(
        ML_model_name=name,
        ML_model_uuid=path,
        user=user)

    db.add(db_model)
    db.commit()


def get_model_list(db: Session, user: User, skip: int = 0, limit: int = 10):
    total = db.query(Model).filter(Model.user_id == user.id).count()
    models = db.query(Model)\
        .order_by(Model.id.desc()).filter(Model.user_id == user.id).offset(skip).limit(limit).all()
    return total, models


def get_model_by_uuid(db: Session, uuid: str):
    return db.query(Model).filter(Model.ML_model_uuid == uuid).first()

