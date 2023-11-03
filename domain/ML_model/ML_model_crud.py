from domain.ML_model.ML_model_schema import ModelCreate
from models import Model, User, History
from sqlalchemy.orm import Session


def save_model(db: Session, user: User, name: str, path: str):
    db_model = Model(
        ML_model_name=name,
        ML_model_uuid=path,
        user=user)

    db.add(db_model)
    db.commit()


def delete_model(db: Session, model: Model):
    db.delete(model)
    db.commit()


def get_model_list(db: Session, user: User, skip: int = 0, limit: int = 10):
    total = db.query(Model).filter(Model.user_id == user.id).count()
    models = db.query(Model)\
        .order_by(Model.id.desc()).filter(Model.user_id == user.id).offset(skip).all()
    return total, models


def get_model_by_uuid(db: Session, uuid: str) -> Model:
    return db.query(Model).filter(Model.ML_model_uuid == uuid).first()


def get_model_history(db: Session, model: Model):
    total = db.query(History).filter(History.model_id == model.id).count()
    histories = db.query(History)\
        .order_by(History.id.desc()).filter(History.model_id == model.id).all()
    
    return total, histories


def save_file(db: Session, model: Model, description: str, file_name: str):
    db_history = History(
        model=model,
        description=description,
        file_name=file_name,
        pending=True
    )
    
    db.add(db_history)
    db.commit()
