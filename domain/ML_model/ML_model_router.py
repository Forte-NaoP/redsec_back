from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from starlette import status
from pathlib import Path

import os
import uuid
from typing import List
from subprocess import Popen, PIPE

from database import get_db
from domain.user.user_router import get_current_user
from models import User
from domain.ML_model import ML_model_schema, ML_model_crud

router = APIRouter(
    prefix="/api/model",
)


user_prefix = Path('./users')
model_prefix = "../"


@router.get("/download")
def download_client(db: Session = Depends(get_db),
                    current_user: User = Depends(get_current_user)):

    redsec_client_path = Path('./client.zip')
    if redsec_client_path.is_file():
        return FileResponse(redsec_client_path, media_type="application/zip", filename=redsec_client_path.name)
    else:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found!")


@router.post("/upload", status_code=status.HTTP_204_NO_CONTENT)
async def upload_model(db: Session = Depends(get_db),
                       current_user: User = Depends(get_current_user),
                       name: str = Form(...),
                       files: List[UploadFile] = File(...)):

    username = current_user.username
    user_folder = os.path.join(user_prefix, username)
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)

    exts = list(map(lambda file: os.path.splitext(file.filename)[1], files))
    print(exts)
    if not ('.key' in exts):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="public key not found!")

    path_prefix = f"{uuid.uuid4()}"
    ML_model_crud.save_model(db=db, user=current_user, name=name, path=path_prefix)

    for file in files:
        file_extension = os.path.splitext(file.filename)[1]
        safe_filename = f"{path_prefix}{file_extension}"

        with open(os.path.join(user_folder, safe_filename), "wb") as buffer:
            contents = await file.read()
            buffer.write(contents)


@router.get("/list", response_model=ML_model_schema.ModelList)
def ml_model_list(db: Session = Depends(get_db),
                  current_user: User = Depends(get_current_user),
                  page: int = 0, size: int = 10):
    total, models = ML_model_crud.get_model_list(db=db, user=current_user, skip=page*size, limit=size)
    return {
        'total': total,
        'models': models
    }


@router.post("/inference/{model_uuid}")
async def inference_model(model_uuid: str,
                          db: Session = Depends(get_db),
                          current_user: User = Depends(get_current_user),
                          file: UploadFile = File(...)):

    model = ML_model_crud.get_model_by_uuid(db=db, uuid=model_uuid)
    if model is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not Registered!")

    user_folder = os.path.join(user_prefix, current_user.username)
    model_path = os.path.join(user_folder, model.ML_model_uuid)

    if not os.path.exists(f"{model_path}.dat"):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model file not found!")

    with open(f"{model_path}.img", "wb") as buffer:
        contents = await file.read()
        buffer.write(contents)

    env = os.environ.copy()
    env["model_path"] = f"{model_prefix}{model_path}.dat"
    env["key_path"] = f"{model_prefix}{model_path}.key"
    env["image_path"] = f"{model_prefix}{model_path}.img"
    env["output_path"] = f"{model_prefix}{model_path}.res"

    print(env["model_path"])
    process = Popen(["make", "cpu-encrypt"], stdout=PIPE, stderr=PIPE, env=env, cwd="./redsec")
    stdout, stderr = process.communicate()

    return {"output": "Inference Processing Now!"}


