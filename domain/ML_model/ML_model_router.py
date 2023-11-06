from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from starlette import status
from pathlib import Path
from pathvalidate import sanitize_filename, sanitize_filepath


import os
import uuid
from typing import List, Dict
from subprocess import Popen, PIPE
import shutil

from database import get_db
from domain.user.user_router import get_current_user
from models import User
from domain.ML_model import ML_model_schema, ML_model_crud


router = APIRouter(
    prefix="/api/model",
)


user_prefix = Path('./users')
model_prefix = "../"
model_keys = ["model", "weight", "ckks_parms", "galois_key", "relin_key", "pub_key"]
model_files: dict = dict.fromkeys(model_keys)

@router.get("/download")
def download_client(db: Session = Depends(get_db),
                    current_user: User = Depends(get_current_user)):
    
    client_path = Path('./client/client.zip')
    headers = {"Content-Disposition": f"attachment; filename={client_path.name}"}
    if client_path.is_file():
        return FileResponse(client_path, media_type="application/zip", headers=headers)
    else:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found!")

@router.get("/list", response_model=ML_model_schema.ModelList)
def ml_model_list(db: Session = Depends(get_db),
                  current_user: User = Depends(get_current_user),
                  page: int = 0, size: int = 10):
    total, models = ML_model_crud.get_model_list(db=db, user=current_user, skip=page*size, limit=size)
    return {
        'total': total,
        'models': models
    }

@router.post("/upload", status_code=status.HTTP_204_NO_CONTENT)
async def upload_model(db: Session = Depends(get_db),
                       current_user: User = Depends(get_current_user),
                       name: str = Form(...),
                       files: ML_model_schema.FileUpload = Depends(ML_model_schema.parse_file_upload)):

    if model_files.keys() != files.model_dump().keys():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid file type!")

    username = current_user.username
    user_folder = os.path.join(user_prefix, username)
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)

    path_prefix = f"{uuid.uuid4()}"
    ML_model_crud.save_model(db=db, user=current_user, name=name, path=path_prefix)

    model_dir_path = os.path.join(user_folder, path_prefix)
    if not os.path.exists(model_dir_path):
        os.makedirs(model_dir_path)

    for file_type, file in files:
        contents = await file.read()
        with open(os.path.join(model_dir_path, f"{file_type}"), "wb") as buffer:
            buffer.write(contents)

    return

@router.post("/{model_uuid}/inference")
async def inference_model(model_uuid: str,
                          db: Session = Depends(get_db),
                          current_user: User = Depends(get_current_user),
                          name : str = Form(...),
                          file: UploadFile = File(...)):

    model = ML_model_crud.get_model_by_uuid(db=db, uuid=model_uuid)
    if model is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not Registered!")

    user_folder = os.path.join(user_prefix, current_user.username)
    model_path = os.path.join(user_folder, model.ML_model_uuid)

    if not os.path.isdir(model_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model file not found!")
    
    image_dir = os.path.join(model_path, 'images')
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    
    safe_filename = f"{uuid.uuid4()}"

    with open(os.path.join(image_dir, safe_filename), "wb") as buffer:
        contents = await file.read()
        buffer.write(contents)

    ML_model_crud.save_file(db=db, model=model, description=name, file_name=safe_filename)

    print(model_path)
    
    return

    # if not os.path.exists(f"{model_path}.dat"):
    #     raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model file not found!")

    # with open(f"{model_path}.img", "wb") as buffer:
    #     contents = await file.read()
    #     buffer.write(contents)

    # env = os.environ.copy()
    # env["model_path"] = f"{model_prefix}{model_path}.dat"
    # env["key_path"] = f"{model_prefix}{model_path}.key"
    # env["image_path"] = f"{model_prefix}{model_path}.img"
    # env["output_path"] = f"{model_prefix}{model_path}.res"

    # print(env["model_path"])
    # process = Popen(["make", "cpu-encrypt"], stdout=PIPE, stderr=PIPE, env=env, cwd="./redsec")
    # stdout, stderr = process.communicate()

    # time_elapsed = list(filter(lambda x: x.startswith("Inference"), stdout.decode().split("\n")))
    # print(time_elapsed)
    # return {"result": time_elapsed}


@router.get("/{model_uuid}/history", response_model=ML_model_schema.ModelHistoryList)
async def get_model_history(model_uuid: str,
                            db: Session = Depends(get_db),
                            current_user: User = Depends(get_current_user)):

    model = ML_model_crud.get_model_by_uuid(db=db, uuid=model_uuid)
    if model is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not Registered!")

    total, history = ML_model_crud.get_model_history(db=db, model=model)
    return {
        'total': total,
        'history': history
    }


@router.get("/{model_uuid}/{file_name}/download")
def download_inference_data(model_uuid: str, file_name: str,
                            db: Session = Depends(get_db),
                            current_user: User = Depends(get_current_user)):
    
    file_path = Path(f"./users/{current_user.username}/{model_uuid}/images/{file_name}")
    print('asd')
    if file_path.is_file():
        return FileResponse(file_path, media_type="application/zip", filename=file_path.name)
    else:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found!")


@router.delete("/{model_uuid}/delete")
def delete_model(model_uuid: str,
                 db: Session = Depends(get_db),
                 current_user: User = Depends(get_current_user)):

    model = ML_model_crud.get_model_by_uuid(db=db, uuid=model_uuid)
    if model is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not Registered!")

    user_folder = os.path.join(user_prefix, current_user.username)
    model_path = os.path.join(user_folder, model.ML_model_uuid)

    if os.path.isdir(model_path):
        shutil.rmtree(model_path)
        ML_model_crud.delete_model(db=db, model=model)
        return {"result": "삭제되었습니다."}
    else:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model file not found!")