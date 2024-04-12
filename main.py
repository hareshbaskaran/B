from predict import read_image
from predict import transformacao
from fastapi import FastAPI, File

app = FastAPI()


@app.post("/files/")
async def create_file(file: bytes = File(...)):
    return {"file_size": len(file)}


@app.post("/uploadfile/")
async def create_upload_file(file: bytes = File(...)):
    # read image
    imagem = read_image(file)
    # transform and prediction
    prediction = transformacao(imagem)

    return prediction
