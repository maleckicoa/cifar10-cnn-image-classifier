import os
import tempfile
from zipfile import ZipFile, ZIP_DEFLATED

from fastapi import BackgroundTasks, FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse

from predict import predict_image

app = FastAPI(title="CIFAR-10")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    if file.content_type not in ["image/png", "image/jpeg"]:
        raise HTTPException(status_code=400, detail="Only PNG or JPEG images are supported")

    try:
        contents = await file.read()
        result = predict_image(contents)
        return result
    except Exception as e:
        return {"error": str(e), "status": "error"}


@app.get("/download-test-images")
async def download_test_images(background_tasks: BackgroundTasks):
    try:
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp_file:
            zip_path = tmp_file.name

        test_images_path = os.path.join(os.path.dirname(__file__), 'test_images')

        with ZipFile(zip_path, mode='w', compression=ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(test_images_path):
                for file_name in files:
                    file_path = os.path.join(root, file_name)
                    arcname = os.path.relpath(file_path, test_images_path)
                    zipf.write(file_path, arcname=arcname)

        background_tasks.add_task(os.unlink, zip_path)

        return FileResponse(
            path=zip_path,
            filename="test-images.zip",
            media_type="application/zip"
        )
    except Exception as e:
        return {"error": str(e), "status": "error"}
