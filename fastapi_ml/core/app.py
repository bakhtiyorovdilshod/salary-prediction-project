import os
from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title='Salary prediction', version='1.0.0')

import pandas as pd

router = APIRouter(
    prefix='/',
    tags=['Salary prediction']
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@router.post(
    "/salary_prediction/",
    response_model=ScoreResponse
)
async def family_weights(data: UserPinfl):
    BASE_DIR = os.getcwd()
    file = f'{BASE_DIR}/trained_models/family_score_model.pkl'
    # some async operation could happen here
    rejected_job_offers = await integrator.get_rejected_job_offers(pinfl=data.pinfl)
    columns = [
        'PINFL',
        '6 OY ICHIDA MUJORAT QILGANLIK',
        'HISOBDA TURGANLIK',
        'RAD QILGANLIK',
        'UZINI BAND QILGANLIK',
        'JAMOAT ISHCHISI'
    ]
    rows = np.array([rejected_job_offers])
    data = pd.DataFrame(rows, columns=columns)
    with open(file, "rb") as f:
        model = pickle.load(open(file, "rb"))
    prediction = model.predict(data)
    return {'score': prediction}