import pandas as pd
from huggingface_hub import HfApi, login

# API 토큰을 사용하여 로그인
login("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")

# 데이터 로드
data = pd.read_csv("result/submission_0708-2.csv")

# CSV 파일로 저장
csv_file_path = "submission_0708-2.csv"
data.to_csv(csv_file_path, index=False)

# HfApi 객체 생성
api = HfApi()

# 사용자명과 저장소 이름
USERNAME = "sonhy02"
REPO_NAME = "INHA-DACON"

# CSV 파일 업로드
api.upload_file(
    path_or_fileobj=csv_file_path,
    path_in_repo=csv_file_path,
    repo_id=f"{USERNAME}/{REPO_NAME}",
    repo_type="dataset"
)

print("파일 업로드 완료")
