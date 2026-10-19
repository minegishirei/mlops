# MLflow + Prefect + BentoML

Iris分類の学習をPrefectで実行し、MLflowへ実験を記録、BentoMLでモデルを保存・サービングし、Flaskの画面/APIから推論するサンプルです。各コンテナは専用ディレクトリのDockerfileと依存関係で独立してビルドします。永続データはすべて `data/` 以下に保存されます。

## ディレクトリ

```text
mlops/
├── mlflow/       # MLflow Tracking Server
├── prefect/      # Prefect Server
├── train/        # 学習Flowと専用依存関係
├── bentoml/      # BentoMLサーバーと専用依存関係
├── api/          # Flask UI/API
├── data/         # MLflow、Prefect、BentoMLの永続データ
└── docker-compose.yaml
```

## 起動

Docker Desktopを起動し、プロジェクトディレクトリから実行します。

```bash
docker compose up -d mlflow prefect
docker compose --profile training run --build --rm train
docker compose up -d --build bentoml api
```

学習コマンドはRandomForestを訓練し、MLflowに精度・パラメーター・モデルを記録し、BentoMLのモデルストアへ保存します。学習完了後、BentoMLサーバーとFlask APIを起動してください。

## アクセス

- Flaskブラウザ画面: http://localhost:8000
- Flask JSON API: http://localhost:8000/api/predict
- BentoML API docs: http://localhost:3000/docs
- MLflow UI: http://localhost:5000
- Prefect UI: http://localhost:4200

予測リクエスト:

```bash
curl -X POST http://localhost:8000/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"features": [[5.1, 3.5, 1.4, 0.2]]}'
```

## テスト

```bash
python -m pip install -r api/requirements.txt -r train/requirements.txt pytest
python -m pytest
```

## 再学習と停止

```bash
docker compose --profile training run --build --rm train
docker compose restart bentoml api
docker compose down
```

コンテナを停止しても `data/` の内容は残ります。実験履歴・モデルを含めて初期化する場合は、`data/` 内のデータを削除してください。