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
./start.sh
```

`worker` は `iris-process` Work Poolを用意し、`iris-training` Deploymentを登録して待機します。Deployment登録後、Prefect UIのDeploymentsからRunを作成し、`n_estimators`、`test_size`、`random_state` を指定して学習できます。学習時のパラメーター・精度・モデルはMLflowに記録されます。

初回の学習がPrefect UIで完了した後、BentoMLサーバーとFlask APIを起動します。BentoMLは起動時にモデルを読み込みます。

```bash
docker compose up -d --build bentoml api
```

2回目以降の学習後は、BentoMLを再起動して更新モデルを読み込ませます。

```bash
docker compose restart bentoml
```

## アクセス

- Flaskブラウザ画面: http://localhost:8000
- Flask JSON API: http://localhost:8000/api/predict
- BentoML API docs: http://localhost:3000/docs
- MLflow UI: http://localhost:5000
- Prefect UI: http://localhost:4200

## 各画面でできること・できないこと

### Flaskブラウザ画面

- メイン画面
  - http://localhost:8000
  - できること: Irisの4つの測定値を入力し、現在サービング中のモデルでクラス予測を実行する。
- json api
  - http://localhost:8000/api/predict
  - できること: `POST /api/predict` に4つの数値特徴量を送信し、クラスIDとクラス名をJSONで取得する。入力形式を検証し、形式不正は `400`、BentoML停止中は `503` を返します。
  - できないこと: 学習の開始、パラメーター変更、モデルの選択・切り替え。

```json
{"features": [[5.1, 3.5, 1.4, 0.2]]}
```

### BentoML API docs

- できること:
  - BentoMLの `/predict` APIを直接試し、入力と応答の形式を確認する。
  - かつては、機械学習済みのモデルを pickle形式で 読み込み実行することもあったが、BentoMLではその代替手段であり、より高速な推論が可能。
- できないこと:
  - 学習の開始やモデルの再登録。起動時に `iris_classifier:latest` を読み込んで推論します。


### MLflow UI

- http://localhost:5000
- できること:
  - `iris-classification` の実験、学習、記録されたパラメーター・精度・モデル成果物を確認・比較する。
- できないこと:
  - この構成ではUIから学習を開始したり、モデルを本番へ昇格したりすること。学習はCLIから実行します。


### Prefect UI

- できること: `iris-training` Deploymentの起動、`n_estimators`・`test_size`・`random_state` の指定、実行状態とログの確認。
- できないこと: MLflowの実験比較やモデル成果物の詳細確認。これらはMLflow UIで行います。

Deployment/Workerを使わず、従来どおりCLIから直接学習する場合:

```bash
docker compose --profile training run --build --rm train
```

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