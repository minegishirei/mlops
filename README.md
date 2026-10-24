
# MLflow + Perfect + BentoMLを用いたMLOpsサンプルアプリ


Iris分類の学習をPrefectで実行し、MLflowへ実験を記録、BentoMLでモデルを保存・サービングし、Flaskの画面/APIから推論するサンプルです。各
コンテナは専用ディレクトリのDockerfileと依存関係で独立してビルドします。


まずはソースコードを入手し、アプリケーションを起動させます。

```sh
git clone https://github.com/minegishirei/mlops.git
cd mlops
./start.sh
```

Docker Desktopを起動し、プロジェクトディレクトリから実行します。

アプリが起動できたら、Perfectの画面 (http://localhost:4200) にアクセスし、Deploymentに移動します。
Prefect UIのDeploymentsからRunを作成し、`n_estimators`、`test_size`、`random_state` を指定して学習できます。
Runボタンを押すと機械学習モデルを生成するパイプラインが起動します。

<img src="https://github.com/minegishirei/store/blob/main/sre/mlops/handson/1/image.png?raw=true" />

<img src="https://github.com/minegishirei/store/blob/main/sre/mlops/handson/2/image.png?raw=true" />

学習時のパラメーター・精度・モデルはMLflowに記録されます。

<img src="https://github.com/minegishirei/store/blob/main/sre/mlops/handson/4/image.png?raw=true" />

初回の学習がPrefect UIで完了した後、
Flaskサンプルアプリ(http://localhost:8000)から動作チェックをしましょう。


<img src="https://github.com/minegishirei/store/blob/main/sre/mlops/handson/3/image.png?raw=true" />


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


## アクセス

- Flaskブラウザ画面: http://localhost:8000
- Flask JSON API: http://localhost:8000/api/predict
- BentoML API docs: http://localhost:3000/docs
- MLflow UI: http://localhost:5000
- Prefect UI: http://localhost:4200

## 各サービスでできること・できないこと

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

PyTorch、TensorFlow、Scikit-Learn、XGBoost、LLM（Hugging Faceなど）といった様々なフレームワークで訓練したモデルを、Bentoと呼ばれる標準化されたコンテナ形式で起動させます。
高性能なREST APIやgRPCのサーバーとしてアクセスできます。

1. 保存： 学習済みのモデルをBentoMLのローカルストアに保存する。
1. 定義： `service.py` などのファイルで、入力データの受け取りやモデル推論の前処理・後処理（RunnerとServiceの概念）をPythonコードで定義する。
1. ビルド： 依存関係を含めてひとつの「Bento」としてアーカイブ化する。
1. 実行・デプロイ： ローカルでAPIサーバーとして起動してテストしたり、Dockerコンテナ化して本番環境にデプロイする。

今回の `service.py`は以下の通りで、

```python
import os
import bentoml


@bentoml.service(name="iris-classifier")
class IrisClassifier:
    def __init__(self) -> None:
        model_tag = os.getenv("BENTOML_MODEL_TAG", "iris_classifier:latest")
        self.model = bentoml.sklearn.load_model(model_tag)

    @bentoml.api
    def predict(self, features: list[list[float]]) -> dict[str, list]:
        predictions = self.model.predict(features).tolist()
        class_names = ["setosa", "versicolor", "virginica"]
        return {
            "predictions": predictions,
            "class_names": [class_names[index] for index in predictions],
        }


svc = IrisClassifier
```

上記のケースはローカル環境での動作のため`./data/bentoml` を経由していますが、
実戦では起動時にS3からロードするなどのてがあります。

```python
import os
import boto3
import bentoml
import joblib

@bentoml.service(name="iris-classifier")
class IrisClassifier:
    def __init__(self) -> None:
        # S3からモデルをダウンロードする処理
        s3_bucket = os.getenv("S3_BUCKET", "my-bucket")
        s3_key = os.getenv("S3_KEY", "models/iris.pkl")
        local_model_path = "/tmp/iris.pkl"

        if not os.path.exists(local_model_path):
            s3 = boto3.client("s3")
            s3.download_file(s3_bucket, s3_key, local_model_path)

        # 読み込み
        self.model = joblib.load(local_model_path)

    @bentoml.api
    def predict(self, features: list[list[float]]) -> dict[str, list]:
        predictions = self.model.predict(features).tolist()
        class_names = ["setosa", "versicolor", "virginica"]
        return {
            "predictions": predictions,
            "class_names": [class_names[index] for index in predictions],
        }

svc = IrisClassifier
```

これを動かしているコンテナに対して、以下のようにリクエストを投げれます。

```sh
curl -X 'POST' \
  'http://localhost:3000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '[[5.1, 3.5, 1.4, 0.2]]'
```

- できること:
  - BentoMLの `/predict` APIを直接試し、入力と応答の形式を確認する。
  - かつては、機械学習済みのモデルを pickle形式で 読み込み実行することもあったが、BentoMLではその代替手段であり、より高速な推論が可能。
- できないこと:
  - 学習の開始やモデルの再登録。起動時に `iris_classifier:latest` を読み込んで推論します。


### MLflow UI

実験管理とモデルレジストリの機能を持つのがMLflowです。
機械学習のトレーニングを実行した際のパラメータや精度をテーブル形式やグラフ形式で比較検討ができます。

<img src="https://github.com/minegishirei/store/blob/main/sre/mlops/handson/5/image.png?raw=true" />


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



