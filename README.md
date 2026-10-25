
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
学習処理はモデルをMLflowの実験 `iris-classification-proxy` に記録します。MLflowサーバーが成果物を代理保存・配信するため、学習コンテナやBentoMLと同じファイルシステムを共有する必要はありません。Model Registryへの登録は自動では行わず、学習後にMLflow UIからモデルを登録し、`champion` エイリアスを設定してください。BentoMLは起動時にはモデルを読み込まず、最初の予測時にRegistryから取得します。正しいモデルを登録して `champion` を設定するまでは予測できません。

MLflowのartifact proxy設定を反映するには、プロジェクトディレクトリでコンテナを再ビルドします。

```sh
docker compose up -d --build mlflow worker bentoml api
```

以前の `iris-classification` 実験はローカルパスを成果物の保存先として記録しているため、proxy経由では読み込めません。旧実験は保持され、新しい学習は `iris-classification-proxy` に記録されます。

<img src="https://github.com/minegishirei/store/blob/main/sre/mlops/handson/1/image.png?raw=true" />

<img src="https://github.com/minegishirei/store/blob/main/sre/mlops/handson/2/image.png?raw=true" />

学習時のパラメーター・精度・モデルはMLflowに記録されます。

<img src="https://github.com/minegishirei/store/blob/main/sre/mlops/handson/4/image.png?raw=true" />

初回の学習がPrefect UIで完了した後、
`Models`タブから `Create Model` で`iris-classifier`という名前で作成します。

<img src="https://github.com/minegishirei/store/blob/main/sre/mlops/handson/models/image.png?raw=true" />

`Experiments` タブから生成してあるモデルを選択し、 `register`からモデルを紐づけます。

<img src="https://github.com/minegishirei/store/blob/main/sre/mlops/handson/connect/image.png?raw=true" />

再び `Models`タブから 該当のモデルに対して `main` エイリアスを与えて完成です。

<img src="https://github.com/minegishirei/store/blob/main/sre/mlops/handson/version/image.png?raw=true" />

サーバーを再起動し
Flaskサンプルアプリ(http://localhost:8000)から動作チェックをしましょう。


<img src="https://github.com/minegishirei/store/blob/main/sre/mlops/handson/3/image.png?raw=true" />




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

1. 保存： 学習済みのモデルをMLflowに記録し、Model Registryに登録する。
1. 定義： `service.py` などのファイルで、入力データの受け取りやモデル推論の前処理・後処理（RunnerとServiceの概念）をPythonコードで定義する。
1. ビルド： 依存関係を含めてひとつの「Bento」としてアーカイブ化する。
1. 実行・デプロイ： ローカルでAPIサーバーとして起動してテストしたり、Dockerコンテナ化して本番環境にデプロイする。

今回の `service.py`はMLflow Model Registryの `Production` モデルを読み込みます。

```python
import os
from functools import cached_property

import bentoml
import mlflow.sklearn


@bentoml.service(name="iris-classifier")
class IrisClassifier:
    def __init__(self) -> None:
        self.model_uri = os.getenv(
            "MLFLOW_MODEL_URI", "models:/iris-classifier@champion"
        )

    @cached_property
    def model(self):
        return mlflow.sklearn.load_model(self.model_uri)

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

この構成ではMLflow Tracking Serverからモデルをロードします。別の構成では、起動時にS3からモデルをロードする方法もあります。

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
  - 学習の開始。MLflow UIで手動登録・選択したProductionモデルを使って推論します。


### MLflow UI

実験管理とモデルレジストリの機能を持つのがMLflowです。
機械学習のトレーニングを実行した際のパラメータや精度をテーブル形式やグラフ形式で比較検討ができます。

<img src="https://github.com/minegishirei/store/blob/main/sre/mlops/handson/5/image.png?raw=true" />

また、作成したモデルにラベリング/エイリアスを行うことで
動作環境のモデルの切り替えも手軽に行えます。
(先ほどの`BentoML` のソースコードは MLflow のURLを呼びだしていることを思い出してください。)


- http://localhost:5000
- できること:
  - `iris-classification-proxy` 実験の学習、パラメーター、精度、モデル成果物を確認・比較する。
  - 学習済みモデルをModel Registryに登録し、バージョンに `champion` エイリアスを設定する。
  - 手順: `iris-classification-proxy` 実験を開き、学習Runの `Logged Models` にある `iris_classifier` を開いて `Register model` を選びます。既存の `iris-classifier` に新しいバージョンとして登録し、Registryの `iris-classifier` でその新バージョンに `champion` エイリアスを割り当てます。Runの `Artifacts` に表示されるファイルパスから手動で登録せず、必ずLogged Modelから登録してください。
- `champion` を設定した後に予測APIを呼び出してください。すでに別モデルを読み込んだBentoMLに新しいバージョンを反映する場合は、BentoMLを再起動します。


### Prefect UI

- できること: `iris-training` Deploymentの起動、`n_estimators`・`test_size`・`random_state` の指定、実行状態とログの確認。
- できないこと: MLflowの実験比較、Model Registryへの登録やProductionバージョンの選択。これらはMLflow UIで行います。

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
