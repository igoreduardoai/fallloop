# DEPLOY.md - FallLoop on AWS. Status 2026-10-24: DEPLOYED in us-east-2 (stack `fallloop-demo`)

Live dashboard (judging period): https://shuwf5nbecmvl633bee4crfkie0nziur.lambda-url.us-east-2.on.aws/
Target: AWS Free plan only, USD 1 monthly budget alert, full teardown after judging. No Docker needed: the processing
Lambda is a python3.12 **zip** package for arm64 (Graviton) built from manylinux aarch64 wheels.

## 1. Build the arm64 package (any Linux/macOS machine, ~1 minute)
```bash
B=.aws-build/process; mkdir -p $B
pip install --target $B --platform manylinux_2_28_aarch64 --platform manylinux2014_aarch64 \
  --implementation cp --python-version 3.12 --only-binary=:all: opencv-python-headless==5.0.0.93 numpy==2.5.4
cp -r fallloop run_pipeline.py aws/handler.py $B/ && mkdir -p $B/models && cp models/yolo26n-pose.onnx $B/models/
```
(`./tools/get_model.sh` creates `models/yolo26n-pose.onnx` first; the weights are not in this repository.)

## 2. Deploy (AWS CLI v2, region us-east-2)
```bash
ART=<your-artifacts-bucket>   # private S3 bucket for the packaged code
aws cloudformation package --template-file aws/template.yaml --s3-bucket $ART --s3-prefix fallloop \
  --output-template-file .aws-build/packaged.yaml --region us-east-2
aws cloudformation deploy --template-file .aws-build/packaged.yaml --stack-name fallloop-demo --region us-east-2 \
  --capabilities CAPABILITY_IAM CAPABILITY_AUTO_EXPAND \
  --parameter-overrides BudgetEmail=<email> BudgetLimitUSD=1 RetentionDays=7
```
`sam build && sam deploy` with `aws/samconfig.toml` also works. `aws/Dockerfile` is an optional container-image variant (not used for the live deploy).
Outputs: `DashboardUrl`, `InBucketName`, `OutBucketName`.

## 3. Run the demo in the cloud
```bash
aws s3 cp demo/demo_input.mp4 s3://<InBucketName>/uploads/demo_input.mp4 --region us-east-2   # triggers the Graviton Lambda
aws logs tail /aws/lambda/<ProcessFunction physical name> --since 10m --region us-east-2
```
Open `DashboardUrl`; the newest run loads automatically (`?run=<id>` for a specific one).
Measured run (2026-10-24): `results/aws-graviton-2026-10-24/` (incident.json + CloudWatch log with the REPORT line).

## 4. Cost guardrails
- USD 1/month COST budget (`fallloop-demo-usd-budget`), emails at 1% actual, 100% actual and 100% forecast. A budget only alerts; run teardown if any alert fires.
- One demo run = ~115 s x 2 GB = ~230 GB-s, far inside Lambda's always-free 400,000 GB-s/month. S3 objects expire after 7 days.
- No NAT gateway, EC2, load balancer, CloudFront or Marketplace items (COOL is a paid Marketplace AMI and is not used).

## 5. Teardown (after winners are announced, or immediately if any budget alert fires)
```bash
bash aws/teardown.sh fallloop-demo us-east-2
```
Then delete the artifacts bucket and the two `/aws/lambda/fallloop-demo-*` log groups, and check Billing the next day.
