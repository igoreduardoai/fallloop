# FallLoop processing function: Lambda container image for arm64 (Graviton). Build context = repository root. Optional: the live deploy uses the zip package (DEPLOY.md).
FROM public.ecr.aws/lambda/python:3.12-arm64
COPY aws/requirements-lambda.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements-lambda.txt
COPY fallloop/ ${LAMBDA_TASK_ROOT}/fallloop/
COPY run_pipeline.py ${LAMBDA_TASK_ROOT}/
COPY models/yolo26n-pose.onnx ${LAMBDA_TASK_ROOT}/models/
COPY aws/handler.py ${LAMBDA_TASK_ROOT}/
ENV CV_THREADS=2 PROC_FPS=10
CMD ["handler.lambda_handler"]
