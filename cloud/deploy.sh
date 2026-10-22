#!/usr/bin/env bash
# Deploy the LUMEN analysis function to AWS Lambda.
# Run inside AWS CloudShell (it already has the AWS CLI, Python and pip):
#   unzip lumen-cloud.zip -d lumen-cloud && cd lumen-cloud && bash deploy.sh
# Safe to re-run: it updates the existing function instead of failing.
set -euo pipefail

FN=lumen-analyze
ROLE=lumen-lambda-role
REGION=${AWS_REGION:-$(aws configure get region || echo eu-north-1)}
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
BUCKET=lumen-deploy-${ACCOUNT}-${REGION}
echo "Account $ACCOUNT, region $REGION"

echo "== 1/5 Building package (OpenCV 5 + NumPy for Lambda Python 3.12) =="
rm -rf build && mkdir -p build
pip3 install -q --target build --platform manylinux2014_x86_64 --implementation cp \
    --python-version 3.12 --only-binary=:all: "opencv-python-headless==5.0.0.93" "numpy<3"
cp -r handler.py lumen build/
find build -name "__pycache__" -type d -prune -exec rm -rf {} +
(cd build && zip -qr9 ../function.zip .)
ls -lh function.zip

echo "== 2/5 Uploading to S3 (the package is too big for a direct upload) =="
aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null || {
  if [ "$REGION" = "us-east-1" ]; then aws s3api create-bucket --bucket "$BUCKET" >/dev/null
  else aws s3api create-bucket --bucket "$BUCKET" --create-bucket-configuration LocationConstraint="$REGION" >/dev/null; fi
  aws s3api put-public-access-block --bucket "$BUCKET" --public-access-block-configuration \
    BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
}
aws s3 cp function.zip "s3://$BUCKET/function.zip" --only-show-errors

echo "== 3/5 Permissions role (logs only) =="
if ! aws iam get-role --role-name "$ROLE" >/dev/null 2>&1; then
  aws iam create-role --role-name "$ROLE" --assume-role-policy-document '{
    "Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}' >/dev/null
  aws iam attach-role-policy --role-name "$ROLE" \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
  echo "waiting for the new role to become usable..."; sleep 15
fi
ROLE_ARN=$(aws iam get-role --role-name "$ROLE" --query Role.Arn --output text)

echo "== 4/5 Lambda function =="
KEY_FILE=~/lumen_key.txt
[ -f "$KEY_FILE" ] || python3 -c "import secrets;print(secrets.token_urlsafe(24))" > "$KEY_FILE"
KEY=$(cat "$KEY_FILE")
if aws lambda get-function --function-name "$FN" >/dev/null 2>&1; then
  aws lambda update-function-code --function-name "$FN" --s3-bucket "$BUCKET" --s3-key function.zip >/dev/null
  aws lambda wait function-updated --function-name "$FN"
  aws lambda update-function-configuration --function-name "$FN" \
    --environment "Variables={LUMEN_KEY=$KEY}" >/dev/null
else
  aws lambda create-function --function-name "$FN" --runtime python3.12 --architectures x86_64 \
    --handler handler.lambda_handler --role "$ROLE_ARN" --memory-size 1024 --timeout 30 \
    --code S3Bucket="$BUCKET",S3Key=function.zip \
    --environment "Variables={LUMEN_KEY=$KEY}" >/dev/null
fi
aws lambda wait function-updated --function-name "$FN" 2>/dev/null || aws lambda wait function-active --function-name "$FN"

echo "== 5/5 Public URL (protected by the access key) =="
if ! aws lambda get-function-url-config --function-name "$FN" >/dev/null 2>&1; then
  aws lambda create-function-url-config --function-name "$FN" --auth-type NONE >/dev/null
  aws lambda add-permission --function-name "$FN" --statement-id url-invoke \
    --action lambda:InvokeFunctionUrl --principal "*" --function-url-auth-type NONE >/dev/null
fi
# Newer accounts also require lambda:InvokeFunction for a public function URL.
# Scope it to calls that arrive through the URL where the CLI supports that.
if ! aws lambda get-policy --function-name "$FN" --query Policy --output text 2>/dev/null | grep -q url-invoke-fn; then
  aws lambda add-permission --function-name "$FN" --statement-id url-invoke-fn \
    --action lambda:InvokeFunction --principal "*" --invoked-via-function-url >/dev/null 2>&1 \
  || aws lambda add-permission --function-name "$FN" --statement-id url-invoke-fn \
    --action lambda:InvokeFunction --principal "*" >/dev/null
fi
URL=$(aws lambda get-function-url-config --function-name "$FN" --query FunctionUrl --output text)

echo "== Test call =="
for i in 1 2 3 4 5 6; do   # permissions can take a few seconds to apply
  sleep 5
  RESP=$(curl -s -X POST "$URL" -H "content-type: application/json" -H "x-lumen-key: $KEY" \
         -d '{"session":"deploy-test","history":[]}' || true)
  echo "$RESP" | grep -q '"next"' && break
done
echo "$RESP"
if echo "$RESP" | grep -q '"next"'; then
  echo; echo "DEPLOYED. Use these on your laptop:"
else
  echo; echo "The function exists but the test call failed. Send the output above to Claude."
fi
echo "  URL: $URL"
echo "  KEY: $KEY"
echo "  python scripts/05_verify.py --cloud $URL --key $KEY"
