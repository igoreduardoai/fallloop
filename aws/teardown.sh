#!/usr/bin/env bash
# Deletes EVERYTHING this demo created in the AWS account (buckets are emptied first) so nothing keeps billing.
# Usage: bash aws/teardown.sh [stack-name] [region]
set -euo pipefail
STACK=${1:-fallloop-demo}; REGION=${2:-us-east-2}
out() { aws cloudformation describe-stacks --stack-name "$STACK" --region "$REGION" --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue" --output text 2>/dev/null || true; }
for KEY in InBucketName OutBucketName; do
  B=$(out $KEY)
  if [ -n "$B" ] && [ "$B" != "None" ]; then echo "Emptying s3://$B"; aws s3 rm "s3://$B" --recursive --region "$REGION" || true; fi
done
echo "Deleting stack $STACK (Lambda functions, buckets, budget)"
aws cloudformation delete-stack --stack-name "$STACK" --region "$REGION"
aws cloudformation wait stack-delete-complete --stack-name "$STACK" --region "$REGION"
echo "Leftover check (should be empty):"
aws cloudformation list-stacks --region "$REGION" --stack-status-filter CREATE_COMPLETE UPDATE_COMPLETE --query "StackSummaries[?contains(StackName,'fallloop')].StackName" --output text || true
aws ecr describe-repositories --region "$REGION" --query "repositories[].repositoryName" --output text 2>/dev/null || true
echo "Done. Also check Billing > Bills next day; the companion SAM bootstrap stack 'aws-sam-cli-managed-default' only holds an empty artifacts bucket and can be deleted the same way."
