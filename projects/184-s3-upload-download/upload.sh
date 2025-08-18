#!/bin/bash

# Simple S3 Upload Script - just uploads upload-me.txt

# Load environment variables
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

aws s3 cp "upload-me.txt" "s3://$S3_BUCKET/upload-me.txt"

if [ $? -eq 0 ]; then
    echo "✅ Upload successful!"
    echo "File URL: https://$S3_BUCKET.s3.us-east-1.amazonaws.com/$S3_KEY"
else
    echo "❌ Upload failed!"
    exit 1
fi 