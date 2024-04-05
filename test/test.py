"""
Tests YOLOv9 Lambda function URL

References:
+ Announcing AWS Lambda Function URLs: Built-in HTTPS Endpoints for Single-Function Microservices
https://aws.amazon.com/blogs/aws/announcing-aws-lambda-function-urls-built-in-https-endpoints-for-single-function-microservices/
+ Invoking Lambda function URLs
https://docs.aws.amazon.com/lambda/latest/dg/urls-invocation.html
"""

import base64
import json
import os
import sys

import cv2  # pylint: disable=import-error
import requests

from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.session import Session

# Get endpoint
url = sys.argv[1]

# Generate payload
img = sys.argv[2]
with open(img, 'rb') as f:
    img_b64 = base64.b64encode(f.read()).decode('ascii')
data = json.dumps({'image': img_b64}).encode('utf-8')

# Define headers
headers = {'Content-Type': 'application/json'}

# Create request
request = AWSRequest(
    method='GET',
    url=url,
    data=data,
    headers=headers
)
request.context['payload_signing_enabled'] = True

# Sign the request with Signature V4
sigv4 = SigV4Auth(
    Session().get_credentials(),
    'lambda',
    os.environ.get('AWS_DEFAULT_REGION', "us-east-1")
)
sigv4.add_auth(request)

# Make the request
prepped = request.prepare()
result = requests.get(
    prepped.url,
    data=data,
    headers=prepped.headers
)

# Extract detections
detections = result.json()['detections']
print(json.dumps(detections, indent=4))

# Display detections
img = cv2.imread(img)
for det in detections:
    x1, y1, x2, y2 = det['box']
    img = cv2.rectangle(
        img,
        (x1, y1),
        (x2, y2),
        (255, 0, 0),
        4
    )
cv2.imwrite("output.jpg", img)
