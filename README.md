# YOLambda: YOLO Inference with Serverless

## Overview

Learn how to run inference at scale with **any YOLO version** ([YOLOv5](https://github.com/ultralytics/yolov5), [YOLOv8/9/10/11](https://github.com/ultralytics/ultralytics), and future versions) in a secure and reliable way with [AWS Lambda](https://aws.amazon.com/lambda/) and [AWS SAM](https://aws.amazon.com/serverless/sam/).

✨ **Features:**
- 🔄 **Version-Agnostic**: Works with any YOLO version (past and future)
- 🚀 **Serverless**: Scales automatically with AWS Lambda
- 🔧 **Configurable**: Easy model swapping via environment variables
- 📦 **ONNX Optimized**: Fast inference with ONNX Runtime

<p>
	<img src="images/example.jpg" width="30%"/>
	>>>
	<img src="images/lambda.png" width="5%"/>
	>>>
	<img src="images/output.jpg" width="30%"/>
</p>

## Instructions

0. Install dependencies

	```bash
	# Create a new environment and activate it
	conda env create -f environment.yml
	conda activate yolambda

	# Install dependencies
	pip install -qr requirements.txt
	```

1. Convert your YOLO model to ONNX

	**For YOLOv8/v9/v10/v11 (Ultralytics):**
	```bash
	# Export PT -> ONNX (replace yolov11n.pt with your model)
	yolo mode=export model=yolov11n.pt format=onnx dynamic=True
	```

	**For YOLOv5:**
	```bash
	# Clone YOLOv5 repo if not already available
	git clone https://github.com/ultralytics/yolov5
	cd yolov5

	# Export PT -> ONNX
	python export.py --weights yolov5s.pt --include onnx
	```

	**Post-processing (Optional but recommended):**
	```bash
	# Simplify ONNX model
	# https://github.com/daquexian/onnx-simplifier
	onnxsim your_model.onnx your_model.onnx
	
	# Optimize ONNX model
	# https://github.com/onnx/optimizer
	python -m onnxoptimizer your_model.onnx your_model.onnx

	# Visualize model structure
	# 🌐 Browser: Visit https://netron.app/
	# 💻 CLI: netron -b your_model.onnx
	```

	**Setup for deployment:**
	```bash
	# Move model to the models folder
	mkdir -p models
	mv your_model.onnx models/yolo.onnx  # Rename to generic name
	```

2. Configure your deployment

	**Environment Variables (Optional):**
	
	You can customize the model path by setting environment variables in your SAM template or during deployment:
	
	```yaml
	# In template.yaml, add to your function's Environment section:
	Environment:
	  Variables:
	    YOLO_MODEL_PATH: /opt/your_custom_model.onnx  # Default: /opt/yolo.onnx
	```

3. Build and deploy the application

	```bash
	# 🏗️ Build
	sam build --use-container

	# 🚀 Deploy with custom parameters (optional)
	sam deploy --guided

	# OR deploy with specific model path
	sam deploy --parameter-overrides YoloModelPath=/opt/yolov8n.onnx

	# 📝 Note down the function URL
	export YOLAMBDA_URL=$(sam list stack-outputs --stack-name yolambda --output json | jq -r .[0].OutputValue)
	```

4. Test the application

	**Development:**

	Using [sam local](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/using-sam-cli-local.html)

	```bash
	# Create event
	echo {\"body\": \"{\\\"image\\\": \\\"$(base64 images/example.jpg)\\\"}\"} > test/event.json

	# Invoke function
	sam local invoke --event test/event.json
	```

	**Production:**

	Using [awscurl](https://github.com/okigan/awscurl.git)

	<!--
	Note: learned a lot by checking the `aws_curl.make_request` implementation
	https://github.com/okigan/awscurl/blob/master/awscurl/awscurl.py
	-->

	```bash
	# Create payload
	echo {\"image\": \"$(base64 images/example.jpg)\"} > test/payload.json

	# Make request
	awscurl --service lambda -X GET -d @test/payload.json $YOLAMBDA_URL

	# Pro tip: pretty-print the output by piping it to jq
	```

	or a custom Python script

	```bash
	python test/test.py $YOLAMBDA_URL images/example.jpg
	```

### 🛠️ Troubleshooting

**Model Loading Issues:**
- Ensure your ONNX model is properly exported
- Check that the model path is correct in the Lambda layer
- Verify the model is compatible with ONNX Runtime

**Inference Errors:**
- Make sure input image format is base64 encoded
- Check confidence and IoU thresholds are appropriate for your model
- Verify your model expects 640x640 input size (or adjust `imgsz` parameter)

**Performance Optimization:**
- Use ONNX model optimization tools for better performance
- Adjust Lambda memory allocation based on your model size
- Consider using ONNX Runtime execution providers for GPU acceleration (if available)
