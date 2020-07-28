let state = "standby";
let stateElement;

let canvas;
let webcam;
let webcamImage;

let poseNameInput;
let standbyButton;
let recordPoseButton;
let classifyPoseButton;

let poseNet;
let pose;
let classifiedPoseElement;

let neuralNet;

function setup() {
	webcam = createCapture(VIDEO);
	webcam.hide();


	canvas = createCanvas(windowWidth, 500);

	stateElement = createElement("p");
	stateElement.html(state);
	stateElement.style("text-align", "center");
	stateElement.style("font-size", "30px");

	classifiedPoseElement = createElement("p");
	classifiedPoseElement.style("text-align", "center");
	classifiedPoseElement.style("font-size", "30px");
	classifiedPoseElement.html("Hellp");
	classifiedPoseElement.hide();

	poseNameInput = createInput("Pose Name");

	standbyButton = createButton("Stand By");
	standbyButton.mousePressed(function() {
		state = "standby";
		classifiedPoseElement.hide();
		console.log(state);
	});

	recordPoseButton = createButton("Record Pose");
	recordPoseButton.mousePressed(recordPose);

	classifyPoseButton = createButton("Classify Mode");
	classifyPoseButton.mousePressed(classifyPose);

	poseNet = ml5.poseNet(webcam, function() {
		console.log("PoseNet loaded");
	});
	poseNet.on("pose", getPoses);


	let neuralNetOptions = {
		inputs: 34,
		outputs: 1,
		task: "classification",
		debug: true
	}
	neuralNet = ml5.neuralNetwork(neuralNetOptions);
}

function draw() {
	background(30, 120, 160);

	translate(canvas.width, 0);
	scale(-1, 1);

	let webcamX = canvas.width / 2 - webcam.width / 2;
	image(webcam, webcamX, 0);

	stateElement.html(state);

	if(pose) {
		for(let i in pose) {
			if(pose[i].confidence > 0.90) {
				let x = pose[i].x + webcamX;
				let y = pose[i].y;

				ellipse(x, y, 10);
			}
		}
	}
}

function getPoses(result) {
	if(result[0]) {
		pose = result[0].pose;
		// console.log(pose);

		if(state == "recording" || state == "classify") {
			let inputs = [];
			let output = [poseNameInput.value()];

			for (let i = 0; i < pose.keypoints.length; i++) {
				let x = pose.keypoints[i].position.x;
				let y = pose.keypoints[i].position.y;
				inputs.push(x);
				inputs.push(y);
			}

			// inputs.push(pose.nose.x);
			// inputs.push(pose.nose.y);

			// inputs.push(pose.rightEye.x);
			// inputs.push(pose.rightEye.y);
			
			// inputs.push(pose.leftEye.x);
			// inputs.push(pose.leftEye.y);

			if(state == "recording") {
				neuralNet.addData(inputs, output);
			}
			else if(state == "classify") {
				neuralNet.predict(inputs, getClassifiedPose);
			}
		}
	}
}

function recordPose() {
	let count = 10;
	let timer = setInterval(function() {
		state = count;
		count--;

		if(count <= 0) {
			clearInterval(timer);
		}
	}, 1000);

	setTimeout(function() {
		state = "recording";
		console.log("recording");

		setTimeout(function() {
			state = "standby";
			console.log("standby");

			neuralNet.saveData(poseNameInput.value());
		}, 10000);
	}, 10000);
}

function classifyPose() {
	classifiedPoseElement.show();

	neuralNet.normalizeData();
	neuralNet.train({epochs: 100}, function() {
		console.log("training");
	}
	,function() {
		console.log("done training");
		state = "classify";
	});
}

function getClassifiedPose(error, result) {
	if(error) {
		console.log(error);
		return;
	}

	// console.log(result);
	console.log(result[0].label);

	classifiedPoseElement.html(result[0].label);
}