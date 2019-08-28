$("#image-selector").change(function () {
    let reader = new FileReader();
    reader.onload = function () {
        let dataURL = reader.result;
        $("#selectedImage").attr("src", dataURL);
        $("#listOfPredictions").empty();
    }
    let file = $("#image-selector").prop("files")[0];
    reader.readAsDataURL(file);
});

$("#loadModal").click(function () {
    loadModel();
});

let model;
model = undefined;
async function loadModel() {
    $("#loadModal").html("Loading model...")
    $("#loadModal").attr("disabled", true);
    model = await mobilenet.load();
    $("#loadModal").html("Model is loaded")
}

$("#makePrediction").click(async function () {
    let image = $("#selectedImage").get(0);
    let tensor = tf.browser.fromPixels(image).resizeNearestNeighbor([224, 224]).toFloat(); //Mobilnet input size
    let offset = tf.scalar(127.5);//255/2 (imagenet was trained on images where RGB values were scaled down from 0-255 to -1 and 1)
    tensor = tensor.sub(offset).div(offset).expandDims(); //So we will also put all RGB values from scale -1 to 1  (We use broadcasting to do it fast)

    let predictions = await model.classify(tensor, 5);
    console.log(predictions)
    //Show top 5 predictions
    $("#listOfPredictions").append('1. ' + predictions[0].className + ' Probability: ' + predictions[0].probability.toFixed(2) + '%<br/>');
    $("#listOfPredictions").append('2. ' + predictions[1].className + ' Probability: ' + predictions[1].probability.toFixed(2) + '%<br/>');
    $("#listOfPredictions").append('3. ' + predictions[2].className + ' Probability: ' + predictions[2].probability.toFixed(2) + '%<br/>');
    $("#listOfPredictions").append('4. ' + predictions[3].className + ' Probability: ' + predictions[3].probability.toFixed(2) + '%<br/>');
    $("#listOfPredictions").append('5. ' + predictions[4].className + ' Probability: ' + predictions[4].probability.toFixed(2) + '%<br/>');
});

