const express = require("express");
const cors = require("cors");
const fileUpload = require("express-fileupload");
const model = require("./controllers/tensorModel");
const ClientError = require("./exception/ClienError");

const PORT = process.env.PORT || 8080;
const app = express();

let isModelInitialize = false;
let modelTensor;

app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(fileUpload());
app.use(express.static("public"));

app.get("/", async (req, res) => {
    res.send("Hello world");
});

app.get("/predict/histories", async (req, res) => {
    model.getPredictHistories().then((data) => {
        res.json({
            status: "success",
            data,
        });
    });
});

app.post("/predict", async (req, res) => {
    try {
        if (req.files.image.size > 1 * 1000 * 1000) {
            throw new ClientError(
                "Payload content length greater than maximum allowed: 1000000",
                413
            );
        }

        let prediction = await model.predictClassification(
            modelTensor,
            req.files.image.data
        );
        console.log("Success prediction with id " + prediction.id);
        res.status(201).json({
            status: "success",
            message: "Model is predicted successfully",
            data: prediction,
        });
    } catch (err) {
        res.status(err.statusCode || 400).json({
            status: "fail",
            message: err.message,
        });
    }
});

app.listen(PORT, async () => {
    console.log("Model not initialized, Initializing model...");
    modelTensor = await model.loadModel();
    console.log("Model is ready");
    console.log(`Backend running on port : ${PORT}`);
});
